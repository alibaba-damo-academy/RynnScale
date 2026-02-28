/*
 * Fused Rotary Position Embedding (RoPE) kernel using CuTe DSL.
 *
 * Applies RoPE to query and key tensors in a single fused kernel launch:
 *   q_out = q * cos + rotate_half(q) * sin
 *   k_out = k * cos + rotate_half(k) * sin
 *
 * where rotate_half(x) = concat(-x[..., head_dim/2:], x[..., :head_dim/2])
 *
 * Inputs:
 *   q   : [total_tokens, num_q_heads, head_dim]  – fp16 or bf16
 *   k   : [total_tokens, num_k_heads, head_dim]  – fp16 or bf16
 *   cos : [total_tokens, head_dim/2]             – float32
 *   sin : [total_tokens, head_dim/2]             – float32
 *
 * Supports bf16 and fp16 element types.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "cute/tensor.hpp"

using namespace cute;


// ---------------------------------------------------------------------------
// CUDA kernel: one CTA per (token, head) pair.
//
// CuTe layout view of the tensors:
//   Q/K     : make_layout(Shape<T, H, D>{}, GenRowMajor{})
//   cos/sin : make_layout(Shape<T, D/2>{}, GenRowMajor{})
//
// Each thread processes a stripe of HALF_DIM elements so that loads and
// stores are naturally coalesced for typical head-dim values (64/128/256).
// ---------------------------------------------------------------------------

template <typename Element, int HEAD_DIM>
__global__ void fused_rope_forward_kernel(
    const Element* __restrict__ q_ptr,
    const Element* __restrict__ k_ptr,
    const float*   __restrict__ cos_ptr,
    const float*   __restrict__ sin_ptr,
    Element*       __restrict__ q_out_ptr,
    Element*       __restrict__ k_out_ptr,
    int total_tokens,
    int num_q_heads,
    int num_k_heads
) {
    constexpr int HALF_DIM = HEAD_DIM / 2;

    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;

    if (token_idx >= total_tokens) return;

    // -----------------------------------------------------------------------
    // Build CuTe tensor views (row-major, packed) using make_tensor.
    //
    // Q  :  [total_tokens, num_q_heads, HEAD_DIM]
    //        stride = (num_q_heads * HEAD_DIM,  HEAD_DIM,  1)
    //
    // cos:  [total_tokens, HALF_DIM]
    //        stride = (HALF_DIM, 1)
    // -----------------------------------------------------------------------

    // Slice to the current (token, head) row for Q
    // Using pointer arithmetic consistent with the CuTe make_tensor API.
    const float* cos_row = cos_ptr + (int64_t)token_idx * HALF_DIM;
    const float* sin_row = sin_ptr + (int64_t)token_idx * HALF_DIM;

    // -------------------------------------------------------------------
    // Process Q
    // -------------------------------------------------------------------
    if (head_idx < num_q_heads) {
        // CuTe: make_tensor over a flat pointer with a 1-D layout for this row
        const Element* q_row  = q_ptr     + (int64_t)token_idx * num_q_heads * HEAD_DIM
                                           + (int64_t)head_idx  * HEAD_DIM;
        Element*       qo_row = q_out_ptr + (int64_t)token_idx * num_q_heads * HEAD_DIM
                                           + (int64_t)head_idx  * HEAD_DIM;

        // Construct CuTe tensors over the row (1-D, packed)
        auto q_frag  = make_tensor(make_gmem_ptr(q_row),  make_layout(Int<HEAD_DIM>{}));
        auto qo_frag = make_tensor(make_gmem_ptr(qo_row), make_layout(Int<HEAD_DIM>{}));

        // Each thread handles one element in [0, HALF_DIM); covers both halves
        CUTE_UNROLL
        for (int i = threadIdx.x; i < HALF_DIM; i += blockDim.x) {
            float c  = cos_row[i];
            float s  = sin_row[i];
            float x0 = float(q_frag(i));             // first half
            float x1 = float(q_frag(i + HALF_DIM));  // second half

            qo_frag(i)           = Element(x0 * c - x1 * s);
            qo_frag(i + HALF_DIM) = Element(x1 * c + x0 * s);
        }
    }

    // -------------------------------------------------------------------
    // Process K
    // -------------------------------------------------------------------
    if (head_idx < num_k_heads) {
        const Element* k_row  = k_ptr     + (int64_t)token_idx * num_k_heads * HEAD_DIM
                                           + (int64_t)head_idx  * HEAD_DIM;
        Element*       ko_row = k_out_ptr + (int64_t)token_idx * num_k_heads * HEAD_DIM
                                           + (int64_t)head_idx  * HEAD_DIM;

        auto k_frag  = make_tensor(make_gmem_ptr(k_row),  make_layout(Int<HEAD_DIM>{}));
        auto ko_frag = make_tensor(make_gmem_ptr(ko_row), make_layout(Int<HEAD_DIM>{}));

        CUTE_UNROLL
        for (int i = threadIdx.x; i < HALF_DIM; i += blockDim.x) {
            float c  = cos_row[i];
            float s  = sin_row[i];
            float x0 = float(k_frag(i));
            float x1 = float(k_frag(i + HALF_DIM));

            ko_frag(i)            = Element(x0 * c - x1 * s);
            ko_frag(i + HALF_DIM) = Element(x1 * c + x0 * s);
        }
    }
}


// ---------------------------------------------------------------------------
// Dispatch helpers
// ---------------------------------------------------------------------------

#define DISPATCH_ROPE_ELEMENT_TYPE(AT_TYPE, ...)                             \
    switch (AT_TYPE) {                                                       \
        case at::kHalf:                                                      \
        { using Element = at::Half; __VA_ARGS__ }                            \
        break;                                                               \
        case at::kBFloat16:                                                  \
        { using Element = at::BFloat16; __VA_ARGS__ }                        \
        break;                                                               \
        default:                                                             \
        TORCH_CHECK(false, "fused_rope: unsupported dtype. Use fp16 or bf16"); \
    }


template <typename Element>
static void launch_fused_rope(
    at::Tensor& q,
    at::Tensor& k,
    at::Tensor& cos,
    at::Tensor& sin,
    at::Tensor& q_out,
    at::Tensor& k_out,
    int total_tokens,
    int num_q_heads,
    int num_k_heads,
    int head_dim,
    cudaStream_t stream
) {
    const int max_heads = std::max(num_q_heads, num_k_heads);
    const int half_dim  = head_dim / 2;
    // Each CTA owns one (token, head) pair; threads stride across half_dim.
    const int threads   = std::min(half_dim, 256);
    const dim3 grid(total_tokens, max_heads);

    auto launch = [&](auto hd_const) {
        constexpr int HD = decltype(hd_const)::value;
        fused_rope_forward_kernel<Element, HD><<<grid, threads, 0, stream>>>(
            reinterpret_cast<const Element*>(q.data_ptr()),
            reinterpret_cast<const Element*>(k.data_ptr()),
            cos.data_ptr<float>(),
            sin.data_ptr<float>(),
            reinterpret_cast<Element*>(q_out.data_ptr()),
            reinterpret_cast<Element*>(k_out.data_ptr()),
            total_tokens,
            num_q_heads,
            num_k_heads
        );
    };

    if      (head_dim ==  64) { launch(cute::Int<64>{});  }
    else if (head_dim == 128) { launch(cute::Int<128>{}); }
    else if (head_dim == 256) { launch(cute::Int<256>{}); }
    else {
        TORCH_CHECK(false, "fused_rope: unsupported head_dim=", head_dim,
                    ". Supported: 64, 128, 256");
    }
}


// ---------------------------------------------------------------------------
// Public C++ entry point
// ---------------------------------------------------------------------------

void fused_rope_forward(
    at::Tensor q,      // [total_tokens, num_q_heads, head_dim]  fp16/bf16
    at::Tensor k,      // [total_tokens, num_k_heads, head_dim]  fp16/bf16
    at::Tensor cos,    // [total_tokens, head_dim/2]             float32
    at::Tensor sin,    // [total_tokens, head_dim/2]             float32
    at::Tensor q_out,  // same shape as q
    at::Tensor k_out   // same shape as k
) {
    TORCH_CHECK(q.dim() == 3, "q must be 3-D [total_tokens, num_q_heads, head_dim]");
    TORCH_CHECK(k.dim() == 3, "k must be 3-D [total_tokens, num_k_heads, head_dim]");
    TORCH_CHECK(cos.dim() == 2, "cos must be 2-D [total_tokens, head_dim/2]");
    TORCH_CHECK(sin.dim() == 2, "sin must be 2-D [total_tokens, head_dim/2]");

    const int total_tokens = q.size(0);
    const int num_q_heads  = q.size(1);
    const int num_k_heads  = k.size(1);
    const int head_dim     = q.size(2);
    const int half_dim     = head_dim / 2;

    TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even");
    TORCH_CHECK(k.size(0) == total_tokens, "q and k must have the same number of tokens");
    TORCH_CHECK(k.size(2) == head_dim,     "q and k must have the same head_dim");
    TORCH_CHECK(cos.size(0) == total_tokens && cos.size(1) == half_dim,
                "cos shape must be [total_tokens, head_dim/2]");
    TORCH_CHECK(sin.size(0) == total_tokens && sin.size(1) == half_dim,
                "sin shape must be [total_tokens, head_dim/2]");
    TORCH_CHECK(cos.scalar_type() == at::kFloat, "cos must be float32");
    TORCH_CHECK(sin.scalar_type() == at::kFloat, "sin must be float32");
    TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q and k must share the same dtype");
    TORCH_CHECK(q_out.sizes() == q.sizes(), "q_out shape must match q");
    TORCH_CHECK(k_out.sizes() == k.sizes(), "k_out shape must match k");

    TORCH_CHECK(q.is_contiguous(),   "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(),   "k must be contiguous");
    TORCH_CHECK(cos.is_contiguous(), "cos must be contiguous");
    TORCH_CHECK(sin.is_contiguous(), "sin must be contiguous");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_ROPE_ELEMENT_TYPE(q.scalar_type(),
        launch_fused_rope<Element>(
            q, k, cos, sin, q_out, k_out,
            total_tokens, num_q_heads, num_k_heads, head_dim, stream
        );
    );
}
