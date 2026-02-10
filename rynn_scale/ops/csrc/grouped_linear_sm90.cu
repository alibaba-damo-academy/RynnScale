#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"


#define DISPATCH_CUTLASS_TYPES(AT_TYPE, ...)                     \
    switch (AT_TYPE) {                                           \
        case at::kHalf:                                          \
        { using Element = cutlass::half_t; __VA_ARGS__ }         \
        break;                                                   \
        case at::kBFloat16:                                      \
        { using Element = cutlass::bfloat16_t; __VA_ARGS__ }     \
        break;                                                   \
        default:                                                 \
        TORCH_CHECK(false, "Unsupported dtype");                 \
    }


using ElementAccumulator = float;


struct BaseKernelConfig {
    using ArchTag = cutlass::arch::Sm90;
    using OpClass = cutlass::arch::OpClassTensorOp;
};


struct ForwardKernelConfig : BaseKernelConfig {
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using TileShape = cute::Shape<cute::_128, cute::_256, cute::_64>;
    using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
};


struct BackwardDxKernelConfig : BaseKernelConfig {
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using TileShape = cute::Shape<cute::_128, cute::_128, cute::_64>;
    using ClusterShape = cute::Shape<cute::_1, cute::_2, cute::_1>;
    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
};


struct BackwardDwKernelConfig : BaseKernelConfig {
    using LayoutA = cutlass::layout::ColumnMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using TileShape = cute::Shape<cute::_256, cute::_128, cute::_64>;
    using ClusterShape = cute::Shape<cute::_1, cute::_2, cute::_1>;
    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
};


using ProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;


template <typename Config, typename Element> struct GemmBuilderGivenConfig {
    using ElementA = Element;
    using ElementB = Element;
    using ElementC = Element;

    using LayoutA = typename Config::LayoutA;
    using LayoutB = typename Config::LayoutB;
    using LayoutC = typename Config::LayoutC;

    using ArchTag = typename Config::ArchTag;
    using OpClass = typename Config::OpClass;
    using TileShape = typename Config::TileShape;
    using ClusterShape = typename Config::ClusterShape;
    using KernelSchedule = typename Config::KernelSchedule;
    using EpilogueSchedule = typename Config::EpilogueSchedule;

    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OpClass,
        TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutC *, AlignmentC,
        ElementC, LayoutC *, AlignmentC,
        EpilogueSchedule,
        cutlass::epilogue::fusion::LinearCombination<ElementC, ElementAccumulator>
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OpClass,
        ElementA, LayoutA *, AlignmentA,
        ElementB, LayoutB *, AlignmentB,
        ElementAccumulator,
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
        >,
        KernelSchedule
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        ProblemShape,
        CollectiveMainloop,
        CollectiveEpilogue
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};


template <typename Config, typename Element>
void grouped_gemm_packed(
    Element* A,
    Element* B,
    Element* D,
    const std::vector<int> &ms,
    const std::vector<int> &ns,
    const std::vector<int> &ks,
    const int &A_split_dim,
    const int &B_split_dim,
    const int &D_split_dim,
    const int &device_id
) {
    using GemmBuilder = GemmBuilderGivenConfig<Config, Element>;
    using GemmKernel = typename GemmBuilder::GemmKernel;
    using Gemm = typename GemmBuilder::Gemm;

    using StrideA = typename Gemm::GemmKernel::InternalStrideA;
    using StrideB = typename Gemm::GemmKernel::InternalStrideB;
    using StrideC = typename Gemm::GemmKernel::InternalStrideC;
    using StrideD = typename Gemm::GemmKernel::InternalStrideD;

    using LayoutA = typename Config::LayoutA;
    using LayoutB = typename Config::LayoutB;
    using LayoutC = typename Config::LayoutC;

    std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;

    std::vector<StrideA> stride_A_host;
    std::vector<StrideB> stride_B_host;
    std::vector<StrideC> stride_C_host;
    std::vector<StrideD> stride_D_host;

	std::vector<Element *> ptr_A_host;
    std::vector<Element *> ptr_B_host;
    std::vector<Element *> ptr_C_host;
    std::vector<Element *> ptr_D_host;

    Element* cur_ptr_A = A;
    Element* cur_ptr_B = B;
    Element* cur_ptr_C = D;
    Element* cur_ptr_D = D;

    int i, m, n, k;

    for (i = 0; i < ms.size(); i++) {
        m = ms[i], n = ns[i], k = ks[i];

        if (m > 0 && n > 0) {
            problem_sizes_host.push_back({m, n, k});

            stride_A_host.push_back(cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1}));
            stride_B_host.push_back(cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1}));
            stride_C_host.push_back(cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1}));
            stride_D_host.push_back(cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1}));

            ptr_A_host.push_back(cur_ptr_A);
            ptr_B_host.push_back(cur_ptr_B);
            ptr_C_host.push_back(cur_ptr_C);
            ptr_D_host.push_back(cur_ptr_D);
        }

        int A_shape[] = {m, k}, B_shape[] = {k, n}, C_shape[] = {m, n};
        int A_offset[] = {0, 0}, B_offset[] = {0, 0}, C_offset[] = {0, 0};

        A_offset[A_split_dim] = A_shape[A_split_dim];
        B_offset[B_split_dim] = B_shape[B_split_dim];
        C_offset[D_split_dim] = C_shape[D_split_dim];

        LayoutA layout_A = LayoutA::packed({A_shape[0], A_shape[1]});
        LayoutB layout_B = LayoutB::packed({B_shape[0], B_shape[1]});
        LayoutC layout_C = LayoutC::packed({C_shape[0], C_shape[1]});

        cur_ptr_A += layout_A({A_offset[0], A_offset[1]});
        cur_ptr_B += layout_B({B_offset[0], B_offset[1]});
        cur_ptr_C += layout_C({C_offset[0], C_offset[1]});
        cur_ptr_D += layout_C({C_offset[0], C_offset[1]});
    }

    int problem_count = problem_sizes_host.size();
	TORCH_CHECK(problem_count > 0, "No valid GEMM problems to solve.");

    cutlass::DeviceAllocation<StrideA> stride_A;
    cutlass::DeviceAllocation<StrideB> stride_B;
    cutlass::DeviceAllocation<StrideC> stride_C;
    cutlass::DeviceAllocation<StrideD> stride_D;

    stride_A.reset(problem_count);
    stride_B.reset(problem_count);
    stride_C.reset(problem_count);
    stride_D.reset(problem_count);
    stride_A.copy_from_host(stride_A_host.data());
    stride_B.copy_from_host(stride_B_host.data());
    stride_C.copy_from_host(stride_C_host.data());
    stride_D.copy_from_host(stride_D_host.data());

    cutlass::DeviceAllocation<const typename Gemm::ElementA *> ptr_A;
    cutlass::DeviceAllocation<const typename Gemm::ElementB *> ptr_B;
    cutlass::DeviceAllocation<const typename Gemm::ElementC *> ptr_C;
    cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput *> ptr_D;

    ptr_A.reset(problem_count);
    ptr_B.reset(problem_count);
    ptr_C.reset(problem_count);
    ptr_D.reset(problem_count);
    ptr_A.copy_from_host(ptr_A_host.data());
    ptr_B.copy_from_host(ptr_B_host.data());
    ptr_C.copy_from_host(ptr_C_host.data());
    ptr_D.copy_from_host(ptr_D_host.data());

    cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;
    problem_sizes.reset(problem_count);
    problem_sizes.copy_from_host(problem_sizes_host.data());

    cutlass::KernelHardwareInfo kernel_hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel>(device_id);

    typename Gemm::Arguments args;
    decltype(args.epilogue.thread) fusion_args;

    fusion_args.alpha = 1.0f;
    fusion_args.beta = 0.0f;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array = nullptr;
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
    fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};

    args = typename Gemm::Arguments {
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {problem_count, problem_sizes.get(), problem_sizes_host.data()},
        {ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get()},
        {fusion_args, ptr_C.get(), stride_C.get(), ptr_D.get(), stride_D.get()},
        kernel_hw_info
    };

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(args);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    Gemm gemm;

    // Check if the problem size is supported or not
    cutlass::Status status = gemm.can_implement(args);
	TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to initialize CUTLASS grouped GEMM");

    // Initialize CUTLASS kernel with arguments and workspace pointer
    status = gemm.initialize(args, workspace.get());
	TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to initialize CUTLASS grouped GEMM");

    status = gemm.run();
    TORCH_CHECK(status == cutlass::Status::kSuccess, std::string("Failed to run CUTLASS grouped GEMM: ") + cudaGetErrorString(cudaDeviceSynchronize()));
}


void grouped_linear_forward(
	at::Tensor input,
	at::Tensor weight,
	at::Tensor output,
	const std::vector<int> &input_group_sizes
) {
    int num_groups = input_group_sizes.size();
    int out_dim = weight.size(0);
    int group_out_dim = out_dim / num_groups;
    int total_input = std::accumulate(input_group_sizes.begin(), input_group_sizes.end(), int64_t{0});
    std::vector<int> weight_group_sizes(num_groups, group_out_dim);

    TORCH_CHECK(input.scalar_type() == weight.scalar_type() && input.scalar_type() == output.scalar_type(),
        "All tensors must share the same dtype."
    );
    TORCH_CHECK(input.size(1) == weight.size(1), "Input dimensions of input and weight must match.");
    TORCH_CHECK(input.size(0) == output.size(0), "Output rows must match input rows.");
    TORCH_CHECK(output.size(1) == group_out_dim, "Output columns must match weight rows.");

    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous.");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous.");
    TORCH_CHECK(output.is_contiguous(), "Output tensor must be contiguous.");

    TORCH_CHECK(total_input == input.size(0), "Sum of input group sizes must equal input batch size.");
    TORCH_CHECK(out_dim % num_groups == 0, "Output dimension must be divisible by number of groups.");

    const int device_id = input.get_device();

    DISPATCH_CUTLASS_TYPES(input.scalar_type(),
        grouped_gemm_packed<ForwardKernelConfig, Element>(
            static_cast<Element *>(input.data_ptr()),
            static_cast<Element *>(weight.data_ptr()),
            static_cast<Element *>(output.data_ptr()),
            input_group_sizes,
            weight_group_sizes,
            std::vector<int>(num_groups, input.size(1)),
            0, 1, 0,
            device_id
        );
    );
}


void grouped_linear_backward_dx(
	at::Tensor input,
	at::Tensor weight,
    at::Tensor grad_output,
	at::Tensor grad_input,
	const std::vector<int> &input_group_sizes
) {
    int num_groups = input_group_sizes.size();
    int out_dim = weight.size(0);
    int group_out_dim = out_dim / num_groups;
    int total_input = std::accumulate(input_group_sizes.begin(), input_group_sizes.end(), int64_t{0});
    std::vector<int> weight_group_sizes(num_groups, group_out_dim);

    TORCH_CHECK(input.scalar_type() == weight.scalar_type() &&
        input.scalar_type() == grad_output.scalar_type() &&
        input.scalar_type() == grad_input.scalar_type(),
        "All tensors must share the same dtype."
    );

    TORCH_CHECK(input.size(1) == weight.size(1), "Input dimensions of input and weight must match.");
    TORCH_CHECK(input.size(0) == grad_output.size(0), "Output rows must match input rows.");
    TORCH_CHECK(grad_output.size(1) == group_out_dim, "Output columns must match weight rows.");
    TORCH_CHECK(grad_input.sizes() == input.sizes(), "grad_input must match input shape.");

    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous.");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous.");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output tensor must be contiguous.");
    TORCH_CHECK(grad_input.is_contiguous(), "grad_input tensor must be contiguous.");

    TORCH_CHECK(total_input == input.size(0), "Sum of input group sizes must equal input batch size.");
    TORCH_CHECK(out_dim % num_groups == 0, "Output dimension must be divisible by number of groups.");

    const int device_id = input.get_device();

    DISPATCH_CUTLASS_TYPES(input.scalar_type(),
        grouped_gemm_packed<BackwardDxKernelConfig, Element>(
            static_cast<Element *>(grad_output.data_ptr()),
            static_cast<Element *>(weight.data_ptr()),
            static_cast<Element *>(grad_input.data_ptr()),
            input_group_sizes,
            std::vector<int>(num_groups, weight.size(1)),
            weight_group_sizes,
            0, 0, 0,
            device_id
        );
    )
}


void grouped_linear_backward_dw(
	at::Tensor input,
	at::Tensor weight,
    at::Tensor grad_output,
	at::Tensor grad_weight,
	const std::vector<int> &input_group_sizes
) {
    int num_groups = input_group_sizes.size();
    int out_dim = weight.size(0);
    int group_out_dim = out_dim / num_groups;
    int total_input = std::accumulate(input_group_sizes.begin(), input_group_sizes.end(), int64_t{0});
    std::vector<int> weight_group_sizes(num_groups, group_out_dim);

    TORCH_CHECK(input.scalar_type() == weight.scalar_type() &&
        input.scalar_type() == grad_output.scalar_type() &&
        input.scalar_type() == grad_weight.scalar_type(),
        "All tensors must share the same dtype."
    );

    TORCH_CHECK(input.size(1) == weight.size(1), "Input dimensions of input and weight must match.");
    TORCH_CHECK(input.size(0) == grad_output.size(0), "Output rows must match input rows.");
    TORCH_CHECK(grad_output.size(1) == group_out_dim, "Output columns must match weight rows.");
    TORCH_CHECK(grad_weight.sizes() == weight.sizes(), "grad_weight must match weight shape.");

    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous.");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous.");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output tensor must be contiguous.");
    TORCH_CHECK(grad_weight.is_contiguous(), "grad_weight tensor must be contiguous.");

    TORCH_CHECK(total_input == input.size(0), "Sum of input group sizes must equal input batch size.");
    TORCH_CHECK(out_dim % num_groups == 0, "Output dimension must be divisible by number of groups.");

    const int device_id = input.get_device();

    DISPATCH_CUTLASS_TYPES(input.scalar_type(),
        grouped_gemm_packed<BackwardDwKernelConfig, Element>(
            static_cast<Element *>(grad_output.data_ptr()),
            static_cast<Element *>(input.data_ptr()),
            static_cast<Element *>(grad_weight.data_ptr()),
            weight_group_sizes,
            std::vector<int>(num_groups, weight.size(1)),
            input_group_sizes,
            1, 0, 0,
            device_id
        );
    );
}
