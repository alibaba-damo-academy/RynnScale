from functools import partial

import torch

from rynn_scale.ops.moe_permutation import (
    moe_token_permute,
)

from .utils import benchmark, check_consistency


def generate_data(
    num_experts: int = 128,
    num_experts_per_token: int = 8,
    hidden_size: int = 4096,
    num_tokens: int = 16384,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 123,
):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    num_routed_experts = torch.randint(
        1,
        num_experts_per_token + 1,
        (num_tokens,),
        generator=generator,
        dtype=torch.int32,
        device="cuda",
    )
    routed_expert_indices = torch.rand(
        num_tokens,
        num_experts,
        generator=generator,
        device="cuda",
    ).argsort(dim=1)[:, :num_experts_per_token]

    for i in range(num_tokens):
        routed_expert_indices[i, num_routed_experts[i] :] = -1
    routed_expert_indices = routed_expert_indices[
        :, torch.randperm(num_experts_per_token, generator=generator, device="cuda")
    ]

    num_routed_tokens = torch.bincount(
        routed_expert_indices[routed_expert_indices >= 0], minlength=num_experts
    ).tolist()

    hidden_states = torch.rand(num_tokens, hidden_size, generator=generator, dtype=dtype, device="cuda") * 0.5 - 0.25
    probs = torch.rand(num_tokens, num_experts_per_token, generator=generator, dtype=dtype, device="cuda")

    return hidden_states, probs, routed_expert_indices, num_routed_tokens


def generate_data_permute(
    num_experts: int = 128,
    num_experts_per_token: int = 8,
    hidden_size: int = 4096,
    num_tokens: int = 16384,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 123,
):
    hidden_states, _, routed_expert_indices, num_routed_tokens = generate_data(
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        hidden_size=hidden_size,
        num_tokens=num_tokens,
        dtype=dtype,
        seed=seed,
    )
    hidden_states.requires_grad_(True)
    return {
        "hidden_states": hidden_states,
        "routed_expert_indices": routed_expert_indices,
        "num_routed_tokens": num_routed_tokens,
        "num_experts_per_token": num_experts_per_token,
    }


def generate_data_unpermute(
    num_experts: int = 128,
    num_experts_per_token: int = 8,
    hidden_size: int = 4096,
    num_tokens: int = 16384,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 123,
):
    hidden_states, probs, routed_expert_indices, num_routed_tokens = generate_data(
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        hidden_size=hidden_size,
        num_tokens=num_tokens,
        dtype=dtype,
        seed=seed,
    )
    permuted_tokens, permuted_indices = moe_token_permute(
        hidden_states,
        routed_expert_indices=routed_expert_indices,
        num_routed_tokens=num_routed_tokens,
        num_experts_per_token=num_experts_per_token,
    )
    permuted_tokens.requires_grad_(True)
    probs.requires_grad_(True)
    return {
        "permuted_tokens": permuted_tokens,
        "probs": probs,
        "permuted_indices": permuted_indices,
        "num_experts_per_token": num_experts_per_token,
    }


def test_moe_token_permute(seed: int = 42):
    data_generator = partial(
        generate_data_permute,
        num_experts=128,
        num_experts_per_token=8,
        hidden_size=4096,
        num_tokens=16384,
        dtype=torch.bfloat16,
    )

    ref_inputs, ref_outputs, ref_grads, ref_times, ref_peak_memory = benchmark(
        ops=moe_token_permute,
        data_generator=data_generator,
        backend="torch",
        seed=seed,
        num_repeats=20,
    )
    ref_outputs = {
        "permuted_tokens": ref_outputs[0],
    }

    inputs, outputs, grads, times, peak_memory = benchmark(
        ops=moe_token_permute,
        data_generator=data_generator,
        backend="triton",
        seed=seed,
        num_repeats=20,
    )
    outputs = {
        "permuted_tokens": outputs[0],
    }

    print("*" * 80)
    print(f"Seed: {seed}")
    print(f"Reference Times: {ref_times}")
    print(f"Times: {times}")
    print(f"Reference Peak Memory: {ref_peak_memory}")
    print(f"Peak Memory: {peak_memory}")

    check_consistency(outputs, ref_outputs)
    check_consistency(grads, ref_grads)
