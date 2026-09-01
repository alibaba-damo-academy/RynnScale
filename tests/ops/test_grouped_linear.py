from functools import partial
from typing import List

import torch

from rynn_scale.ops.grouped_linear import grouped_linear

from .utils import benchmark, check_consistency


def generate_data(
    input_group_sizes: List[int],
    hidden_size: int = 4096,
    num_experts: int = 128,
    moe_intermediate_size: int = 1536,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    hidden_states = torch.rand((sum(input_group_sizes), hidden_size), generator=generator, dtype=dtype, device="cuda")
    weight = torch.rand(
        (num_experts * moe_intermediate_size, hidden_size), generator=generator, dtype=dtype, device="cuda"
    )
    hidden_states.requires_grad_(True)
    weight.requires_grad_(True)
    return {
        "input": hidden_states,
        "weight": weight,
        "input_group_sizes": input_group_sizes,
    }


def test_grouped_linear(seed: int = 42):
    generator = torch.Generator().manual_seed(seed)

    mean_tokens = 16384 // 16
    input_group_sizes = torch.randint(
        round(mean_tokens * 0.8), round(mean_tokens * 1.2), (128,), generator=generator
    ).tolist()

    data_generator = partial(
        generate_data,
        input_group_sizes=input_group_sizes,
        hidden_size=4096,
        num_experts=128,
        moe_intermediate_size=1536,
        dtype=torch.bfloat16,
    )

    ref_inputs, ref_outputs, ref_grads, ref_times, ref_peak_memory = benchmark(
        ops=grouped_linear,
        data_generator=data_generator,
        backend="torch",
        seed=seed,
        num_repeats=20,
    )
    ref_outputs = {"output": ref_outputs}

    inputs, outputs, grads, times, peak_memory = benchmark(
        ops=grouped_linear,
        data_generator=data_generator,
        backend="cutlass",
        seed=seed,
        num_repeats=20,
    )
    outputs = {"output": outputs}

    print("*" * 80)
    print(f"Seed: {seed}")
    print(f"Reference Times: {ref_times}")
    print(f"Times: {times}")
    print(f"Reference Peak Memory: {ref_peak_memory}")
    print(f"Peak Memory: {peak_memory}")

    check_consistency(outputs, ref_outputs)
    check_consistency(grads, ref_grads)
