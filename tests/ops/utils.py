import gc
import time
from typing import Any, Callable, Dict

import torch
from tqdm import tqdm


def benchmark(
    ops: Callable,
    data_generator: Callable[Any, Dict[str, Any]],
    backend: str,
    seed: int = 42,
    num_repeats: int = 20,
):
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    start_memory = torch.cuda.memory_allocated()

    torch.cuda.reset_max_memory_allocated()

    times = {
        "forward": [],
        "backward": [],
    }

    for i in tqdm(range(num_repeats)):
        inputs = data_generator(seed=seed)

        torch.cuda.synchronize()
        start_time = time.time()

        outputs = ops(**inputs, backend=backend)

        torch.cuda.synchronize()
        times["forward"].append(time.time() - start_time)

        if isinstance(outputs, tuple):
            outputs_require_grad = [output for output in outputs if torch.is_tensor(output) and output.requires_grad]
        else:
            outputs_require_grad = [outputs]
        assert len(outputs_require_grad) > 0

        start_time = time.time()
        outputs_require_grad[0].backward(outputs_require_grad[0])

        torch.cuda.synchronize()
        times["backward"].append(time.time() - start_time)

        if i == 0:
            peak_memory = (torch.cuda.max_memory_allocated() - start_memory) / 1e6
            grads = {k: v.grad for k, v in inputs.items() if torch.is_tensor(v) and v.requires_grad}

    times = {k: sum(v) / len(v) for k, v in times.items()}
    times["total"] = sum(times.values())

    return inputs, outputs, grads, times, peak_memory


def check_consistency(
    outputs: Dict[str, torch.Tensor],
    ref_outputs: Dict[str, torch.Tensor],
):
    assert outputs.keys() == ref_outputs.keys()
    for key in outputs:
        passed = torch.allclose(outputs[key], ref_outputs[key])
        if passed:
            print(f"Test <{key}> passed")
        else:
            print(f"Test <{key}> failed, summary:")
            print("Ref Output:", ref_outputs[key], sep="\n")
            print("Output:", outputs[key], sep="\n")
            diff = torch.abs(outputs[key] - ref_outputs[key])
            print("Diff:", diff, sep="\n")
            print(f"Max Diff: {diff.amax()}, Mean Diff: {diff.mean()}")
        print("\n")
