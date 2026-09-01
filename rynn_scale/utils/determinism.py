import torch
from transformers.trainer_utils import enable_full_determinism as _enable_full_determinism
from transformers.trainer_utils import set_seed as _set_seed


def set_seed(seed: int, full_determinism: bool = False) -> None:
    if full_determinism:
        _enable_full_determinism(seed)
        # use_deterministic_algorithms(True) re-enables fill_uninitialized_memory,
        # whose torch.empty() fill kernel races with DeepEP comm streams. Disable
        # it after every enable_full_determinism call.
        torch.utils.deterministic.fill_uninitialized_memory = False
    else:
        _set_seed(seed)
