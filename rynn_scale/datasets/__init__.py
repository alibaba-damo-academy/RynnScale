from ..constants import RotationRepresentation
from ..registry import DATASET_REGISTRY
from .vla_datasets import BaseVLADataset
from .vlm_datasets import VLMDataset
from .wrappers import ConcatDataset, StreamingVLADataset


def _build_dataset(
    data_type: str,
    data_path: str,
    model_max_length: int,
    mm_max_length: int,
    fps: int,
    max_frames: int,
    action_chunk_size: int,
    use_delta_action: bool,
    eef_rotation_repr: RotationRepresentation,
    action_only: bool,
    seed: int = 0,
    target_fps=None,
    **kwargs,
):
    dataset_class = DATASET_REGISTRY[data_type]
    if issubclass(dataset_class, VLMDataset):
        return dataset_class(
            data_path=data_path,
            model_max_length=model_max_length,
            mm_max_length=mm_max_length,
            fps=fps,
            max_frames=max_frames,
            seed=seed,
            **kwargs,
        )
    elif BaseVLADataset is not None and issubclass(dataset_class, BaseVLADataset):
        return dataset_class(
            data_path=data_path,
            action_chunk_size=action_chunk_size,
            use_delta_action=use_delta_action,
            eef_rotation_repr=eef_rotation_repr,
            action_only=action_only,
            target_fps=target_fps,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown dataset type: {data_type}")


def build_dataset(args):
    defaults = {
        # VLM processing configs
        "model_max_length": args.model_max_length,
        "mm_max_length": args.mm_max_length,
        "fps": args.fps,
        "max_frames": args.max_frames,
        "seed": args.seed,
        # VLA processing configs
        "action_chunk_size": args.action_chunk_size,
        "use_delta_action": args.use_delta_action,
        "eef_rotation_repr": args.eef_rotation_repr,
        "action_only": args.action_only,
        "target_fps": args.target_fps,
    }

    if args.data_mixture is None:
        data_mixture = [
            {"data_type": args.data_type, "data_path": args.data_path},
        ]
    else:
        data_mixture = args.data_mixture

    datasets = []
    for data_source in data_mixture:
        assert "data_type" in data_source and "data_path" in data_source
        datasets.append(_build_dataset(**{**defaults, **data_source}))

    if getattr(args, "use_episode_iterator", False):
        vla_datasets = [ds for ds in datasets if isinstance(ds, BaseVLADataset)]
        assert len(vla_datasets) == len(datasets), "episode iterator mode only supports VLA datasets"
        return StreamingVLADataset(datasets=vla_datasets)

    if len(datasets) == 1:
        return datasets[0]

    return ConcatDataset(datasets)
