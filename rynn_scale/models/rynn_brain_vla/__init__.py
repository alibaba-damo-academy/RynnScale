from .configuration_rynn_brain_vla import RynnBrainVLAConfig
from .modeling_rynn_brain_vla import RynnBrainVLAModel
from .processing_rynn_brain_vla import RynnBrainVLAProcessor


def apply_monkey_patch():
    from transformers import CONFIG_MAPPING, MODEL_MAPPING, PROCESSOR_MAPPING

    CONFIG_MAPPING.register("rynn_brain_vla", RynnBrainVLAConfig, exist_ok=True)
    MODEL_MAPPING.register(RynnBrainVLAConfig, RynnBrainVLAModel, exist_ok=True)
    PROCESSOR_MAPPING.register(RynnBrainVLAConfig, RynnBrainVLAProcessor, exist_ok=True)
