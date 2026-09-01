from transformers.models.qwen3_vl import Qwen3VLConfig


class RynnBrainVLAConfig(Qwen3VLConfig):
    model_type = "rynn_brain_vla"

    def __init__(
        self,
        action_dim=6,
        action_chunk_size=20,
        state_token_id=-1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.state_token_id = state_token_id
