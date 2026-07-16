from typing import Any, List, Dict

from rynn_scale.datasets.vlm_datasets import VLMDataset
from rynn_scale.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class RynnBrainDataset(VLMDataset):
    def _is_rynnbrain_data(self, conversation):
        for message in conversation:
            for content in message["content"]:
                if content["type"] == "text" and "<frame " in content["text"]:
                    return True
        return False

    def _convert_conversation(self, conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        conversation = super()._convert_conversation(conversation)
        if not self._is_rynnbrain_data(conversation):
            return conversation

        new_conversation = []
        for message in conversation:
            if message["role"] == "user":
                image_idx = 0
                new_contents = []
                for i, content in enumerate(message["content"]):
                    if content["type"] == "image":
                        last_content = message["content"][i - 1] if i > 0 else content
                        if last_content["type"] != "text" or "<frame " not in last_content["text"]:
                            new_contents.append({"type": "text", "text": f"<frame {image_idx}>: "})
                            image_idx += 1
                        else:
                            continue
                    new_contents.append(content)
                new_conversation.append({"role": message["role"], "content": new_contents})

            elif message["role"] == "assistant":
                new_contents = []
                for content in message["content"]:
                    if content["type"] == "text":
                        new_contents.append(content)
                new_conversation.append({"role": message["role"], "content": new_contents})

            else:
                new_conversation.append(message)

        return new_conversation
