from typing import List

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, ResNetModel, VisualBertModel

from src.model import register_model


@register_model(name="halloc_text")
class HallocTextModel(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        vlm_out_dim: int,
    ) -> None:
        super().__init__()
        hidden_dim = 768
        visual_embedding_dim = 2048

        self.img_seq_len = 1

        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        self.vision_encoder = ResNetModel.from_pretrained("microsoft/resnet-50")
        self.vision_proj = nn.Linear(
            2048 * 7 * 7 // self.img_seq_len,
            visual_embedding_dim,
        )

        self.multimodal_encoder = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

        self.obj_head = nn.Linear(hidden_dim, 2)
        # nn.init.constant_(self.obj_head.bias, class_bias)
        self.att_head = nn.Linear(hidden_dim, 2)
        # nn.init.constant_(self.att_head.bias, class_bias)
        self.rel_head = nn.Linear(hidden_dim, 2)
        # nn.init.constant_(self.rel_head.bias, class_bias)
        self.sce_head = nn.Linear(hidden_dim, 2)
        # nn.init.constant_(self.sce_head.bias, class_bias)
        self.oth_head = nn.Linear(hidden_dim, 2)
        # nn.init.constant_(self.oth_head.bias, class_bias)
        self.all_head = nn.Linear(hidden_dim, 2)
        # nn.init.constant_(self.all_head.bias, class_bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_masks: torch.Tensor,
        token_type_ids: torch.Tensor,
        images: List[Image.Image],
        is_all: bool = False,
    ) -> torch.Tensor:
        visual_inputs = self.image_processor(images, return_tensors="pt")
        visual_inputs = {k: v.to(input_ids.device) for k, v in visual_inputs.items()}
        visual_outputs = self.vision_encoder(**visual_inputs)
        visual_last_hidden_states = visual_outputs.last_hidden_state  # (batch_size, 2048, 7, 7)

        visual_embeddings = self.vision_proj(
            visual_last_hidden_states.view(visual_last_hidden_states.size(0), self.img_seq_len, -1)
        )  # (batch_size, img_seq_len, visual_embedding_dim)

        visual_token_type_ids = torch.ones(
            visual_embeddings.shape[:-1],
            dtype=torch.long,
            device=visual_embeddings.device
        )  # (batch_size, img_seq_len)

        visual_attention_mask = torch.ones(
            visual_embeddings.shape[:-1],
            dtype=torch.long,
            device=visual_embeddings.device,
        )  # (batch_size, img_seq_len)

        out_embeddings = self.multimodal_encoder(
            input_ids=input_ids,  # (batch_size, seq_len)
            attention_mask=attention_masks,  # (batch_size, seq_len)
            token_type_ids=token_type_ids,  # (batch_size, seq_len)
            visual_embeds=visual_embeddings,  # (batch_size, img_seq_len, visual_embedding_dim) = (64, 1, 512)
            visual_token_type_ids=visual_token_type_ids,  # (batch_size, img_seq_len) = (64, 1)
            visual_attention_mask=visual_attention_mask,  # (batch_size, img_seq_len) = (64, 1)
        ).last_hidden_state

        out_embeddings = out_embeddings[:, 1:-2, :]

        if is_all:
            all_logits = self.all_head(out_embeddings)
            return all_logits
        else:
            obj_logits = self.obj_head(out_embeddings)
            att_logits = self.att_head(out_embeddings)
            rel_logits = self.rel_head(out_embeddings)
            sce_logits = self.sce_head(out_embeddings)
            oth_logits = self.oth_head(out_embeddings)
            return obj_logits, att_logits, rel_logits, sce_logits, oth_logits
