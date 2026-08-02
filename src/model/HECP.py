from typing import List, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
from transformers import AutoImageProcessor, CLIPVisionModel
from src.model import register_model


class HiddenStateDynamicsAnalyzer(nn.Module):
    def __init__(self, hidden_dim, num_layers=4, probe_dim=512):
        super().__init__()
        self.layer_projs = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, probe_dim), nn.LayerNorm(probe_dim), nn.GELU())
            for _ in range(num_layers)
        ])
        
        self.evolution_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=probe_dim, nhead=8, dim_feedforward=probe_dim*2, batch_first=True),
            num_layers=2
        )
        
        self.layer_query = nn.Parameter(torch.randn(1, 1, probe_dim))
        self.layer_attn = nn.MultiheadAttention(probe_dim, num_heads=8, batch_first=True)

    def forward(self, hidden_states, padding_mask=None):
        B, L, S, D = hidden_states.shape 
        
        projected = torch.stack([self.layer_projs[i](hidden_states[:, i, :, :]) for i in range(L)], dim=1)
        diffs = projected[:, 1:, :, :] - projected[:, :-1, :, :]
        evolution_seq = torch.cat([projected, diffs], dim=1)
        seq_len_evo = evolution_seq.shape[1]
        
        token_evo_seq = evolution_seq.permute(0, 2, 1, 3).reshape(B * S, seq_len_evo, -1)
        encoded_evo = self.evolution_encoder(token_evo_seq)
        
        query = self.layer_query.expand(B * S, -1, -1)
        aggregated, _ = self.layer_attn(query, encoded_evo, encoded_evo)
        out = aggregated.reshape(B, S, -1)
        
        if padding_mask is not None:
            out = out.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            
        return out

class CrossSignalFusionTransformer(nn.Module):
    def __init__(self, hidden_dim=768, num_signals=2, num_layers=3, num_heads=8):
        super().__init__()
        self.num_signals = num_signals
        self.signal_type_embed = nn.Embedding(num_signals, hidden_dim)
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'cross_signal_attn': nn.TransformerEncoderLayer(
                    d_model=hidden_dim, nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    batch_first=True, activation='gelu'
                ),
                'inter_token_attn': nn.TransformerEncoderLayer(
                    d_model=hidden_dim, nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    batch_first=True, activation='gelu'
                ),
            }))
        
        self.signal_fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_signals, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, signal_features, padding_mask=None):
        B, S, D = signal_features[0].shape
        N = len(signal_features) 
        
        stacked = torch.stack(signal_features, dim=2)
        
        signal_ids = torch.arange(N, device=stacked.device).unsqueeze(0).expand(B * S, -1)
        type_embeds = self.signal_type_embed(signal_ids)  
        
        x = stacked.reshape(B * S, N, D) + type_embeds  
        
        for layer in self.layers:
            x = layer['cross_signal_attn'](x)  
            
            x_reshaped = x.reshape(B, S, N, D).permute(0, 2, 1, 3)  
            x_reshaped = x_reshaped.reshape(B * N, S, D)
            
            expanded_mask = None
            if padding_mask is not None:
                expanded_mask = padding_mask.unsqueeze(1).expand(-1, N, -1).reshape(B * N, S)
                
            x_reshaped = layer['inter_token_attn'](
                x_reshaped, 
                src_key_padding_mask=expanded_mask
            )  
            
            x = x_reshaped.reshape(B, N, S, D).permute(0, 2, 1, 3).reshape(B * S, N, D)
        
        x = x.reshape(B, S, N, D)
        fused = self.signal_fusion(x.reshape(B, S, N * D))  
        
        if padding_mask is not None:
            fused = fused.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            
        return fused

class MultiScaleTemporalAggregation(nn.Module):

    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.local_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.mid_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3, groups=hidden_dim)
        self.global_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)

        self.scale_gate = nn.Sequential(
            nn.Sequential(nn.Linear(hidden_dim*3, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        B, S, D = x.shape

        # 局部 / 中等 / 全局
        xt = x.transpose(1, 2)
        local = self.local_conv(xt).transpose(1, 2)
        mid = self.mid_conv(xt).transpose(1, 2)
        global_feat, _ = self.global_attn(x, x, x, key_padding_mask=padding_mask)

        multi = torch.cat([local, mid, global_feat], dim=-1)
        fused = self.scale_gate(multi)
        out = fused + x
        out = self.output_norm(out)

        if padding_mask is not None:
            out = out.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return out


@register_model(name="hecp")
class HECP(nn.Module):

    def __init__(self, hidden_dim: int = 512, vlm_out_dim: int = 4096,
                 num_layers: int = 4, 
                 vlm_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
                 **kwargs):
        super().__init__()

        self.ahda = HiddenStateDynamicsAnalyzer(vlm_out_dim, num_layers, hidden_dim)
        
        self.visual_seq_len = 50 

        self.image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_hidden_size = 768

        self.text_encoder = AutoModel.from_pretrained(
            vlm_model_name,
            trust_remote_code=True,
            attn_implementation="eager",
            torch_dtype="auto" 
        )
            
        self.llm_hidden_dim = self.text_encoder.config.hidden_size
   
        self.encoder_dtype = self.text_encoder.dtype

        self.visual_proj = nn.Linear(clip_hidden_size, self.llm_hidden_dim)

        self.fusion_out_proj = nn.Linear(self.llm_hidden_dim, hidden_dim)

        self.visual_proj = self.visual_proj.to(self.encoder_dtype)
        self.fusion_out_proj = self.fusion_out_proj.to(self.encoder_dtype)
    
        self.csft = CrossSignalFusionTransformer(hidden_dim, num_signals=2)
        self.mta = MultiScaleTemporalAggregation(hidden_dim)

        self.all_head = nn.Linear(hidden_dim, 2)
        

    def forward(self,
        input_ids: torch.Tensor,
        embeddings: torch.Tensor,
        attention_masks: torch.Tensor,
        token_type_ids: torch.Tensor,
        images: List[Image.Image],
        is_all: bool = False,
    ):
        device = embeddings.device
        B = embeddings.shape[0]
        input_ids = input_ids.to(device)

        attention_masks = attention_masks.to(device)
        token_type_ids = token_type_ids.to(device)

        padding_mask = (attention_masks == 0)

        hidden_feat = self.ahda(embeddings.float(), padding_mask=padding_mask)

        visual_inputs = self.image_processor(images, return_tensors="pt")
        visual_inputs = {k: v.to(device) for k, v in visual_inputs.items()}
        visual_outputs = self.vision_encoder(**visual_inputs)
        visual_last_hidden = visual_outputs.last_hidden_state  

        visual_last_hidden = visual_last_hidden.to(self.encoder_dtype)
        visual_embeds = self.visual_proj(visual_last_hidden)  
        visual_attn_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.long, device=device)

   
        text_embeds = self.text_encoder.get_input_embeddings()(input_ids)  # [B, S, llm_hidden_dim]

        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)  # [B, 50+S, D]
        combined_mask = torch.cat([visual_attn_mask, attention_masks], dim=1)  # [B, 50+S]

        fusion_output = self.text_encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            is_causal=False,       # 禁用因果掩码 → 双向注意力
            output_hidden_states=False,
            use_cache=False
        )
        fusion_hidden = fusion_output.last_hidden_state  # [B, 50+S, llm_hidden_dim]

        text_fusion_feat = fusion_hidden[:, self.visual_seq_len:, :]  # [B, S_text, llm_hidden_dim]
        visual_embeddings = self.fusion_out_proj(text_fusion_feat).float()  # [B, S_text, hidden_dim]
   
        signal_features = [hidden_feat, visual_embeddings]
        fused = self.csft(signal_features, padding_mask=padding_mask)  
        fused = self.mta(fused, padding_mask=padding_mask)

        all_logits = self.all_head(fused)

        return all_logits