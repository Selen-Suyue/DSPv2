import os
import torch
import torchvision

from torch import nn
from einops import rearrange
from transformers import AutoModel
from peft import LoraConfig, get_peft_model
from torchvision.models._utils import IntermediateLayerGetter
from torch.nn import functional as F

class DINOv3Encoder(nn.Module):
    """DINOv2 backbone with optional LoRA fine-tuning."""
    def __init__(
        self, 
        name: str = "dinov3-base", 
        out_dim: int = 512,
        finetune: str = "lora", 
        dtype = torch.float32,
        lora_rank: int = 16, 
        lora_dropout: float = 0.1
    ):
        super().__init__()
        assert finetune in ["full", "lora", "none"], "finetune parameter should be one of [full, lora, none]."
        
        dino = AutoModel.from_pretrained(os.path.join("./weights", name), torch_dtype = dtype)

        if finetune == "lora":
            dino.requires_grad_(False)
            config = LoraConfig(
                r              = lora_rank,
                lora_alpha     = lora_rank,
                target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj'],
                lora_dropout   = lora_dropout,
                bias           = 'none',
                use_rslora     = True,
            )
            dino = get_peft_model(dino, config)
            for name, param in dino.named_parameters():
                if "lora_" in name:
                    param.data = param.data.float()
        elif finetune == "none":
            dino.requires_grad_(False)
            self.model = dino
        
        self.model = dino

        self.patch_size = dino.config.patch_size
        hidden_size = dino.config.hidden_size
        if hidden_size != out_dim:
            self.proj = nn.Linear(hidden_size, out_dim)
        else:
            self.proj = nn.Identity()
        self.num_channels = out_dim

    def forward(self, img):
        feats = self.model(img).last_hidden_state[:, 5:]
        feats = self.proj(feats)    # B, L, num_channels
        return feats

class DINOv2Encoder(nn.Module):
    """DINOv2 backbone with optional LoRA fine-tuning."""
    def __init__(
        self, 
        name: str = "dinov2-base", 
        out_dim: int = 512,
        finetune: str = "lora", 
        dtype = torch.float32,
        lora_rank: int = 16, 
        lora_dropout: float = 0.1
    ):
        super().__init__()
        assert finetune in ["full", "lora", "none"], "finetune parameter should be one of [full, lora, none]."
        
        dino = AutoModel.from_pretrained(os.path.join("./weights", name), torch_dtype = dtype)

        if finetune == "lora":
            dino.requires_grad_(False)
            config = LoraConfig(
                r              = lora_rank,
                lora_alpha     = lora_rank,
                target_modules = ['projection', 'query', 'key', 'value', 'dense', 'fc1', 'fc2'],
                lora_dropout   = lora_dropout,
                bias           = 'none',
                use_rslora     = True,
            )
            dino = get_peft_model(dino, config)
            for name, param in dino.named_parameters():
                if "lora_" in name:
                    param.data = param.data.float()
        elif finetune == "none":
            dino.requires_grad_(False)
            self.model = dino
        
        self.model = dino

        self.patch_size = dino.config.patch_size
        hidden_size = dino.config.hidden_size
        if hidden_size != out_dim:
            self.proj = nn.Linear(hidden_size, out_dim)
        else:
            self.proj = nn.Identity()
        self.num_channels = out_dim

    def forward(self, img):
        feats = self.model(img).last_hidden_state[:, 1:]
        feats = self.proj(feats)    # B, L, num_channels
        return feats


 
