import torch
import torch.nn as nn
from torch.nn import functional as F
from policy.transformer import Transformer, QFormer
from policy.tokenizer import Sparse3DEncoder
from policy.dense import DensePolicy
from policy.v_model import DINOv2Encoder, DINOv3Encoder

class dspv2(nn.Module):
    def __init__(
        self, 
        Tp = 16, 
        Ta = 16, 
        input_dim = 3,
        obs_feature_dim = 512, 
        action_dim = 33, 
        hidden_dim = 512,
        nheads = 8, 
        num_encoder_layers = 4, 
        num_decoder_layers = 1, 
        dim_feedforward = 2048, 
        dropout = 0.1,
        num_views = 4,
        dino_version = "v3",
    ):
        super().__init__()
        self.sparse_encoder = Sparse3DEncoder(input_dim, obs_feature_dim)
        self.transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.action_decoder = DensePolicy(action_dim, 
                                                  Tp, 
                                                  Ta, 
                                                  obs_feature_dim,
                                                )
        self.readout_embed = nn.Embedding(1, hidden_dim)
        if dino_version == "v3":
            self.v_enc = DINOv3Encoder(out_dim=obs_feature_dim, finetune="lora", dtype=torch.float32)
        else:
            self.v_enc = DINOv2Encoder(out_dim=obs_feature_dim, finetune="lora", dtype=torch.float32)

        num_visual_tokens = num_views * (224 // self.v_enc.patch_size) ** 2  # Img_size // patch size in dino
        self.visual_pos_embed = nn.Parameter(torch.randn(1, num_visual_tokens, hidden_dim))
        self.Query = QFormer(hidden_dim, nheads, dim_feedforward)

    def forward(self, cloud, actions = None, qpos=None, imgs=None, batch_size = 24):
        
        bs, cams, c, h, w = imgs.shape
        imgs_reshaped = imgs.view(bs * cams, c, h, w)
        visual_feats_flat = self.v_enc(imgs_reshaped) 
        _, l, d = visual_feats_flat.shape
        visual_src = visual_feats_flat.view(bs, cams * l, d) 
        visual_pos = self.visual_pos_embed.repeat(bs, 1, 1)
        visual_src = visual_src + visual_pos

        src, pos, src_padding_mask = self.sparse_encoder(cloud, batch_size=batch_size)
        attn_output = self.Query(visual_features=visual_src, query_pos_embed=pos) 
        src = src + attn_output 

        readout = self.transformer(src, src_padding_mask, self.readout_embed.weight, pos)[-1]
        readout = readout[:, 0]

        if actions is not None:
                loss = self.action_decoder.compute_loss(readout, actions, qpos)
                return loss
        else:
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(readout, qpos)
            return action_pred

