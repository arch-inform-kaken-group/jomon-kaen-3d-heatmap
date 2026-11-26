from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointSpatialAffectDecoder(nn.Module):
    """
    Cross-attends every 3D pottery point [B, N, D] to the predicted JEPA spatial tokens
    [B, Q, D] to decode:
      - gaze_logits:    [B, N, 1] (normalized via softmax/sigmoid over surface)
      - emotion_logits: [B, N, 5] (5 canonical affective categories)
    """

    def __init__(self, latent_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
        )
        self.gaze_head = nn.Linear(latent_dim // 2, 1)
        self.emotion_head = nn.Linear(latent_dim // 2, 5)
        # Initialize emotion logits to sparse prior (~7.5% active) to prevent uniform saturation
        nn.init.constant_(self.emotion_head.bias, -2.5)

    def forward(
        self, point_feats: torch.Tensor, pred_spatial_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, _ = self.cross_attn(
            query=self.norm1(point_feats),
            key=pred_spatial_tokens,
            value=pred_spatial_tokens,
        )
        fused = torch.cat([point_feats, attn_out], dim=-1)
        h = self.mlp(fused)
        gaze_logits = self.gaze_head(h)
        emotion_logits = self.emotion_head(h)
        return gaze_logits, emotion_logits

    @staticmethod
    def sample_outputs(
        gaze_logits: torch.Tensor,
        emotion_logits: torch.Tensor,
        temperature: float = 0.7,
        stochastic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        temp = max(temperature, 1e-3)
        gaze_probs = torch.sigmoid(gaze_logits / temp)
        gaze_probs = gaze_probs / (gaze_probs.amax(dim=1, keepdim=True) + 1e-6)

        # Keep spatial emotion maps coherent across neighboring points (no per-point Bernoulli salt-and-pepper)
        emo_probs = torch.sigmoid(emotion_logits / temp)
        return gaze_probs, emo_probs