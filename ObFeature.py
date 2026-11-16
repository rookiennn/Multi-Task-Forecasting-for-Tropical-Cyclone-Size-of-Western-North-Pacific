import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuadrantSelfAttention(nn.Module):
    def __init__(self, in_dim=6, d_model=64, num_heads=4):
        super().__init__()

        self.embed = nn.Linear(in_dim, d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.embed(x)
        attn_out, attn_weights = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        return x

class QuadrantAttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x):
        w = torch.softmax(self.score(x), dim=1)
        return torch.sum(w * x, dim=1)

class PerTimeStepEncoder(nn.Module):
    def __init__(self, env_dim, d_model=128):
        super().__init__()

        self.scale_encoder = QuadrantSelfAttention(
            in_dim=6, d_model=64, num_heads=4
        )

        self.quad_pool = QuadrantAttentionPooling(64)

        self.env_proj = nn.Linear(env_dim, 64)

        self.fusion = nn.Linear(128, d_model)

    def forward(self, scale_feat, env_feat):

        scale_out = self.scale_encoder(scale_feat)
        scale_vec = self.quad_pool(scale_out)

        env_vec = self.env_proj(env_feat)

        h = torch.cat([scale_vec, env_vec], dim=-1)
        return h

class TemporalConvBlock(nn.Module):
    def __init__(self, d_model, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            d_model, d_model,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        out = self.conv(x)
        out = out[:, :, :-self.padding]
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = out.transpose(1, 2)
        return out + residual
class TemporalEncoder(nn.Module):
    def __init__(self, d_model, num_layers=3, kernel_size=3):
        super().__init__()
        self.blocks = nn.ModuleList([
            TemporalConvBlock(
                d_model,
                kernel_size=kernel_size,
                dilation=2**i
            )
            for i in range(num_layers)
        ])

    def forward(self, x):
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        return x.transpose(1, 2)
class TemporalAttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, x):

        weights = torch.softmax(self.attn(x), dim=1)
        return torch.sum(weights * x, dim=1)
class CycloneFeatureExtractor(nn.Module):
    def __init__(self, env_dim, d_model=128):
        super().__init__()

        self.per_time_encoder = PerTimeStepEncoder(
            env_dim=env_dim,
            d_model=d_model
        )

        self.temporal_encoder = TemporalEncoder(d_model)
        self.temporal_pool = TemporalAttentionPooling(d_model)

    def forward(self, addition):


        arr_reshaped = addition.view(addition.shape[0], 9, 37)

        B, T, F = arr_reshaped.shape

        part1 = torch.cat([arr_reshaped[:, :, 0:12], arr_reshaped[:, :, 15:27]], dim=2)

        quadrant_indices = [
            [0, 4, 8, 12, 13, 14],  # Q1
            [1, 5, 9, 15, 16, 17],  # Q2
            [2, 6, 10, 18, 19, 20],  # Q3
            [3, 7, 11, 21, 22, 23]  # Q4
        ]

        arr_quadrants = torch.stack([part1[:, :, idx] for idx in quadrant_indices], dim=2)  # (B, 9, 4, 6)
        scale_feats = arr_quadrants


        env_feats = torch.cat([arr_reshaped[:, :, 12:15], arr_reshaped[:, :, 27:37]], dim=2)


        per_time_out = []
        for t in range(T):
            h_t = self.per_time_encoder(
                scale_feats[:, t],  # (B, 4, 6)
                env_feats[:, t]  # (B, env_dim)
            )
            per_time_out.append(h_t)


        H = torch.stack(per_time_out, dim=1)  # (B, T, d_model)
        H = self.temporal_encoder(H)


        Z = self.temporal_pool(H)  # (B, d_model)

        return Z


