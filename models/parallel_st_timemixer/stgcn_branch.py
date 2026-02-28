"""
═══════════════════════════════════════════════════════════════
STGCN Branch v2 — Spatial-Temporal Graph Convolution
═══════════════════════════════════════════════════════════════
Cải tiến so với v1:
  1. Thay flatten T*d_model (9216 dim!) bằng Temporal Attention Pooling
     → giảm mạnh overfitting, giữ thông tin quan trọng
  2. Thêm Skip/Residual connection giữa các ST-Block
  3. Multi-kernel TCN (kernel 3 + kernel 7) cho cả short & long patterns
═══════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChebConv(nn.Module):
    """Chebyshev Graph Convolution Layer."""
    def __init__(self, d_in: int, d_out: int, K: int = 3, dropout: float = 0.2):
        super().__init__()
        self.K = K
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(d_in, d_out)) for _ in range(K + 1)
        ])
        self.bias = nn.Parameter(torch.zeros(d_out))
        self.norm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self._init_weights()

    def _init_weights(self):
        for w in self.weights:
            nn.init.xavier_uniform_(w)

    def _chebyshev_polynomials(self, A: torch.Tensor) -> list:
        N = A.shape[0]
        T_0 = torch.eye(N, device=A.device, dtype=A.dtype)
        T_1 = A
        polys = [T_0, T_1]
        for k in range(2, self.K + 1):
            T_k = 2.0 * A @ polys[-1] - polys[-2]
            polys.append(T_k)
        return polys[:self.K + 1]

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        polys = self._chebyshev_polynomials(A)
        out = torch.zeros(
            H.shape[0], H.shape[1], self.weights[0].shape[1],
            device=H.device, dtype=H.dtype
        )
        for k in range(self.K + 1):
            T_k = polys[k]
            W_k = self.weights[k]
            agg = torch.einsum('nm,bmd->bnd', T_k, H)
            out = out + agg @ W_k
        out = out + self.bias
        out = self.activation(out)
        out = self.dropout(out)
        return self.norm(out)


class GatedActivation(nn.Module):
    """Gated Linear Unit for TCN."""
    def forward(self, x):
        d = x.shape[1] // 2
        return x[:, :d, :] * torch.sigmoid(x[:, d:, :])


class STBlock(nn.Module):
    """
    Spatial-Temporal Block v2: Multi-kernel TCN -> Graph Conv
    Cải tiến: dùng 2 kernel (short + long) temporal convolution.
    """
    def __init__(self, d_model: int, K: int = 3, dropout: float = 0.2):
        super().__init__()
        # Short-range temporal (kernel=3)
        self.tcn_short = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=3, padding=1),
            GatedActivation(),
            nn.Dropout(dropout)
        )
        # Long-range temporal (kernel=7)
        self.tcn_long = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=7, padding=3),
            GatedActivation(),
            nn.Dropout(dropout)
        )
        # Merge short + long
        self.temporal_merge = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Spatial: ChebConv
        self.graph_conv = ChebConv(d_model, d_model, K=K, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        B, N, T, C = x.shape

        # --- Multi-Kernel Temporal Pass ---
        x_flat = x.reshape(B * N, T, C).transpose(1, 2)  # [B*N, C, T]
        x_short = self.tcn_short(x_flat).transpose(1, 2)  # [B*N, T, C]
        x_long = self.tcn_long(x_flat).transpose(1, 2)    # [B*N, T, C]
        
        # Merge: concat then project
        x_merged = self.temporal_merge(
            torch.cat([x_short, x_long], dim=-1)  # [B*N, T, 2C]
        )  # [B*N, T, C]
        
        # Residual + Norm
        x = self.norm1(x + x_merged.reshape(B, N, T, C))

        # --- Spatial Graph Conv ---
        x_s = x.permute(0, 2, 1, 3).reshape(B * T, N, C)  # [B*T, N, C]
        x_s = self.graph_conv(x_s, A)                      # [B*T, N, C]
        x_s = x_s.reshape(B, T, N, C).permute(0, 2, 1, 3)  # [B, N, T, C]
        x = self.norm2(x + x_s)

        return x


class TemporalAttentionPool(nn.Module):
    """
    Attention-weighted pooling over temporal dimension.
    Thay thế cho flatten T*d → d (tiết kiệm tham số, tránh overfitting).
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, T, d]
        scores = self.attn(x)                          # [B, N, T, 1]
        weights = F.softmax(scores, dim=2)             # [B, N, T, 1]
        pooled = (weights * x).sum(dim=2)              # [B, N, d]
        return self.dropout(pooled)


class STGCNBranch(nn.Module):
    """
    STGCN Branch v2: Spatial-Temporal Graph Learning.
    
    Cải tiến:
      1. TemporalAttentionPool thay thế flatten T*d_model
      2. Multi-kernel TCN (short + long)
      3. Skip connection từ input
    """
    def __init__(self, seq_len: int, num_features: int, d_model: int,
                 K: int = 3, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(num_features, d_model)

        self.st_blocks = nn.ModuleList([
            STBlock(d_model, K=K, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Temporal Attention Pooling thay flatten
        self.temporal_pool = TemporalAttentionPool(d_model, dropout=dropout)
        
        # Skip connection from input mean-pool
        self.skip_proj = nn.Linear(d_model, d_model)
        
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: [B, N, T, F]
        B, N, T, F = x.shape
        x_emb = self.input_proj(x)  # [B, N, T, d_model]

        h = x_emb
        for block in self.st_blocks:
            h = block(h, A)  # [B, N, T, d_model]

        # Temporal Attention Pooling: [B, N, T, d] -> [B, N, d]
        z_st = self.temporal_pool(h)
        
        # Skip residual from input (mean-pool over T)
        skip = self.skip_proj(x_emb.mean(dim=2))  # [B, N, d]
        
        return self.out_norm(z_st + skip)
