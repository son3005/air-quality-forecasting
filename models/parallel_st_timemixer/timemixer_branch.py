"""
═══════════════════════════════════════════════════════════════
TimeMixer Branch v2 — Multi-Scale Temporal Decomposition
═══════════════════════════════════════════════════════════════
Cải tiến so với v1:
  1. PastDecomMixing giữ lại feature dimension (không mean(dim=-1))
     → dùng transpose + Linear projection trên trục T
  2. Thêm residual connection trong từng scale mixer
  3. Dùng Attention-weighted scale fusion thay vì concat thô
═══════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeriesDecomposition(nn.Module):
    """Tách chuỗi thời gian thành Trend + Seasonal bằng Moving Average."""
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        assert kernel_size % 2 == 1, f"kernel_size phải lẻ, nhận: {kernel_size}"
        self.avg_pool = nn.AvgPool1d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )

    def forward(self, x: torch.Tensor):
        # x: [B, T, d]
        x_permuted = x.permute(0, 2, 1)  # [B, d, T]
        trend = self.avg_pool(x_permuted)
        trend = trend[:, :, :x.shape[1]]
        trend = trend.permute(0, 2, 1)   # [B, T, d]
        seasonal = x - trend
        return trend, seasonal


def multiscale_downsample(x: torch.Tensor, scales: list):
    """Tạo các bản sao dữ liệu ở nhiều thang đo thời gian khác nhau."""
    result = {}
    for s in scales:
        if s == 1:
            result[s] = x
        else:
            x_t = x.permute(0, 2, 1)  # [B, d, T]
            x_down = F.avg_pool1d(x_t, kernel_size=s, stride=s)
            result[s] = x_down.permute(0, 2, 1)  # [B, T//s, d]
    return result


class PastDecomMixing(nn.Module):
    """
    MLP Mixing trên trục thời gian cho 1 scale cụ thể.
    
    v2: Giữ nguyên feature dimension, project T -> d_model riêng cho
    trend và seasonal, rồi merge. Có residual connection.
    """
    def __init__(self, seq_len: int, d_model: int, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        # Trend: project T dim -> d_ff -> 1 (pooling learned)
        self.trend_time_proj = nn.Sequential(
            nn.Linear(seq_len, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1)
        )
        # Seasonal: project T dim -> d_ff -> 1
        self.seasonal_time_proj = nn.Sequential(
            nn.Linear(seq_len, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1)
        )
        # Feature refinement after merging trend + seasonal
        self.feat_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, trend: torch.Tensor, seasonal: torch.Tensor) -> torch.Tensor:
        # trend, seasonal: [B*N, T, d_model]
        # Transpose to [B*N, d_model, T] for projecting the T dimension
        trend_t = trend.permute(0, 2, 1)        # [B*N, d_model, T]
        seasonal_t = seasonal.permute(0, 2, 1)  # [B*N, d_model, T]
        
        # Project T -> 1, then squeeze: [B*N, d_model]
        z_trend = self.trend_time_proj(trend_t).squeeze(-1)          # [B*N, d_model]
        z_seasonal = self.seasonal_time_proj(seasonal_t).squeeze(-1) # [B*N, d_model]
        
        # Merge trend + seasonal feature vectors
        z = self.feat_proj(torch.cat([z_trend, z_seasonal], dim=-1))  # [B*N, d_model]
        return self.norm(z)


class TimeMixerBranch(nn.Module):
    """
    Nhánh TimeMixer v2: Multi-scale temporal modeling.
    Cải tiến:
      1. PastDecomMixing giữ feature dim (không mean collapse)
      2. Attention-weighted scale fusion
      3. Residual global: input_proj mean-pool + scale_fused
    """
    def __init__(
        self, seq_len: int, num_features: int, d_model: int,
        scales: list = None, d_ff: int = 256, dropout: float = 0.1,
        decomp_kernel: int = 25
    ):
        super().__init__()
        self.scales = scales or [1, 4, 24]
        self.seq_len = seq_len
        self.d_model = d_model
        num_scales = len(self.scales)

        # Input projection: F -> d_model
        self.input_proj = nn.Linear(num_features, d_model)
        self.decomp = SeriesDecomposition(kernel_size=decomp_kernel)
        
        # Per-scale mixers
        self.past_mixers = nn.ModuleDict()
        for s in self.scales:
            T_s = seq_len // s if s > 1 else seq_len
            self.past_mixers[str(s)] = PastDecomMixing(
                seq_len=T_s, d_model=d_model, d_ff=d_ff, dropout=dropout
            )

        # Attention-weighted scale fusion
        self.scale_attn = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1)
        )
        
        # Final projection with residual
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)
        
        # Residual from raw input (global mean-pool T)
        self.residual_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*N, T, F]
        x_proj = self.input_proj(x)  # [B*N, T, d_model]
        x_scales = multiscale_downsample(x_proj, self.scales)

        scale_outputs = []
        for s in self.scales:
            x_s = x_scales[s]
            trend_s, seasonal_s = self.decomp(x_s)
            z_s = self.past_mixers[str(s)](trend_s, seasonal_s)  # [B*N, d_model]
            scale_outputs.append(z_s)

        # Stack: [B*N, num_scales, d_model]
        z_stack = torch.stack(scale_outputs, dim=1)
        
        # Attention-weighted fusion: learn which scales matter
        attn_scores = self.scale_attn(z_stack)  # [B*N, num_scales, 1]
        attn_weights = F.softmax(attn_scores, dim=1)
        z_fused = (attn_weights * z_stack).sum(dim=1)  # [B*N, d_model]

        # Residual from raw input (mean-pool over T)
        residual = self.residual_proj(x_proj.mean(dim=1))  # [B*N, d_model]
        
        z_out = self.out_proj(z_fused) + residual
        return self.norm(z_out)
