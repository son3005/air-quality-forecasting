"""
=============================================================================
Module: ST-TimeMixer — Mô hình hoàn chỉnh (v3)
=============================================================================
Đây là file CHÍNH của kiến trúc ST-TimeMixer, tích hợp tất cả component
vào một file duy nhất.

Cấu trúc:
  - SeriesDecomposition, multiscale_downsample (Tách Trend/Seasonal, Downsample)
  - PastDecomMixing, TimeMixerModule (Phase 1: TimeMixer Branch)
  - ChebGCNLayer, SpatialGCNBlock (Phase 2: GCN Branch)
  - ScaleFusion, FutureMixingDecoder (Phase 3: Scale Fusion + Output)
  - STTimeMixer (Full model - v3 với Adaptive Adjacency & Node Embeddings)
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────
# CÁC THÀNH PHẦN HỖ TRỢ / UTILS
# ──────────────────────────────────────────────────────────────────────

class SeriesDecomposition(nn.Module):
    """
    Tách chuỗi thời gian thành Trend + Seasonal bằng Moving Average.
    """
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        assert kernel_size % 2 == 1, f"kernel_size phải lẻ, nhận: {kernel_size}"
        self.avg_pool = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )

    def forward(self, x: torch.Tensor):
        x_permuted = x.permute(0, 2, 1)
        trend = self.avg_pool(x_permuted)
        trend = trend[:, :, :x.shape[1]]
        trend = trend.permute(0, 2, 1)
        seasonal = x - trend
        return trend, seasonal

def multiscale_downsample(x: torch.Tensor, scales: list):
    """Tạo các bản sao dữ liệu ở nhiều thang đo thời gian khác nhau."""
    result = {}
    for s in scales:
        if s == 1:
            result[s] = x
        else:
            x_t = x.permute(0, 2, 1)
            x_down = F.avg_pool1d(x_t, kernel_size=s, stride=s)
            result[s] = x_down.permute(0, 2, 1)
    return result

# ──────────────────────────────────────────────────────────────────────
# PHASE 1: TIMEMIXER BRANCH
# ──────────────────────────────────────────────────────────────────────

class PastDecomMixing(nn.Module):
    """MLP Mixing trên trục thời gian cho một scale cụ thể."""
    def __init__(self, seq_len: int, d_model: int, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.trend_mlp = nn.Sequential(
            nn.Linear(seq_len, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.seasonal_mlp = nn.Sequential(
            nn.Linear(seq_len, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.feat_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU()
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, trend: torch.Tensor, seasonal: torch.Tensor) -> torch.Tensor:
        trend_temporal = trend.mean(dim=-1)
        seasonal_temporal = seasonal.mean(dim=-1)
        z_trend = self.trend_mlp(trend_temporal)
        z_seasonal = self.seasonal_mlp(seasonal_temporal)
        z = self.feat_proj(torch.cat([z_trend, z_seasonal], dim=-1))
        return self.norm(z)

class TimeMixerModule(nn.Module):
    """Module TimeMixer hoàn chỉnh."""
    def __init__(
        self, seq_len: int, num_features: int, d_model: int,
        scales: list = None, d_ff: int = 256, dropout: float = 0.1,
        decomp_kernel: int = 25
    ):
        super().__init__()
        self.scales = scales or [1, 4, 24]
        self.seq_len = seq_len
        self.input_proj = nn.Linear(num_features, d_model)
        self.decomp = SeriesDecomposition(kernel_size=decomp_kernel)
        self.past_mixers = nn.ModuleDict()
        for s in self.scales:
            T_s = seq_len // s if s > 1 else seq_len
            self.past_mixers[str(s)] = PastDecomMixing(
                seq_len=T_s, d_model=d_model, d_ff=d_ff, dropout=dropout
            )

    def forward(self, x: torch.Tensor) -> dict:
        x = self.input_proj(x)
        x_scales = multiscale_downsample(x, self.scales)
        scale_embs = {}
        for s in self.scales:
            x_s = x_scales[s]
            trend_s, seasonal_s = self.decomp(x_s)
            z_s = self.past_mixers[str(s)](trend_s, seasonal_s)
            scale_embs[s] = z_s
        return scale_embs

# ──────────────────────────────────────────────────────────────────────
# PHASE 2: SPATIAL GCN BRANCH
# ──────────────────────────────────────────────────────────────────────

class ChebGCNLayer(nn.Module):
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
        self.activation = nn.ReLU()
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

class SpatialGCNBlock(nn.Module):
    """Block GCN cho một scale."""
    def __init__(self, d_model: int, K: int = 3, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gcn_layers = nn.ModuleList([
            ChebGCNLayer(d_model, d_model, K=K, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, z_s: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        h = z_s
        for gcn in self.gcn_layers:
            h = gcn(h, A) + h
        return h

# ──────────────────────────────────────────────────────────────────────
# PHASE 3: FUSION & DECODER 
# ──────────────────────────────────────────────────────────────────────

class ScaleFusion(nn.Module):
    """Gộp embeddings từ nhiều scale thời gian thành một embedding duy nhất."""
    def __init__(self, num_scales: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.fuse_proj = nn.Sequential(
            nn.Linear(num_scales * d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, scale_outputs: list) -> torch.Tensor:
        H_concat = torch.cat(scale_outputs, dim=-1)
        H_fused = self.fuse_proj(H_concat)
        return self.norm(H_fused)

class FutureMixingDecoder(nn.Module):
    """Decoder: từ fused embedding → dự báo H bước tương lai."""
    def __init__(self, d_model: int, pred_len: int, n_targets: int = 2, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.pred_len = pred_len
        self.n_targets = n_targets
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, pred_len * n_targets)

    def forward(self, H_fused: torch.Tensor) -> torch.Tensor:
        B, N, d = H_fused.shape
        h = self.norm(H_fused + self.mlp(H_fused))
        out = self.out_proj(h)
        return out.view(B, N, self.pred_len, self.n_targets)

# ──────────────────────────────────────────────────────────────────────
# FULL MODEL: ST-TIMEMIXER
# ──────────────────────────────────────────────────────────────────────

class STTimeMixer(nn.Module):
    """
    ST-TimeMixer v3 (Enhanced Architecture):
    - Learnable Node Embeddings: Capture unique characteristics of each province.
    - Adaptive Adjacency Matrix: Learn hidden spatial dependencies during training.
    - Scale-wise Cross Mixing: Improved integration of multi-scale temporal features.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.N = config.get('num_nodes', 12)
        self.T = config.get('seq_len', 24)
        self.H = config.get('pred_len', 12)
        self.C = config.get('num_features', 22)
        self.d = config.get('d_model', 128)
        self.n_targets = config.get('num_targets', 2)
        self.scales = config.get('scales', [1, 4, 24])
        S = len(self.scales)

        d_ff = config.get('d_ff', 512)
        K = config.get('K_cheby', 3)
        num_gcn = config.get('num_gcn_layers', 3)
        dropout = config.get('dropout', 0.15)
        decomp_kernel = config.get('decomp_kernel', 25)

        # 1. NODE EMBEDDINGS
        self.node_emb = nn.Parameter(torch.randn(self.N, self.d))

        # 2. ADAPTIVE ADJACENCY
        self.node_vec1 = nn.Parameter(torch.randn(self.N, 10))
        self.node_vec2 = nn.Parameter(torch.randn(self.N, 10))

        # PHASE 1: TimeMixer Branch
        self.timemixer = TimeMixerModule(
            seq_len=self.T, num_features=self.C, d_model=self.d,
            scales=self.scales, d_ff=d_ff, dropout=dropout,
            decomp_kernel=decomp_kernel
        )

        # PHASE 2: STGCN Branch
        self.gcn_per_scale = nn.ModuleDict({
            str(s): SpatialGCNBlock(d_model=self.d, K=K, num_layers=num_gcn, dropout=dropout)
            for s in self.scales
        })

        # PHASE 3: Scale Fusion + Decoder
        self.scale_fusion = ScaleFusion(num_scales=S, d_model=self.d, dropout=dropout)
        self.decoder = FutureMixingDecoder(
            d_model=self.d, pred_len=self.H, n_targets=self.n_targets,
            d_ff=d_ff, dropout=dropout
        )

        total_params = sum(p.numel() for p in self.parameters())
        print(f"[STTimeMixer v3] Tổng params: {total_params:,}")

    def get_adaptive_adj(self) -> torch.Tensor:
        """Computes learnable adjacency: A_adapt = Softmax(Relu(V1 @ V2^T))"""
        a_adapt = F.relu(torch.mm(self.node_vec1, self.node_vec2.transpose(0, 1)))
        return F.softmax(a_adapt, dim=1)

    def forward(self, x: torch.Tensor, A_static: torch.Tensor) -> torch.Tensor:
        B, N, T, C = x.shape

        # 1. TimeMixer
        x_flat = x.reshape(B * N, T, C)
        scale_embs = self.timemixer(x_flat)

        # 2. Node Embeddings
        node_emb_expanded = self.node_emb.unsqueeze(0).expand(B, -1, -1)

        # 3. Adaptive Adjacency matrix
        A_adapt = self.get_adaptive_adj()
        if A_static.device != x.device:
            A_static = A_static.to(x.device)
        A_total = 0.7 * A_static + 0.3 * A_adapt

        # 4. Spatial GCN per scale
        spatial_outputs = []
        for s in self.scales:
            z_s = scale_embs[s].reshape(B, N, self.d)
            z_s = z_s + node_emb_expanded
            h_s = self.gcn_per_scale[str(s)](z_s, A_total)
            spatial_outputs.append(h_s)

        # 5. Scale Fusion
        h_fused = self.scale_fusion(spatial_outputs)

        # 6. Forecasting Decoder
        out = self.decoder(h_fused)

        return out

def create_st_timemixer(**kwargs) -> STTimeMixer:
    """Utility to create model from kwargs."""
    config = {
        'num_nodes': kwargs.get('num_nodes', 12),
        'seq_len': kwargs.get('seq_len', 24),
        'pred_len': kwargs.get('pred_len', 12),
        'num_features': kwargs.get('num_features', 22),
        'num_targets': kwargs.get('num_targets', 2),
        'd_model': kwargs.get('d_model', 128),
        'd_ff': kwargs.get('d_ff', 512),
        'scales': kwargs.get('scales', [1, 4, 24]),
        'K_cheby': kwargs.get('K_cheby', 3),
        'num_gcn_layers': kwargs.get('num_gcn_layers', 3),
        'dropout': kwargs.get('dropout', 0.15),
        'decomp_kernel': kwargs.get('decomp_kernel', 25),
    }
    return STTimeMixer(config)
