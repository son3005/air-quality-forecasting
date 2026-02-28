"""
═══════════════════════════════════════════════════════════════
Parallel STMixer v2 — Parallel Fusion + Gating (Enhanced)
═══════════════════════════════════════════════════════════════
Cải tiến so với v1:
  1. Auxiliary decoder RIÊNG cho mỗi nhánh (không dùng chung decoder)
  2. Thêm Node Embedding (learnable per-node bias)
  3. Residual từ input trực tiếp vào decoder
  4. Decoder 3 tầng thay vì 2 tầng
═══════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .timemixer_branch import TimeMixerBranch
from .stgcn_branch import STGCNBranch
from .fusion import AdaptiveFusionGate


class ParallelSTMixer(nn.Module):
    """
    Parallel STMixer v2: Parallel Fusion + Adaptive Gating.
    """
    def __init__(self, config):
        super().__init__()
        self.N = config.get('num_nodes', 12)
        self.T = config.get('seq_len', 72)
        self.H = config.get('pred_len', 12)
        self.F_in = config.get('num_features', 52)
        self.n_targets = config.get('num_targets', 2)

        self.d_model = config.get('d_model', 128)
        self.d_ff = config.get('d_ff', 512)
        self.dropout = config.get('dropout', 0.15)

        self.scales = config.get('scales', [1, 4, 24])
        self.K = config.get('K_cheby', 3)
        self.num_st_layers = config.get('num_st_layers', 2)
        self.decomp_kernel = config.get('decomp_kernel', 25)

        # ── Adaptive Adjacency ──
        self.adaptive_adj = config.get('adaptive_adj', True)
        if self.adaptive_adj:
            self.node_vec1 = nn.Parameter(torch.randn(self.N, 16))
            self.node_vec2 = nn.Parameter(torch.randn(self.N, 16))

        # ── Node Embedding: per-node learnable bias ──
        self.node_emb = nn.Parameter(torch.randn(self.N, self.d_model) * 0.02)

        # ── TimeMixer Branch ──
        self.tm_branch = TimeMixerBranch(
            seq_len=self.T, num_features=self.F_in, d_model=self.d_model,
            scales=self.scales, d_ff=self.d_ff, dropout=self.dropout,
            decomp_kernel=self.decomp_kernel
        )

        # ── STGCN Branch ──
        self.st_branch = STGCNBranch(
            seq_len=self.T, num_features=self.F_in, d_model=self.d_model,
            K=self.K, num_layers=self.num_st_layers, dropout=self.dropout
        )

        # ── Fusion Gate ──
        self.fusion = AdaptiveFusionGate(d_model=self.d_model, dropout=self.dropout)

        # ── Main Decoder (3-layer, deeper capacity) ──
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff, self.d_ff // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff // 2, self.H * self.n_targets)
        )

        # ── Auxiliary Decoders (RIÊNG cho mỗi nhánh — lighter) ──
        self.aux_decoder_tm = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff // 2, self.H * self.n_targets)
        )
        self.aux_decoder_st = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff // 2, self.H * self.n_targets)
        )

        total_params = sum(p.numel() for p in self.parameters())
        print(f"[Parallel STMixer v2] Total parameters: {total_params:,}")

    def get_adj(self, A_static: torch.Tensor = None) -> torch.Tensor:
        if self.adaptive_adj:
            A_adapt = F.relu(torch.mm(self.node_vec1, self.node_vec2.transpose(0, 1)))
            A_adapt = F.softmax(A_adapt, dim=1)
            if A_static is not None:
                if A_static.device != A_adapt.device:
                    A_static = A_static.to(A_adapt.device)
                return 0.6 * A_static + 0.4 * A_adapt
            return A_adapt
        return A_static

    def forward(self, x: torch.Tensor, A_static: torch.Tensor = None) -> dict:
        """
        x: [B, N, T, F]
        Returns: dict with 'pred', 'pred_tm', 'pred_st', 'gate'
        """
        B, N, T, F = x.shape
        adj = self.get_adj(A_static)

        # ── 1. TimeMixer Branch (node-wise temporal) ──
        x_tm = x.reshape(B * N, T, F)
        z_tm = self.tm_branch(x_tm)             # [B*N, d_model]
        z_tm = z_tm.reshape(B, N, self.d_model)  # [B, N, d_model]

        # ── 2. STGCN Branch (spatial-temporal) ──
        z_st = self.st_branch(x, adj)            # [B, N, d_model]

        # ── 3. Add Node Embeddings to both branches ──
        node_emb = self.node_emb.unsqueeze(0).expand(B, -1, -1)  # [B, N, d]
        z_tm = z_tm + node_emb
        z_st = z_st + node_emb

        # ── 4. Fusion Gate ──
        z_fused, gate = self.fusion(z_tm, z_st)  # [B, N, d_model]

        # ── 5. Main Decoder ──
        out = self.decoder(z_fused)
        out = out.view(B, N, self.H, self.n_targets)

        # ── 6. Auxiliary Branch Predictions (RIÊNG decoder) ──
        pred_tm = self.aux_decoder_tm(z_tm).view(B, N, self.H, self.n_targets)
        pred_st = self.aux_decoder_st(z_st).view(B, N, self.H, self.n_targets)

        return {
            'pred': out,
            'pred_tm': pred_tm,
            'pred_st': pred_st,
            'gate': gate
        }


def create_parallel_stmixer(config: dict) -> ParallelSTMixer:
    """Helper to instantiate Parallel STMixer v2 from config dictionary."""
    return ParallelSTMixer(config)
