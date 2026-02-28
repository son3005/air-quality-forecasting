"""
═══════════════════════════════════════════════════════════════
Adaptive Fusion Gate v2
═══════════════════════════════════════════════════════════════
Cải tiến: Thêm residual connection qua gate
═══════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveFusionGate(nn.Module):
    """
    Adaptive Fusion Gate v2.
    Học trọng số kết hợp giữa TimeMixer Branch và STGCN Branch.
    Thêm residual connection từ average hai nhánh.
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, z_tm: torch.Tensor, z_st: torch.Tensor):
        """
        z_tm: [B, N, d_model]
        z_st: [B, N, d_model]
        """
        combined = torch.cat([z_tm, z_st], dim=-1)
        gate = self.gate_proj(combined)  # [B, N, d_model]
        
        # Gated fusion
        z_fused = gate * z_tm + (1 - gate) * z_st
        
        # Residual: average of both inputs 
        z_avg = (z_tm + z_st) * 0.5
        z_fused = z_fused + z_avg

        return self.norm(z_fused), gate
