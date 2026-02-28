"""
Hybrid STGCN + XLinear + EVT-GPD Model v2 (PyTorch).

Thiết kế lại: KHÔNG thay thế Conv1D bằng Gating, mà KẾT HỢP cả hai.
  - Conv1D TimeBlock (GLU): bắt pattern thời gian CỤC BỘ (3 giờ liên tiếp)
  - XLinear Gating: lọc feature TOÀN CỤC (giữ tín hiệu quan trọng, bỏ noise)
  → Conv1D cho locality + Gating cho selectivity = best of both worlds

Kiến trúc mỗi block:
  TimeBlock(Conv1D+GLU) → XLinear Gating → Graph Conv → TimeBlock → Gating → BN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ========================================================================
# XLinear Gating Block
# ========================================================================

class GatingBlock(nn.Module):
    """x * sigmoid(MLP(x)) — chọn lọc features quan trọng."""

    def __init__(self, d_model, hidden_dim, dropout=0.):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.gate(x)


# ========================================================================
# Enhanced TimeBlock: Conv1D GLU + XLinear Gating
# ========================================================================

class EnhancedTimeBlock(nn.Module):
    """
    Conv1D (GLU) + XLinear Gating.
    Conv1D bắt pattern cục bộ → Gating lọc noise toàn cục.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, gating_ff=64):
        super().__init__()
        # Conv1D + GLU (giữ nguyên từ STGCN)
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

        # XLinear Gating (thêm mới): lọc theo chiều channel
        self.gating = GatingBlock(out_channels, gating_ff)

    def forward(self, X):
        """X: (batch, in_channels, nodes, seq_len)"""
        # Conv1D + GLU (local temporal patterns)
        v1 = self.conv1(X)
        v2 = torch.sigmoid(self.conv2(X))
        residual = self.conv3(X)
        out = residual + v1 * v2  # (batch, out_ch, nodes, seq-2)

        # XLinear Gating (global feature selection trên channel dimension)
        # Permute: (batch, ch, nodes, seq) → (batch, nodes, seq, ch)
        out = out.permute(0, 2, 3, 1)
        out = self.gating(out)
        # Permute back: (batch, nodes, seq, ch) → (batch, ch, nodes, seq)
        out = out.permute(0, 3, 1, 2)

        return out


# ========================================================================
# STX Block (Spatio-Temporal-XLinear Block)
# ========================================================================

class STXBlock(nn.Module):
    """
    Enhanced TimeBlock → Graph Conv → Enhanced TimeBlock → BatchNorm

    So với STGCN: thêm XLinear Gating sau mỗi TimeBlock.
    So với XLinear thuần: giữ Conv1D cho locality.
    """

    def __init__(self, in_channels, spatial_channels, out_channels, num_nodes, gating_ff=64):
        super().__init__()
        self.temporal1 = EnhancedTimeBlock(in_channels, out_channels, gating_ff=gating_ff)
        self.theta = nn.Parameter(torch.FloatTensor(out_channels, spatial_channels))
        nn.init.xavier_uniform_(self.theta)
        self.temporal2 = EnhancedTimeBlock(spatial_channels, out_channels, gating_ff=gating_ff)
        self.batch_norm = nn.BatchNorm2d(num_nodes)

    def forward(self, X, A_hat):
        """X: (batch, in_ch, nodes, seq_len), A_hat: (nodes, nodes)"""
        # Temporal + Gating 1
        t = self.temporal1(X)

        # Spectral Graph Conv
        t_p = t.permute(0, 3, 2, 1)  # (batch, seq, nodes, ch)
        s = torch.einsum('btnd,nm->btmd', t_p, A_hat)
        s = torch.relu(torch.matmul(s, self.theta))
        s = s.permute(0, 3, 2, 1)  # (batch, spatial_ch, nodes, seq)

        # Temporal + Gating 2
        t2 = self.temporal2(s)

        # BatchNorm
        out = self.batch_norm(t2.permute(0, 2, 1, 3))
        return out.permute(0, 2, 1, 3)


# ========================================================================
# Cross-Variable Attention (từ XLinear)
# ========================================================================

class CrossVariableGating(nn.Module):
    """
    Lọc feature nào quan trọng cho prediction (từ XLinear Forcast_with_exogenous).
    Áp dụng Gating trên chiều channels sau khi qua STX blocks.
    """

    def __init__(self, num_channels, hidden_dim):
        super().__init__()
        self.gate = GatingBlock(num_channels, hidden_dim)

    def forward(self, X):
        """X: (batch, channels, nodes, seq_left)"""
        # Permute: (batch, ch, nodes, seq) → (batch, nodes, seq, ch)
        X = X.permute(0, 2, 3, 1)
        X = self.gate(X)
        # (batch, nodes, seq, ch) → (batch, ch, nodes, seq)
        return X.permute(0, 3, 1, 2)


# ========================================================================
# Main Model
# ========================================================================

class STGCN_XLinear(nn.Module):
    """
    STGCN + XLinear Hybrid v2.

    Kiến trúc:
      2x STXBlock (Conv1D+Gating → GCN → Conv1D+Gating → BN)
      → Cross-Variable Gating
      → Last Enhanced TimeBlock
      → FC Head

    Ưu điểm kết hợp:
      - Conv1D (STGCN): bắt pattern cục bộ 3h liên tiếp (ngày/đêm, rush hour)
      - Gating (XLinear): lọc bỏ noise, tăng cường tín hiệu quan trọng
      - Cross-var Gating: tự động chọn features ảnh hưởng nhất đến target
      - Graph Conv: quan hệ không gian giữa trạm
    """

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, gating_ff=64):
        super().__init__()

        # STXBlock 1: (num_features -> 16 spatial -> 64 out)
        self.block1 = STXBlock(num_features, 16, 64, num_nodes, gating_ff)
        # STXBlock 2: (64 in -> 16 spatial -> 64 out)
        self.block2 = STXBlock(64, 16, 64, num_nodes, gating_ff)

        self.cross_var = CrossVariableGating(64, gating_ff)
        self.last_temporal = EnhancedTimeBlock(64, 64, gating_ff=gating_ff)

        # Output head
        # STXBlock has 2 EnhancedTimeBlocks (kernel=3 without padding -> length decreases by 2 per block)
        # 1 STXBlock reduces length by 2 * 2 = 4
        # 2 STXBlocks reduce length by 8
        # last_temporal reduces length by 2
        # Total reduction = 10
        out_time_len = num_timesteps_input - 10
        self.fc1 = nn.Linear(out_time_len * 64, 256)
        self.fc2 = nn.Linear(256, num_timesteps_output)

    def forward(self, A_hat, X):
        """X: (batch, seq_len, nodes, features)"""
        X = X.permute(0, 3, 2, 1)  # → (batch, features, nodes, seq)

        out = self.block1(X, A_hat)
        out = self.block2(out, A_hat)

        # Cross-Variable Gating: lọc channels quan trọng
        out = self.cross_var(out)

        out = self.last_temporal(out)

        b, c, n, t = out.shape
        out = out.permute(0, 2, 1, 3).reshape(b, n, c * t)

        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out  # (batch, nodes, pred_len)


# ========================================================================
# EVT-GPD Loss
# ========================================================================

class EVTGPDLoss(nn.Module):
    def __init__(self, xi, sigma, mean_vals, std_vals,
                 threshold=60.0, beta_1=0.99, beta_2=0.01):
        super().__init__()
        self.threshold, self.beta_1, self.beta_2 = threshold, beta_1, beta_2
        self.warmup_done = False
        self.register_buffer('xi', torch.FloatTensor(xi))
        self.register_buffer('sig', torch.FloatTensor(sigma))
        self.register_buffer('mean_val', torch.tensor(mean_vals, dtype=torch.float32))
        self.register_buffer('std_val', torch.tensor(std_vals, dtype=torch.float32))

    def set_warmup(self, enabled):
        self.warmup_done = enabled

    def forward(self, y_pred, y_true):
        mse = F.mse_loss(y_pred, y_true)
        if not self.warmup_done:
            return mse
        y_d = y_pred.detach() * self.std_val + self.mean_val
        xi = self.xi.unsqueeze(0).unsqueeze(-1)
        sig = self.sig.unsqueeze(0).unsqueeze(-1)
        z_safe = torch.clamp(1.0 + xi * y_d / (sig + 1e-6), min=1e-6)
        gpd = torch.log(sig + 1e-6) + (1 + 1 / (xi + 1e-6)) * torch.log(z_safe)
        mask = (y_d > self.threshold).float()
        penalty = torch.clamp((gpd * mask).mean(), -50, 50)
        return self.beta_1 * mse + self.beta_2 * penalty
