"""
Hybrid STGCN + EVT-GPD Model (PyTorch) — v3 Final.
Quay lại kiến trúc STGCN gốc (đã chứng minh hiệu quả nhất: R²=0.43)
Chỉ thêm Dropout nhẹ vào FC layers để regularize.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TimeBlock(nn.Module):
    """Temporal Convolution Block với GLU (Gated Linear Unit)."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        v1 = self.conv1(X)
        v2 = torch.sigmoid(self.conv2(X))
        residual = self.conv3(X)
        return residual + v1 * v2


class STGCNBlock(nn.Module):
    """Temporal Conv → Spectral Graph Conv → Temporal Conv → BatchNorm"""

    def __init__(self, in_channels, spatial_channels, out_channels, num_nodes):
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels, out_channels=out_channels)
        self.theta = nn.Parameter(torch.FloatTensor(out_channels, spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels, out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        nn.init.xavier_uniform_(self.theta)

    def forward(self, X, A_hat):
        t = self.temporal1(X)
        t_permuted = t.permute(0, 3, 2, 1)
        out = torch.einsum('btnd,nm->btmd', t_permuted, A_hat)
        out = torch.relu(torch.matmul(out, self.theta))
        out = out.permute(0, 3, 2, 1)
        t2 = self.temporal2(out)
        out = self.batch_norm(t2.permute(0, 2, 1, 3))
        return out.permute(0, 2, 1, 3)


class STGCN(nn.Module):
    """
    STGCN — 2 ST-Conv blocks + Last temporal + FC (với Dropout).
    Kiến trúc đã chứng minh tốt nhất trên dữ liệu Việt Nam.
    """

    def __init__(self, num_nodes, num_features, num_timesteps_input, num_timesteps_output):
        super(STGCN, self).__init__()

        self.block1 = STGCNBlock(in_channels=num_features, spatial_channels=64,
                                 out_channels=128, num_nodes=num_nodes)
        self.block2 = STGCNBlock(in_channels=128, spatial_channels=64,
                                 out_channels=128, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=128, out_channels=128)
        self.dropout = nn.Dropout(p=0.2)

        self.fully_connected_1 = nn.Linear((num_timesteps_input - 10) * 128, 256)
        self.fully_connected_2 = nn.Linear(256, num_timesteps_output)

    def forward(self, A_hat, X):
        X = X.permute(0, 3, 2, 1)

        out1 = self.block1(X, A_hat)
        out2 = self.block2(out1, A_hat)
        out3 = self.last_temporal(out2)

        batch_size, channels, num_nodes, seq_len_left = out3.shape
        out4 = out3.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, channels * seq_len_left)

        out5 = self.dropout(F.relu(self.fully_connected_1(out4)))
        out6 = self.fully_connected_2(out5)

        return out6


# ========================================================================
# EVT-GPD Loss
# ========================================================================

class EVTGPDLoss(nn.Module):
    """Loss = β₁ × MSE + β₂ × GPD Penalty (detached gradient)."""

    def __init__(self, xi, sigma, mean_vals, std_vals,
                 threshold=60.0, beta_1=0.99, beta_2=0.01):
        super(EVTGPDLoss, self).__init__()
        self.threshold = threshold
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.warmup_done = False

        self.register_buffer('xi', torch.FloatTensor(xi))
        self.register_buffer('sig', torch.FloatTensor(sigma))
        self.register_buffer('mean_val', torch.tensor(mean_vals, dtype=torch.float32))
        self.register_buffer('std_val', torch.tensor(std_vals, dtype=torch.float32))

    def set_warmup(self, enabled):
        self.warmup_done = enabled

    def forward(self, y_pred, y_true):
        mse_loss = F.mse_loss(y_pred, y_true)

        if not self.warmup_done:
            return mse_loss

        y_pred_d = y_pred.detach()
        y_original = y_pred_d * self.std_val + self.mean_val

        xi = self.xi.unsqueeze(0).unsqueeze(-1)
        sig = self.sig.unsqueeze(0).unsqueeze(-1)

        z = xi * y_original / (sig + 1e-6)
        z_safe = torch.clamp(1.0 + z, min=1e-6)
        gpd_nll = torch.log(sig + 1e-6) + (1.0 + 1.0 / (xi + 1e-6)) * torch.log(z_safe)

        mask = (y_original > self.threshold).float()
        gpd_penalty = torch.clamp((gpd_nll * mask).mean(), min=-50, max=50)

        return self.beta_1 * mse_loss + self.beta_2 * gpd_penalty
