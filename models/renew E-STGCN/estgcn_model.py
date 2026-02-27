"""
E-STGCN Model (PyTorch) - Extreme Spatio-Temporal Graph Convolutional Networks.
Kiến trúc: GraphConv (Message-Passing) → LSTM → Dense
Hỗ trợ Đa biến (Multivariate) + EVT-GPD Loss.

Phiên bản 2: Đã tinh chỉnh chống Overfitting:
  - Thêm Dropout sau GraphConv và LSTM
  - Thêm LayerNorm cho ổn định gradient
  - Tăng GCN capacity mặc định
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphConvOptimized(nn.Module):
    """
    Lớp Tích chập Đồ thị dạng Message-Passing sử dụng ma trận kề.
    Tối ưu hóa bằng phép nhân ma trận thay vì vòng lặp.
    """

    def __init__(self, in_feat, out_feat, num_nodes, combination_type="concat", dropout=0.3):
        super(GraphConvOptimized, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_nodes = num_nodes
        self.combination_type = combination_type

        self.weight = nn.Parameter(torch.FloatTensor(in_feat, out_feat))
        nn.init.xavier_uniform_(self.weight)

        self.dropout = nn.Dropout(dropout)

        # Output dimension phụ thuộc combination type
        combined_dim = out_feat * 2 if combination_type == "concat" else out_feat
        self.layer_norm = nn.LayerNorm(combined_dim)

    def forward(self, features, A_hat):
        """
        Args:
            features: (batch_size, seq_len, num_nodes, in_feat)
            A_hat: (num_nodes, num_nodes)
        Returns:
            output: (batch_size, seq_len, num_nodes, combined_feat)
        """
        # 1. Tính đại diện cho mỗi node
        nodes_repr = torch.matmul(features, self.weight)

        # 2. Message Passing bằng ma trận nhân
        aggregated = torch.einsum('bsni,nm->bsmi', features, A_hat)
        aggregated_repr = torch.matmul(aggregated, self.weight)

        # 3. Kết hợp
        if self.combination_type == "concat":
            output = torch.cat([nodes_repr, aggregated_repr], dim=-1)
        elif self.combination_type == "add":
            output = nodes_repr + aggregated_repr
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}")

        # 4. LayerNorm + Dropout
        output = self.layer_norm(output)
        output = self.dropout(output)

        return output


class LSTMGC(nn.Module):
    """
    Lớp chính của E-STGCN: GraphConv → LSTM (2 layers) → Dense.
    Kết hợp thông tin không gian (GraphConv) với phụ thuộc thời gian dài hạn (LSTM).
    
    Tinh chỉnh v2:
      - LSTM 2 layers + dropout giữa các layer
      - Dropout sau Dense
      - Residual connection nếu dims khớp
    """

    def __init__(self, in_feat, out_feat, lstm_units, input_seq_len, output_seq_len,
                 num_nodes, combination_type="concat", dropout=0.3):
        super(LSTMGC, self).__init__()

        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.num_nodes = num_nodes

        # Graph Convolution Layer
        self.graph_conv = GraphConvOptimized(in_feat, out_feat, num_nodes, combination_type, dropout)

        # LSTM input size depends on combination type
        lstm_input_size = out_feat * 2 if combination_type == "concat" else out_feat

        # LSTM Layer - 2 layers với dropout
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_units,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(lstm_units, output_seq_len)

    def forward(self, inputs, A_hat):
        """
        Args:
            inputs: (batch_size, input_seq_len, num_nodes, in_feat)
            A_hat: (num_nodes, num_nodes)
        Returns:
            output: (batch_size, num_nodes, output_seq_len)
        """
        batch_size = inputs.shape[0]

        # 1. Graph Convolution
        gcn_out = self.graph_conv(inputs, A_hat)
        gcn_out_feat = gcn_out.shape[-1]

        # 2. Reshape cho LSTM
        gcn_out = gcn_out.permute(0, 2, 1, 3)  # (batch, nodes, seq, gcn_out_feat)
        gcn_out = gcn_out.reshape(batch_size * self.num_nodes, self.input_seq_len, gcn_out_feat)

        # 3. LSTM
        lstm_out, _ = self.lstm(gcn_out)  # (batch*nodes, seq, lstm_units)
        lstm_out = lstm_out[:, -1, :]  # Lấy output cuối: (batch*nodes, lstm_units)

        # 4. Dropout + Dense
        lstm_out = self.dropout(lstm_out)
        dense_out = self.dense(lstm_out)  # (batch*nodes, output_seq_len)

        # 5. Reshape
        output = dense_out.reshape(batch_size, self.num_nodes, self.output_seq_len)

        return output


class ESTGCN(nn.Module):
    """
    E-STGCN: Extreme Spatio-Temporal Graph Convolutional Networks.
    Phiên bản 2: Tăng cường chống overfitting + capacity lớn hơn.
    """

    def __init__(self, num_nodes, num_features, input_seq_len, output_seq_len,
                 gcn_out_feat=32, lstm_units=128, combination_type="concat", dropout=0.3):
        super(ESTGCN, self).__init__()

        self.lstmgc = LSTMGC(
            in_feat=num_features,
            out_feat=gcn_out_feat,
            lstm_units=lstm_units,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_nodes=num_nodes,
            combination_type=combination_type,
            dropout=dropout
        )

    def forward(self, A_hat, inputs):
        """
        Args:
            A_hat: (num_nodes, num_nodes)
            inputs: (batch_size, input_seq_len, num_nodes, num_features)
        Returns:
            output: (batch_size, num_nodes, output_seq_len)
        """
        return self.lstmgc(inputs, A_hat)


class EVTGPDLoss(nn.Module):
    """
    Hàm Loss kết hợp MSE + GPD Penalty (Extreme Value Theory).
    
    Phiên bản 2: Cải thiện ổn định tính toán:
      - Detach y_pred khi tính GPD penalty (không cho gradient GPD đẩy model lệch MSE)
      - Clamping chặt chẽ hơn
      - Warmup: chỉ bật GPD penalty sau N epochs đầu
    """

    def __init__(self, xi, sigma, mean_vals, std_vals, target_col_idx=0,
                 threshold=60.0, beta_1=0.99, beta_2=0.01):
        super(EVTGPDLoss, self).__init__()
        self.threshold = threshold
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.warmup_done = False  # Bật GPD penalty sau warmup

        self.register_buffer('xi', torch.FloatTensor(xi))
        self.register_buffer('sig', torch.FloatTensor(sigma))
        self.register_buffer('mean_val', torch.tensor(mean_vals, dtype=torch.float32))
        self.register_buffer('std_val', torch.tensor(std_vals, dtype=torch.float32))

    def set_warmup(self, enabled):
        """Bật/tắt GPD penalty. Nên tắt trong 5-10 epochs đầu."""
        self.warmup_done = enabled

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: (batch_size, num_nodes, output_seq_len)
            y_true: (batch_size, num_nodes, output_seq_len)
        Returns:
            loss: scalar
        """
        # 1. MSE Loss (trên thang chuẩn hóa)
        mse_loss = F.mse_loss(y_pred, y_true)

        if not self.warmup_done:
            return mse_loss

        # 2. GPD Penalty (chỉ khi warmup xong)
        # QUAN TRỌNG: Detach y_pred để GPD penalty không phá gradient MSE
        y_pred_detached = y_pred.detach()
        y_pred_original = y_pred_detached * self.std_val + self.mean_val

        xi = self.xi.unsqueeze(0).unsqueeze(-1)   # (1, nodes, 1)
        sig = self.sig.unsqueeze(0).unsqueeze(-1)  # (1, nodes, 1)

        # Tính GPD negative log-likelihood
        z = xi * y_pred_original / (sig + 1e-6)
        z_safe = torch.clamp(1.0 + z, min=1e-6)

        gpd_nll = torch.log(sig + 1e-6) + (1.0 + 1.0 / (xi + 1e-6)) * torch.log(z_safe)

        # Chỉ áp dụng khi vượt ngưỡng
        mask = (y_pred_original > self.threshold).float()
        gpd_penalty = (gpd_nll * mask).mean()
        gpd_penalty = torch.clamp(gpd_penalty, min=-50, max=50)

        # 3. Tổng hợp Loss
        total_loss = self.beta_1 * mse_loss + self.beta_2 * gpd_penalty

        return total_loss
