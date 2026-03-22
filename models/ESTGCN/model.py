"""
models/ESTGCN/model.py

E-STGCN model architecture (Enhanced Spatio-Temporal Graph Convolutional Network).
GraphConv → LSTMGC → ESTGCN wrapper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    """Graph Convolution: X*W → A*(X*W) → Combine + ReLU."""
    def __init__(self, in_feat, out_feat, combination_type="concat"):
        super().__init__()
        self.combination_type = combination_type
        self.weight = nn.Parameter(torch.FloatTensor(in_feat, out_feat))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj):
        nodes_repr = torch.matmul(features, self.weight)
        aggregated = torch.matmul(adj, nodes_repr)
        if self.combination_type == "concat":
            h = torch.cat([nodes_repr, aggregated], dim=-1)
        elif self.combination_type == "add":
            h = nodes_repr + aggregated
        else:
            raise ValueError("combination_type must be 'concat' or 'add'")
        return F.relu(h)


class LSTMGC(nn.Module):
    """GraphConv → Reshape → LSTM → Dense."""
    def __init__(self, in_feat, out_feat, lstm_units, input_seq_len, output_seq_len, combination_type="concat"):
        super().__init__()
        self.graph_conv = GraphConv(in_feat, out_feat, combination_type=combination_type)
        lstm_input_dim = out_feat * 2 if combination_type == "concat" else out_feat
        self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=lstm_units, num_layers=1, batch_first=True)
        self.dense = nn.Linear(lstm_units, output_seq_len)

    def forward(self, inputs, adj):
        b, t, n, f = inputs.shape
        gcn_out = self.graph_conv(inputs, adj)
        gcn_out = gcn_out.permute(0, 2, 1, 3).contiguous().view(b * n, t, -1)
        lstm_out, _ = self.lstm(gcn_out)
        lstm_last = lstm_out[:, -1, :]
        dense_out = self.dense(lstm_last)
        output = dense_out.view(b, n, -1).permute(0, 2, 1).contiguous()
        return output


class ESTGCN(nn.Module):
    """E-STGCN wrapper."""
    def __init__(self, num_nodes, num_features, seq_len, pred_len):
        super().__init__()
        out_feat = 10
        lstm_units = 64
        combination_type = "concat"
        self.core = LSTMGC(
            in_feat=num_features, out_feat=out_feat,
            lstm_units=lstm_units, input_seq_len=seq_len,
            output_seq_len=pred_len, combination_type=combination_type,
        )

    def forward(self, x, adj):
        return self.core(x, adj)
