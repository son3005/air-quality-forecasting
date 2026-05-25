"""
models/TFT/model.py

Self-contained PyTorch implementation of the Temporal Fusion Transformer (TFT)
adapted for PM2.5 multi-station forecasting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GLU(nn.Module):
    """
    Gated Linear Unit (GLU) for gating network activation.
    """
    def __init__(self, d_input, d_output):
        super().__init__()
        self.fc = nn.Linear(d_input, d_output * 2)

    def forward(self, x):
        out = self.fc(x)
        val, gate = out.chunk(2, dim=-1)
        return val * torch.sigmoid(gate)


class GRN(nn.Module):
    """
    Gated Residual Network (GRN) to suppress or pass features.
    """
    def __init__(self, d_input, d_hidden, d_output, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_input, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.glu = GLU(d_hidden, d_output)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_output)
        
        # Linear residual skip connection if dimensions differ
        if d_input != d_output:
            self.skip = nn.Linear(d_input, d_output)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = self.dropout(self.fc2(h))
        h = self.glu(h)
        return self.norm(h + self.skip(x))


class VSN(nn.Module):
    """
    Variable Selection Network (VSN) for feature weight determination.
    """
    def __init__(self, num_vars, d_model, dropout=0.1):
        super().__init__()
        self.num_vars = num_vars
        self.d_model = d_model
        
        # GRN for each variable to map feature values to d_model
        self.var_grns = nn.ModuleList([
            GRN(1, d_model, d_model, dropout)
            for _ in range(num_vars)
        ])
        
        # Flattened GRN to compute attention weights for variables
        self.flatten_grn = GRN(num_vars * 1, d_model, num_vars, dropout)

    def forward(self, x):
        # x: (B, L, num_vars)
        B, L, num_vars = x.shape
        
        # Reshape to (B, L, num_vars, 1) for variable-specific GRNs
        x_split = x.unsqueeze(-1)  # (B, L, num_vars, 1)
        
        # Compute selection weights across variables
        flat_x = x_split.view(B, L, num_vars * 1)
        weights = torch.softmax(self.flatten_grn(flat_x), dim=-1).unsqueeze(-1)  # (B, L, num_vars, 1)
        
        # Pass each variable through its corresponding GRN
        var_outputs = []
        for i in range(num_vars):
            var_outputs.append(self.var_grns[i](x_split[:, :, i]))  # (B, L, d_model)
            
        var_outputs = torch.stack(var_outputs, dim=2)  # (B, L, num_vars, d_model)
        
        # Weighted sum: (B, L, d_model)
        out = torch.sum(weights * var_outputs, dim=2)
        return out


class TFTModel(nn.Module):
    """
    Temporal Fusion Transformer for multi-station forecasting.
    """
    def __init__(self, seq_len=48, pred_len=1, enc_in=37,
                 d_model=64, n_heads=4, dropout=0.1, use_norm=True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.use_norm = use_norm

        # Variable Selection Network for features selection
        self.vsn = VSN(num_vars=enc_in, d_model=d_model, dropout=dropout)

        # Gated Residual Network for temporal representation
        self.grn = GRN(d_model, d_model, d_model, dropout)

        # Multihead self-attention layer
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)

        # Output projection to pred_len * enc_in
        self.projector = nn.Linear(d_model * seq_len, pred_len * enc_in)

    def forward(self, x):
        """
        x: (B, seq_len, enc_in)
        Returns: (B, pred_len, enc_in)
        """
        if self.use_norm:
            # Reversible instance normalization
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev

        # 1. Variable Selection Network: (B, seq_len, enc_in) -> (B, seq_len, d_model)
        out = self.vsn(x)

        # 2. GRN layer: (B, seq_len, d_model) -> (B, seq_len, d_model)
        out = self.grn(out)

        # 3. Multi-Head Attention layer: (B, seq_len, d_model) -> (B, seq_len, d_model)
        attn_out, _ = self.mha(out, out, out, need_weights=False)
        out = self.attn_norm(out + self.attn_dropout(attn_out))

        # 4. Flatten and Project: (B, seq_len * d_model) -> (B, pred_len * enc_in) -> (B, pred_len, enc_in)
        out = out.reshape(out.size(0), -1)
        dec_out = self.projector(out).view(out.size(0), self.pred_len, self.enc_in)

        if self.use_norm:
            # Inverse normalization
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out
