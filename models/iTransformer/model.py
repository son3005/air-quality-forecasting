"""
models/iTransformer/model.py

iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
Paper: https://arxiv.org/abs/2310.06625 (ICLR 2024 Spotlight)

Self-contained implementation — no external layers needed.
Adapted for PM2.5 multi-station forecasting.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ══════════════════════════════════════════════════════════════════════════
# LAYERS
# ══════════════════════════════════════════════════════════════════════════

class DataEmbedding_inverted(nn.Module):
    """Inverted embedding: (B, L, N) → permute → (B, N, L) → Linear → (B, N, d_model)"""
    def __init__(self, seq_len, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: (B, L, N) → (B, N, L) → (B, N, d_model)
        x = x.permute(0, 2, 1)
        x = self.value_embedding(x)
        return self.dropout(x)


class FullAttention(nn.Module):
    """Standard scaled dot-product attention."""
    def __init__(self, attention_dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        # queries, keys, values: (B, H, N, d_k)
        B, H, N, d_k = queries.shape
        scale = 1.0 / math.sqrt(d_k)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale  # (B, H, N, N)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, values)  # (B, H, N, d_k)
        return out


class AttentionLayer(nn.Module):
    """Multi-head attention wrapper."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        d_k = d_model // n_heads

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn = FullAttention(dropout)

    def forward(self, queries, keys, values):
        B, N, _ = queries.shape
        H = self.n_heads
        d_k = self.d_model // H

        q = self.query_proj(queries).view(B, N, H, d_k).transpose(1, 2)
        k = self.key_proj(keys).view(B, N, H, d_k).transpose(1, 2)
        v = self.value_proj(values).view(B, N, H, d_k).transpose(1, 2)

        out = self.attn(q, k, v)  # (B, H, N, d_k)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.out_proj(out)


class EncoderLayer(nn.Module):
    """Transformer encoder layer: attention + FFN with residuals."""
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation='gelu'):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = AttentionLayer(d_model, n_heads, dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == 'gelu' else F.relu

    def forward(self, x):
        # Self-attention
        new_x = self.attention(x, x, x)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)

        # FFN (Conv1d = pointwise linear)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


# ══════════════════════════════════════════════════════════════════════════
# iTransformer
# ══════════════════════════════════════════════════════════════════════════

class iTransformer(nn.Module):
    """
    iTransformer for PM2.5 forecasting.

    Key idea: each VARIATE (feature channel) is a token.
    Attention learns cross-variate correlations.
    FFN learns temporal representations per variate.

    Input:  (batch, seq_len, num_variates)
    Output: (batch, pred_len, num_variates) — forecast for ALL variates
    """
    def __init__(self, seq_len=48, pred_len=1, enc_in=37,
                 d_model=128, n_heads=4, e_layers=2, d_ff=256,
                 dropout=0.1, activation='gelu', use_norm=True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.use_norm = use_norm

        # Inverted embedding: each variate → d_model
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, dropout)

        # Encoder
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, activation)
            for _ in range(e_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Projector: d_model → pred_len
        self.projector = nn.Linear(d_model, pred_len, bias=True)

    def forward(self, x):
        """
        x: (B, seq_len, N) where N = num_variates
        Returns: (B, pred_len, N)
        """
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev

        # Embedding: (B, L, N) → (B, N, d_model)
        enc_out = self.enc_embedding(x)

        # Encoder: (B, N, d_model) → (B, N, d_model)
        for layer in self.encoder:
            enc_out = layer(enc_out)
        enc_out = self.norm(enc_out)

        # Project: (B, N, d_model) → (B, N, pred_len) → (B, pred_len, N)
        dec_out = self.projector(enc_out).permute(0, 2, 1)

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out  # (B, pred_len, N)
