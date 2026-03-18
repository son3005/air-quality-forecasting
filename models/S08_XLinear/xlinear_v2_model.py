"""
XLinear V2: Cai tien kien truc XLinear cho bai toan Air Quality Forecasting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
3 cai tien chinh:
  A1. Multi-Scale Temporal Projection (Conv kernels thay vi 1 Linear)
  A2. Deeper Cross-attention (3 Gating Blocks xen ke)
  A3. Horizon-aware Prediction Heads (4 heads rieng cho T+1,6,12,24)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleTemporalProjection(nn.Module):
    """
    A1: Thay Linear(seq_len, d_model) bang nhieu Conv1d kernels o scale khac nhau.
    Moi kernel bat signal o do phan giai thoi gian khac nhau:
      - kernel=3: pattern ngan (3h, nhu ngay/dem)
      - kernel=12: pattern trung binh (12h, chu ky nua ngay)
      - kernel=24: pattern dai (24h, chu ky ngay)
      - kernel=48: pattern cuc dai (2 ngay)
    """
    def __init__(self, seq_len, d_model, dropout=0.1):
        super().__init__()
        kernels = [3, 6, 12, 24, 48]
        d_per_scale = d_model // len(kernels)
        remainder = d_model - d_per_scale * len(kernels)
        
        self.convs = nn.ModuleList()
        self.dims = []
        for i, k in enumerate(kernels):
            out_d = d_per_scale + (1 if i < remainder else 0)
            self.dims.append(out_d)
            # Conv1d: (B, 1, seq_len) -> (B, 1, out_d)
            self.convs.append(nn.Sequential(
                nn.Conv1d(1, out_d, kernel_size=min(k, seq_len), 
                         padding=0, stride=1),
                nn.AdaptiveAvgPool1d(1),  # Compress to 1 value per channel
            ))
        
        # Final projection to get exactly d_model dims
        total_d = sum(self.dims)
        self.proj = nn.Linear(total_d, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        x: (B, C, L) where C=channels, L=seq_len
        returns: (B, C, d_model)
        """
        B, C, L = x.shape
        # Process each channel independently through multi-scale convs
        x_flat = x.reshape(B * C, 1, L)  # (B*C, 1, L)
        
        scale_outs = []
        for conv in self.convs:
            out = conv(x_flat)            # (B*C, d_i, 1)
            out = out.squeeze(-1)         # (B*C, d_i)
            scale_outs.append(out)
        
        multi = torch.cat(scale_outs, dim=-1)  # (B*C, total_d)
        projected = self.proj(multi)            # (B*C, d_model)
        projected = projected.reshape(B, C, -1) # (B, C, d_model)
        
        return self.norm(self.dropout(projected))


class DeepGatingBlock(nn.Module):
    """
    A2: Deeper version of Gating_Block. 
    Stack 3 gating layers with residual connections.
    """
    def __init__(self, d_model, hidden_ff, dropout=0.1, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(d_model, hidden_ff),
                nn.GELU(),       # GELU thay vi ReLU - smooth nhung van giu duoc gradient
                nn.Dropout(dropout),
                nn.Linear(hidden_ff, d_model),
                nn.Sigmoid()
            ))
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        residual = x
        for gate_layer in self.layers:
            weight = gate_layer(x)
            x = x * weight  # Element-wise gating
        return self.norm(x + residual)  # Residual connection


class XLinearV2(nn.Module):
    """
    XLinear V2: Ket hop A1 + A2 + A3
    - Multi-scale temporal projection
    - Deep 3-layer gating for both temporal + cross-variable
    - Horizon-aware prediction head
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.channel = configs.enc_in
        self.norm_enabled = configs.usenorm
        self.horizon = getattr(configs, 'horizon', 24)
        
        # A1: Multi-Scale Temporal Projection
        self.projection = MultiScaleTemporalProjection(
            self.seq_len, self.d_model, 
            dropout=configs.embed_dropout
        )
        
        # Global learnable token
        self.global_token = nn.Parameter(
            torch.randn(1, 1, self.d_model) * 0.02
        )
        
        # A2: Deep Gating (3 layers each)
        self.temporal_gate = DeepGatingBlock(
            2 * self.d_model, configs.t_ff, configs.t_dropout, n_layers=3
        )
        self.cross_gate = DeepGatingBlock(
            self.channel, configs.c_ff, configs.c_dropout, n_layers=3
        )
        
        # A3: Horizon-aware head
        # Different linear heads for different forecast horizons
        self.heads = nn.ModuleDict({
            '1':  nn.Sequential(nn.Dropout(configs.head_dropout), nn.Linear(2 * self.d_model, 1)),
            '6':  nn.Sequential(nn.Dropout(configs.head_dropout), nn.Linear(2 * self.d_model, 1)),
            '12': nn.Sequential(nn.Dropout(configs.head_dropout), nn.Linear(2 * self.d_model, 1)),
            '24': nn.Sequential(nn.Dropout(configs.head_dropout), nn.Linear(2 * self.d_model, 1)),
        })
        # Fallback head
        self.default_head = nn.Sequential(
            nn.Dropout(configs.head_dropout),
            nn.Linear(2 * self.d_model, self.pred_len)
        )
        
    def forward(self, x_enc):
        """
        x_enc: (B, seq_len, channels) — standard time series input
        returns: (B, pred_len, 1)
        """
        # Instance normalization (optional)
        if self.norm_enabled:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        
        # Permute: (B, L, C) -> (B, C, L) for projection
        x = x_enc.permute(0, 2, 1)   # (B, C, L)
        
        # A1: Multi-scale projection
        embed = self.projection(x)     # (B, C, d_model)
        
        # Split endogenous (last channel) and exogenous (rest)
        en = embed[:, -1:, :]          # (B, 1, d)
        ex = embed[:, :-1, :]          # (B, C-1, d)
        
        # Global token interaction
        b = en.shape[0]
        glob = self.global_token.repeat(b, 1, 1)   # (B, 1, d)
        en_d = torch.cat([en, glob], dim=-1)        # (B, 1, 2d)
        
        # A2: Deep temporal gating
        en_atten = self.temporal_gate(en_d)          # (B, 1, 2d)
        
        origin_atten = en_atten[:, :, :self.d_model] # (B, 1, d)
        glob_out = en_atten[:, :, self.d_model:]     # (B, 1, d)
        
        # Cross-variable interaction 
        ex_d = torch.cat([ex, glob_out], dim=1)      # (B, C, d)
        ex_atten = self.cross_gate(ex_d.permute(0, 2, 1))  # (B, d, C)
        
        glob_cross = ex_atten[:, :, -1:]             # (B, d, 1)
        en_final = torch.cat(
            [origin_atten, glob_cross.permute(0, 2, 1)], dim=-1
        )  # (B, 1, 2d)
        
        # A3: Horizon-aware prediction
        h_key = str(self.horizon)
        if h_key in self.heads:
            dec_out = self.heads[h_key](en_final)    # (B, 1, 1)
        else:
            dec_out = self.default_head(en_final)     # (B, 1, pred_len)
        
        dec_out = dec_out.permute(0, 2, 1)            # (B, pred_len, 1)
        
        # De-normalize
        if self.norm_enabled:
            dec_out = dec_out * stdev[:, 0, -1:].unsqueeze(1).repeat(1, dec_out.shape[1], 1)
            dec_out = dec_out + means[:, 0, -1:].unsqueeze(1).repeat(1, dec_out.shape[1], 1)
        
        return dec_out
