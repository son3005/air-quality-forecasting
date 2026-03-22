"""
models/STXLinear/model.py  (v5)

ST-XLinear: Per-Horizon + Spatial Global Token.

Kiến trúc:
  - XLinear Forcast_with_exogenous GIỮ NGUYÊN 100% (temporal pipeline thuần túy)
  - GCN tạo Spatial Global Token THAY THẾ learnable global token
  - pred_len=1, train riêng mỗi horizon (như XLinear gốc)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialGCN(nn.Module):
    """GCN trên temporal summaries → spatial global token per node."""
    def __init__(self, d_model):
        super().__init__()
        self.w = nn.Parameter(torch.FloatTensor(d_model, d_model))
        nn.init.xavier_uniform_(self.w)
        self.proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, adj):
        """x: (b, n, d), adj: (n, n) → (b, n, d)"""
        h = torch.matmul(x, self.w)
        agg = torch.matmul(adj, h)
        out = self.proj(torch.cat([h, agg], dim=-1))
        return self.norm(F.gelu(out + x))


class GatingBlock(nn.Module):
    """sigma(MLP(x)) * x — giữ nguyên XLinear paper."""
    def __init__(self, d, hf, drop=0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d, hf), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hf, d), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.gate(x)


class STXLinear(nn.Module):
    """
    Per-horizon ST-XLinear: XLinear backbone + Spatial Global Token.

    Input:  (batch, seq_len, num_nodes, num_features)  [PM2.5 = last channel]
    Output: (batch, 1, num_nodes)                       [PM2.5 prediction]
    """
    def __init__(self, num_nodes, num_features, seq_len=48, pred_len=1,
                 d_model=128, t_ff=256, c_ff=256, dropout=0.1, **kwargs):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.pred_len = pred_len

        # Temporal embedding (shared across channels, per XLinear paper)
        self.projection = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.Dropout(dropout)
        )

        # Node summary → attention pooling over features
        self.feat_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1)
        )

        # Spatial GCN → produces spatial global token per node
        self.spatial_gcn = SpatialGCN(d_model)

        # TGM: Time-wise gating [endogenous, spatial_token] → 2*d_model
        self.tgm = GatingBlock(2 * d_model, t_ff, dropout)

        # VGM: Cross-channel gating
        self.vgm = GatingBlock(num_features, c_ff, dropout)

        # Prediction Head (pred_len=1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, pred_len)
        )

    def forward(self, x, adj):
        b, t, n, f = x.shape

        # 1. Temporal embedding per-node per-channel
        x_flat = x.permute(0, 2, 3, 1).reshape(b * n, f, t)
        embed = self.projection(x_flat)              # (b*n, f, d)

        # 2. Node summary via attention pooling
        scores = self.feat_pool(embed)               # (b*n, f, 1)
        weights = F.softmax(scores, dim=1)           # (b*n, f, 1)
        summary = (embed * weights).sum(dim=1)       # (b*n, d)
        summary = summary.view(b, n, self.d_model)   # (b, n, d)

        # 3. Spatial GCN → spatial global token
        spatial_tok = self.spatial_gcn(summary, adj)  # (b, n, d)
        spatial_tok = spatial_tok.reshape(b * n, 1, self.d_model)

        # 4. XLinear Forcast_with_exogenous
        en = embed[:, -1:, :]                        # endogenous: PM2.5
        ex = embed[:, :-1, :]                        # exogenous: everything else

        en_d = torch.cat([en, spatial_tok], dim=-1)  # (b*n, 1, 2d)
        en_atten = self.tgm(en_d)

        origin = en_atten[:, :, :self.d_model]
        glob = en_atten[:, :, self.d_model:]

        ex_d = torch.cat([ex, glob], dim=1)          # (b*n, f, d)
        ex_a = self.vgm(ex_d.permute(0, 2, 1))      # (b*n, d, f)
        glob_ref = ex_a[:, :, -1:]                   # (b*n, d, 1)

        final = torch.cat([origin, glob_ref.permute(0, 2, 1)], dim=-1)

        # 5. Predict
        pred = self.head(final).squeeze(1)           # (b*n, pred_len)
        return pred.view(b, n, self.pred_len).permute(0, 2, 1)
