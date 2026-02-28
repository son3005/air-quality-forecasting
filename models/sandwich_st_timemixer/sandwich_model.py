"""
═══════════════════════════════════════════════════════════════
Sandwich ST-TimeMixer Model (ITERATION 2: TRỊ Liệu Nặng)
Kiến trúc: Multi-scale T -> Independent S -> Cross-Attention T
═══════════════════════════════════════════════════════════════
Lý do đập đi xây lại: Bản cũ ép các scale thời gian gom lại bằng 
interpolation làm nát bét Sequence. GCN chổi quét cào bằng sạch sẽ 
làm node Ninh Bình và Hà Nội sụp đổ.
Giải pháp: 
1. TimeMixer trả về List các scale riêng biệt.
2. GCN chạy đa luồng trên từng Scale (để giữ nguyên thứ bậc tần số).
3. Layer cuối dùng Transformer Decoder (Q là Tương lai, K/V là Quá khứ)
để chắt lọc thay vì dùng Flatten hay T-Pooling tầm thường.
═══════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
from models.parallel_st_timemixer.timemixer_branch import SeriesDecomposition, multiscale_downsample
from models.parallel_st_timemixer.stgcn_branch import ChebConv


class PastDecomMixingSequence(nn.Module):
    """Giữ nguyên từ bản cũ: Tách Trend và Seasonal trên từng Scale."""
    def __init__(self, seq_len: int, d_model: int, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.trend_proj = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.seasonal_proj = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.feat_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, trend: torch.Tensor, seasonal: torch.Tensor) -> torch.Tensor:
        z_trend = self.trend_proj(trend)
        z_seasonal = self.seasonal_proj(seasonal)
        z = self.feat_proj(torch.cat([z_trend, z_seasonal], dim=-1))
        return self.norm(z)


class MultiScaleTimeMixerLayer(nn.Module):
    """
    TEMPORAL 1 (Đã Sửa): Output ra một LIST thay vì gộp chung.
    Bảo toàn tính toàn vẹn của chuỗi chu kỳ ngắn (Fine) và dài (Coarse).
    """
    def __init__(self, seq_len: int, num_features: int, d_model: int, scales: list, decomp_kernel: int = 25, dropout: float = 0.1):
        super().__init__()
        self.scales = scales
        self.seq_len = seq_len
        self.d_model = d_model
        
        self.input_proj = nn.Linear(num_features, d_model)
        self.decomp = SeriesDecomposition(kernel_size=decomp_kernel)
        
        self.mixers = nn.ModuleDict()
        for s in scales:
            T_s = seq_len // s if s > 1 else seq_len
            self.mixers[str(s)] = PastDecomMixingSequence(T_s, d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> dict:
        # Input X: [B, N, T, F] -> [B*N, T, F]
        B, N, T, F = x.shape
        x_flat = x.reshape(B * N, T, F)
        x_proj = self.input_proj(x_flat) # [B*N, T, d_model]
        
        x_scales = multiscale_downsample(x_proj, self.scales)
        
        scale_outputs = {}
        for s in self.scales:
            x_s = x_scales[s]
            trend_s, seasonal_s = self.decomp(x_s)
            
            # z_s có dạng [B*N, T/s, d_model]
            z_s = self.mixers[str(s)](trend_s, seasonal_s) 
            
            # Trả lại [B, N, T/s, d_model] để đưa vào GCN
            T_s = z_s.shape[1]
            scale_outputs[s] = z_s.reshape(B, N, T_s, self.d_model)
            
        return scale_outputs # Dictionary chứa các dải tần riêng biệt


class MultiScaleSpatialGCNLayer(nn.Module):
    """
    SPATIAL (Đã sửa): Chạy Multi-thread GCN. 
    Mỗi dải tần số (Scale) sẽ bị lan truyền KHÔNG GIAN một cách riêng rẽ.
    Scale lớn (nhìn xa 6 tháng) lan truyền rộng. Scale nhỏ (nhìn 1 giờ) lan truyền hẹp.
    """
    def __init__(self, d_model: int, scales: list, seq_len: int, K_cheby: int = 3, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.scales = scales
        
        # Tạo GCN riêng biệt cho từng Scale!
        self.gcn_dict = nn.ModuleDict()
        self.pos_emb_dict = nn.ParameterDict()
        
        for s in scales:
            T_s = seq_len // s if s > 1 else seq_len
            self.gcn_dict[str(s)] = nn.ModuleList([
                ChebConv(d_in=d_model, d_out=d_model, K=K_cheby, dropout=dropout)
                for _ in range(num_layers)
            ])
            self.pos_emb_dict[str(s)] = nn.Parameter(torch.randn(1, 1, T_s, d_model))

    def forward(self, x_dict: dict, A: torch.Tensor) -> dict:
        out_dict = {}
        
        for s, x in x_dict.items():
            B, N, T_s, D = x.shape
            
            # Cấy ghép Positional Encoding
            x = x + self.pos_emb_dict[str(s)]
            
            # Chuẩn bị ma trận Graph: [B*T_s, N, D]
            x_space = x.permute(0, 2, 1, 3).reshape(B * T_s, N, D)
            
            # Luồng GCN cho scale này
            for gcn in self.gcn_dict[str(s)]:
                out_gcn = gcn(x_space, A) 
                x_space = x_space + out_gcn # Residual
                
            out_dict[s] = x_space.reshape(B, T_s, N, D).permute(0, 2, 1, 3) # [B, N, T_s, D]
            
        return out_dict


class TransformerDecoderLayer(nn.Module):
    """
    TEMPORAL 2 (Đã Sửa): Nâng cấp lên vũ khí hạng nặng.
    Thay vì đập bẹp quá khứ, ta coi Tương lai (Pred_len) là Query (Q).
    Quá khứ đa tỷ lệ (Multi-scale T-S) sẽ là Key, Value (K, V).
    Cross-Attention sẽ chọn lọc cực kỳ tinh vi: Trạm Ninh Bình hỏng thì nó tự né.
    """
    def __init__(self, d_model: int, pred_len: int, num_targets: int, num_scales: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.pred_len = pred_len
        self.num_targets = num_targets
        
        # Token tượng trưng cho Tương lai (Learnable Query)
        self.future_query = nn.Parameter(torch.randn(1, 1, pred_len, d_model))
        
        # Gom các scale lại bằng 1 lớp chập đặc trưng
        self.scale_fusion = nn.Linear(d_model * num_scales, d_model)
        
        # Cross Attention: Q (Future) đi nhặt K,V (Past)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        # Phát nổ (Projection ra Targets)
        self.out_proj = nn.Linear(d_model, num_targets)

    def forward(self, x_dict: dict, seq_len: int) -> torch.Tensor:
        # Trong dictionary x_dict, các scale có T_s khác nhau. 
        # Cần nội suy upsample nó về chuẩn T (chỉ upsample ở bước cuối cùng này để làm Context Length)
        
        B = list(x_dict.values())[0].shape[0]
        N = list(x_dict.values())[0].shape[1]
        D = list(x_dict.values())[0].shape[-1]
        
        upsampled_scales = []
        for s, x in x_dict.items():
            # x: [B, N, T_s, D] -> [B*N, D, T_s]
            x_reshaped = x.reshape(B * N, x.shape[2], D).permute(0, 2, 1)
            if x.shape[2] != seq_len:
                x_reshaped = torch.nn.functional.interpolate(x_reshaped, size=seq_len, mode='linear', align_corners=False)
            
            # [B*N, T, D]
            x_reshaped = x_reshaped.permute(0, 2, 1)
            upsampled_scales.append(x_reshaped)
            
        # Concat tất cả Scale lại dọc theo channel để Fussion
        # [B*N, T, D * num_scales] -> [B*N, T, D]
        memory_kv = torch.cat(upsampled_scales, dim=-1)
        memory_kv = self.scale_fusion(memory_kv) 
        
        # Căng Query Tương lai ra cho mọi Batch và Node
        # future_query: [1, 1, Pred_Len, D] -> Cần nảy mầm thành [B*N, Pred_Len, D]
        # Xóa bớt dimension thừa trước khi căng
        q_base = self.future_query.squeeze(0).squeeze(0) # [Pred_Len, D]
        q = q_base.unsqueeze(0).expand(B * N, -1, -1) # [B*N, Pred_Len, D]
        
        # Cross Attention Phase
        attn_out, _ = self.cross_attn(query=q, key=memory_kv, value=memory_kv)
        q = self.norm1(q + attn_out)
        
        # FFN Phase
        ffn_out = self.ffn(q)
        q = self.norm2(q + ffn_out) # [B*N, Pred_Len, D]
        
        # Projection
        preds = self.out_proj(q) # [B*N, Pred_Len, Targets]
        
        # Bung lại ra [B, N, Pred_Len, Targets]
        return preds.view(B, N, self.pred_len, self.num_targets)


class SandwichSTTimeMixer(nn.Module):
    """
    Bức tranh toàn cảnh MỚI: 
    1. TimeMixer xuất Dictionary các mức sóng.
    2. GCN luồn lách qua từng mức sóng (Không đụng chạm vào nhau).
    3. Transformer Decoder đứng ở tương lai và kéo dãn các mức sóng lên lấy số.
    """
    def __init__(self, config: dict):
        super().__init__()
        
        self.seq_len = config['seq_len']
        self.pred_len = config['pred_len']
        self.num_nodes = config['num_nodes']
        self.d_model = config['d_model']
        self.num_targets = config['num_targets']
        self.scales = config.get('scales', [1, 4, 24])
        
        self.adaptive_adj = config.get('adaptive_adj', True)
        if self.adaptive_adj:
            self.node_emb = nn.Parameter(torch.randn(self.num_nodes, 10))
        
        self.temporal_1 = MultiScaleTimeMixerLayer(
            seq_len=self.seq_len,
            num_features=config['num_features'],
            d_model=self.d_model,
            scales=self.scales,
            decomp_kernel=config.get('decomp_kernel', 25),
            dropout=config.get('dropout', 0.15)
        )
        
        self.spatial = MultiScaleSpatialGCNLayer(
            d_model=self.d_model,
            scales=self.scales,
            seq_len=self.seq_len,
            K_cheby=config.get('K_cheby', 3),
            num_layers=config.get('num_gcn_layers', 2),
            dropout=config.get('dropout', 0.15)
        )
        
        self.temporal_2 = TransformerDecoderLayer(
            d_model=self.d_model,
            pred_len=self.pred_len,
            num_targets=self.num_targets,
            num_scales=len(self.scales),
            n_heads=4,
            dropout=config.get('dropout', 0.15)
        )

    def forward(self, x: torch.Tensor, base_adj: torch.Tensor) -> torch.Tensor:
        if self.adaptive_adj:
            adj_adp = torch.relu(torch.mm(self.node_emb, self.node_emb.transpose(0, 1)))
            adj = base_adj + adj_adp
        else:
            adj = base_adj
            
        # 1. Multi-scale TimeMixer Dictionary
        dict_t1 = self.temporal_1(x) 
        
        # 2. Multi-channel Graph Propagation
        dict_space = self.spatial(dict_t1, adj)
        
        # 3. Transformer Decoder Focus
        y_pred = self.temporal_2(dict_space, self.seq_len) 
        
        return y_pred
