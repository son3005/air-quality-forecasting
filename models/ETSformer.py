import torch
import torch.nn as nn
import torch.fft as fft
from einops import rearrange, repeat

# Hàm tính tích chập 1D bằng FFT để tăng tốc độ tính toán
def conv1d_fft(u, v, dim=-1):
    # u: input signal, v: kernel (weights)
    N = u.size(dim)
    M = v.size(dim)
    
    # Padding để thực hiện FFT
    fast_len = N + M - 1
    
    # Chuyển sang miền tần số (Frequency Domain)
    F_u = fft.rfft(u, n=fast_len, dim=dim)
    F_v = fft.rfft(v, n=fast_len, dim=dim)
    
    # Nhân trong miền tần số = Tích chập trong miền thời gian
    F_uv = F_u * F_v
    
    # Chuyển ngược lại miền thời gian
    out = fft.irfft(F_uv, n=fast_len, dim=dim)
    
    # Cắt lấy phần kích thước mong muốn (bỏ padding thừa)
    out = out[..., :N] 
    return out


class ExponentialSmoothingAttention(nn.Module):
    def __init__(self, model_dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.model_dim = model_dim
        
        self.alpha = nn.Parameter(torch.tensor(0.5))
        # Initial state v0: Shape (1, Heads, D_head)
        self.v0 = nn.Parameter(torch.randn(1, heads, model_dim // heads))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        H = self.heads

        # 1. Chia head
        x = rearrange(x, 'b n (h d) -> b h n d', h=H)

        # 2. Tạo kernel weights
        powers = torch.arange(N, device=x.device)
        alpha = torch.sigmoid(self.alpha)
        weights = alpha * (1 - alpha) ** torch.flip(powers, dims=(0,))

        # 3. Convolution FFT
        x_trans = rearrange(x, 'b h n d -> (b h) d n') 
        output = conv1d_fft(x_trans, weights, dim=-1)
        output = rearrange(output, '(b h) d n -> b h n d', h=H)

        # 4. Cộng Initial State (v0)
        # SỬA LỖI Ở ĐÂY: Thay '1' bằng 'o' (hoặc ký tự bất kỳ)
        init_weights = (1 - alpha) ** (powers + 1)
        
        # 'n' là init_weights (N,)
        # 'o h d' là self.v0 (1, H, D) -> dùng 'o' đại diện cho chiều size=1
        # output mong muốn: (1, H, N, D) -> 'o h n d'
        init_part = torch.einsum('n, o h d -> o h n d', init_weights, self.v0)

        # PyTorch sẽ tự động broadcast chiều 'o' (size 1) thành 'B' khi cộng
        out = output + init_part
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.dropout(out)
    
class FrequencyAttention(nn.Module):
    def __init__(self, model_dim, heads=8, top_k=4, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.heads = heads
        self.top_k = top_k # K frequencies lớn nhất để giữ lại
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, d = x.shape
        
        # Chuyển sang miền tần số
        x_freq = torch.fft.rfft(x, dim=1)
        
        # Tính biên độ (amplitude)
        amplitudes = torch.abs(x_freq)
        
        # Chọn Top-K tần số có biên độ lớn nhất
        # Logic: Giữ lại các tần số chính, lọc bỏ nhiễu (noise)
        if self.top_k > 0:
            topk_values, topk_indices = torch.topk(amplitudes, k=min(self.top_k, amplitudes.size(1)), dim=1)
            # Tạo mask chỉ giữ lại top-k
            mask = torch.zeros_like(amplitudes)
            mask.scatter_(1, topk_indices, 1.0)
            x_freq = x_freq * mask

        # Chuyển ngược lại miền thời gian
        out = fft.irfft(x_freq, n=n, dim=1)
        return self.dropout(out)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.Sigmoid(), # Paper dùng Sigmoid cho Gating
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )
    def forward(self, x):
        return self.net(x)

class ETSFormer(nn.Module):
    def __init__(self, 
                 time_features=1,    # Số lượng feature input (ví dụ: chỉ có PM2.5 thì là 1)
                 model_dim=512,      # Hidden dimension (Paper dùng 512)
                 embed_kernel_size=3,
                 layers=2,           # Số lớp Encoder
                 heads=8,
                 top_k=4,            # Số tần số giữ lại
                 dropout=0.2,
                 pred_len=24):       # Độ dài dự báo (Forecast Horizon)
        super().__init__()
        
        self.pred_len = pred_len
        self.model_dim = model_dim
        
        # 1. Input Embedding
        self.embedding = nn.Conv1d(time_features, model_dim, kernel_size=embed_kernel_size, padding=embed_kernel_size//2)
        
        # 2. Encoder Blocks
        self.encoders = nn.ModuleList([])
        for _ in range(layers):
            self.encoders.append(nn.ModuleDict({
                'freq_attn': FrequencyAttention(model_dim, heads, top_k, dropout),
                'esa': ExponentialSmoothingAttention(model_dim, heads, dropout),
                'ff': FeedForward(model_dim, dropout=dropout),
                'norm': nn.LayerNorm(model_dim)
            }))
            
        # 3. Output Projection
        self.out_proj = nn.Linear(model_dim, time_features)
        
        # Damping factor cho Growth (Xu hướng tăng trưởng)
        self.growth_damping = nn.Parameter(torch.ones(1) * 0.9) # Phi trong paper

    def forward(self, x):
        # x: (Batch, Seq_Len, Features)
        b, n, f = x.shape
        
        # Embedding (Conv1d cần input dạng B, C, T)
        x_embed = rearrange(x, 'b n f -> b f n')
        res = self.embedding(x_embed)
        res = rearrange(res, 'b f n -> b n f') # Residual state
        
        level_components = []
        growth_components = []
        seasonal_components = []
        
        # --- ENCODER FLOW ---
        # Paper [cite: 97-98]: Tách dần Seasonality và Growth qua từng lớp
        for layer in self.encoders:
            # 1. Tách Seasonality
            season = layer['freq_attn'](res)
            res = res - season # Remove seasonality khỏi residual
            
            # 2. Tách Growth (Trend) dùng ESA
            growth = layer['esa'](res)
            res = res - growth # Remove growth
            
            # 3. FeedForward & Norm cho phần dư
            res = layer['norm'](res + layer['ff'](res))
            
            seasonal_components.append(season)
            growth_components.append(growth)
            
        # --- DECODER / FORECASTING ---
        # Theo Paper[cite: 84]: Forecast = Level + Growth (damped) + Seasonality
        
        # 1. Tính Level (đơn giản hóa bằng trung bình mượt hoặc lấy last state)
        # Trong code lucidrains họ dùng input gốc làm base level
        level_pred = x[:, -1:, :] # Lấy giá trị cuối làm level nền (naive)
        level_pred = repeat(level_pred, 'b 1 f -> b h f', h=self.pred_len)
        
        # 2. Dự báo Growth (Trend) với Damping
        # Growth dự báo = Growth cuối cùng * hệ số tắt dần (damping)
        final_growth = growth_components[-1][:, -1:, :] # (B, 1, D)
        final_growth = self.out_proj(final_growth)      # Map về feature space
        
        # Tạo chuỗi damping [phi, phi^2, phi^3...]
        damping_steps = torch.arange(1, self.pred_len + 1, device=x.device)
        damping_factors = self.growth_damping ** damping_steps
        growth_pred = final_growth * damping_factors.view(1, -1, 1)
        
        # 3. Dự báo Seasonality (Extrapolate từ Frequency Attention)
        # Lấy thành phần mùa vụ từ lớp cuối, ngoại suy ra tương lai
        final_season_latent = seasonal_components[-1] # (B, N, D)
        # Đơn giản hóa: Cắt lấy đoạn đuôi hoặc dùng FA để generate tiếp
        # Ở đây tôi dùng Fourier Extrapolation đơn giản như logic của paper
        season_pred_latent = fft.rfft(final_season_latent, dim=1)
        season_pred = fft.irfft(season_pred_latent, n=self.pred_len, dim=1)
        season_pred = self.out_proj(season_pred)

        # TỔNG HỢP
        forecast = level_pred + growth_pred + season_pred
        
        return forecast