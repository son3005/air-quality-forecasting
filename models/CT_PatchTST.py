import torch
import torch.nn as nn
import torch.nn.functional as F

class RevIN(nn.Module):
    """
    Reversible Instance Normalization - Chuẩn hóa dữ liệu đầu vào và có thể đảo ngược
    """
    def __init__(self, num_features:int, eps:float = float("-inf"), affine: bool = True):
        super(RevIN, self).__init__()
        self.num_features = num_features  # Số lượng features/channels
        self.eps = eps  # Epsilon để tránh chia cho 0
        self.affine = affine  # Có sử dụng affine transformation hay không

        if self.affine:
            # Khởi tạo weight và bias cho affine transformation
            self.affine_weight = nn.Parameter(torch.ones(num_features,device='cuda' if torch.cuda.is_available() else 'cpu'))
            self.affine_bias = nn.Parameter(torch.zeros(num_features,device='cuda' if torch.cuda.is_available() else 'cpu'))
        
    def forward(self, x, mode: str = None):
        """
        x: input tensor
        mode: 'norm' để chuẩn hóa, 'denorm' để đảo ngược chuẩn hóa
        """
        if mode == "norm":
            self._get_statistic(x)  # Tính mean và std
            x = self._normalize(x)  # Chuẩn hóa
        elif mode == "denorm":
            x = self._denormalize(x)  # Đảo ngược chuẩn hóa
        return x
    
    def _get_statistic(self, x):
        """Tính toán mean và standard deviation của input"""
        # Tính mean theo chiều thời gian (dim=1), giữ lại chiều để broadcast
        self.mean = torch.mean(x, dim = 1, keepdim = True).detach()
        # Tính standard deviation theo chiều thời gian
        self.stdev = torch.std(x, dim = 1, keepdim = True, unbiased=False).detach()
    
    def _normalize(self, x):
        """Thực hiện chuẩn hóa: (x - mean) / std"""
        x -= self.mean  # Trừ mean
        x /= (self.stdev + self.eps)  # Chia cho std (+ eps để tránh chia 0)
        if self.affine:
            # Áp dụng affine transformation: weight * x + bias
            x = x * self.affine_weight + self.affine_bias

        return x

class PatchEbedding(nn.Module):
    """
    Chia time series thành các patches và embedding chúng
    """
    def __init__(self, patch_len: int , stride: int, d_model: int):
        super(PatchEbedding, self).__init__()
        self.patch_len = patch_len  # Độ dài của mỗi patch
        self.stride = stride  # Bước nhảy giữa các patch
        # Linear layer để project patch từ patch_len chiều lên d_model chiều
        self.projection = nn.Linear(patch_len, d_model, device='cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        # x shape: (B, L, M) -> (B, M, L) - Đổi channels và time steps
        x = x.permute(0, 2, 1) 
        # Unfold để tạo patches: chia time series thành các đoạn có độ dài patch_len
        x_unfold = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        # Project mỗi patch lên không gian d_model chiều
        x_out = self.projection(x_unfold)
        return x_out

class ChannelTimeEncoderLayer(nn.Module):
    """
    Encoder layer kết hợp Channel Attention và Time Attention
    """
    def __init__(self, d_model, n_head_channel, n_head_time, d_ff, dropout=0.1):
        super(ChannelTimeEncoderLayer, self).__init__()
        # Multi-head attention cho Channel dimension
        self.channel_attn = nn.MultiheadAttention(d_model, n_head_channel, dropout=dropout, batch_first=True, device='cuda' if torch.cuda.is_available() else 'cpu')
        # Layer normalization
        self.norm = nn.LayerNorm(d_model, device='cuda' if torch.cuda.is_available() else 'cpu')
        # Dropout layer
        self.dropout =  nn.Dropout(dropout)

        # Multi-head attention cho Time dimension
        self.time_attn = nn.MultiheadAttention(d_model, n_head_time, dropout=dropout, batch_first=True, device='cuda' if torch.cuda.is_available() else 'cpu')
        # Feed-forward network
        self.ffn = nn.Sequential(  # Sửa lỗi chính tả: Sequential thay vì Sequqential
            nn.Linear(d_model, d_ff, device='cuda' if torch.cuda.is_available() else 'cpu'),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, device='cuda' if torch.cuda.is_available() else 'cpu')
        )

    def forward(self, input, target=None):
        B, M, N, D= input.shape  # B: batch, M: channels, N: num_patches, D: d_model
        
        # 1. Channel Attention - Học mối quan hệ giữa các channels
        # Reshape để áp dụng attention trên channel dimension
        x_c = input.permute(0, 2, 1, 3).reshape(B*N, M, D)  # (B*N, M, D)
        # Self Attention giữa các channels
        attn_output_c, _ = self.channel_attn(x_c, x_c, x_c)
        # Residual connection + Dropout + Layer Norm
        x_c = self.norm(x_c + self.dropout(attn_output_c))
        # Reshape về dạng ban đầu
        x_c  = x_c.reshape(B, N, M, D).permute(0, 2, 1, 3)  # (B, M, N, D)

        # Residual connection từ input
        x = input + x_c  # Sửa lỗi: x_c + x_c thành x + x_c
        
        # 2. Time Attention - Học mối quan hệ giữa các time patches
        # Reshape để áp dụng attention trên time dimension
        x_t = x.reshape(B*M, N, D)  # (B*M, N, D)
        # Self Attention giữa các time patches
        attn_output_t, _ = self.time_attn(x_t, x_t, x_t)
        # Residual connection + Dropout + Layer Norm
        x_t = self.norm(x_t + self.dropout(attn_output_t))
        # Reshape về dạng ban đầu
        x_t = x_t.reshape(B, M, N, D)
        # Residual connection
        x = x + x_t
        
        # 3. Feed Forward Network - Xử lý phi tuyến
        x_ffn = self.ffn(x)
        # Residual connection + Dropout + Layer Norm
        x = self.norm(x + self.dropout(x_ffn))

        return x

class CT_PatchTST(nn.Module):
    """
    Channel-Time Patch Time Series Transformer
    Model kết hợp patching và dual attention (channel + time) cho forecasting
    """
    def __init__(self, seq_len:int, pred_len:int, enc_in: int, patch_len: int, stride: int, 
                 d_model:int, e_layers:int, n_head_channel:int, n_head_time:int, d_ff:int, dropout:float=0.1):
        super(CT_PatchTST, self).__init__()
        self.seq_len = seq_len  # Độ dài chuỗi đầu vào
        self.pred_len = pred_len  # Độ dài chuỗi dự đoán
        self.enc_in = enc_in  # Số lượng input channels/features
        self.patch_len = patch_len  # Độ dài mỗi patch
        self.stride = stride  # Bước nhảy giữa các patches
        self.d_model = d_model  # Số chiều embedding
        self.e_layers = e_layers  # Số lượng encoder layers
        self.n_head_channel = n_head_channel  # Số heads cho channel attention
        self.n_head_time = n_head_time  # Số heads cho time attention
        self.d_ff = d_ff  # Số chiều của feed-forward network
        self.dropout = dropout  # Tỷ lệ dropout

        # 1. RevIN - Chuẩn hóa đầu vào
        self.revin = RevIN(self.enc_in)

        # 2. Patch Embedding
        # Tính toán số lượng patches được tạo từ chuỗi đầu vào
        self.num_patches = int((self.seq_len - self.patch_len) // self.stride + 1)
        # Khởi tạo lớp Patch Embedding để chia và embed patches
        self.patch_embedding = PatchEbedding(self.patch_len, self.stride, self.d_model)
        # Positional Encoding - Thêm thông tin vị trí cho các patches
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, self.num_patches, self.d_model, device='cuda' if torch.cuda.is_available() else 'cpu'))
        # Dropout sau khi thêm Positional Encoding
        self.dropout_layer = nn.Dropout(self.dropout)

        # 3. Channel-Time Transformer Encoder - Stack nhiều encoder layers
        self.encoder_layers = nn.ModuleList([
            ChannelTimeEncoderLayer(self.d_model, self.n_head_channel, self.n_head_time, self.d_ff, self.dropout)
            for _ in range(self.e_layers)
        ])

        # 4. Prediction Head - Linear layer để dự đoán
        # Input: tất cả patches được flatten, Output: pred_len time steps
        self.head = nn.Linear(self.num_patches * self.d_model, self.pred_len, device='cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, input):
        # x shape: (B, L, M) - Batch, Length, Channels
        
        # 1. RevIN Normalization - Chuẩn hóa input
        x = self.revin(input, mode="norm")
        # 2. Patching & Embedding
        x_enc = self.patch_embedding(x)  # Chia thành patches và embed
        x_enc = x_enc + self.pos_embedding  # Thêm positional encoding
        x_enc = self.dropout_layer(x_enc)  # Áp dụng dropout

        # 3. Encoder - Đưa qua các encoder layers
        for layer in self.encoder_layers:
            x_enc = layer(x_enc)
        
        # 4. Dự đoán
        B, M, N, D = x_enc.shape  # B: batch, M: channels, N: patches, D: d_model
        # Flatten patches và d_model dimensions
        x_out = x_enc.reshape(B, M, N*D)
        # Linear projection để dự đoán pred_len time steps
        x_out = self.head(x_out)

        # Chuyển vị về định dạng (B, pred_len, M)
        x_out = x_out.permute(0, 2, 1)

        # De-normalization RevIN - Đảo ngược chuẩn hóa
        x_out = self.revin(x_out, mode="denorm")
        return x_out
