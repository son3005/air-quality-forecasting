import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeBlock(nn.Module):
    """
    Temporal Convolutional Layer (1D-CNN) with Gated Linear Unit (GLU).
    Theo Paper: P_i = (X * W_1 + b_1) ⊗ σ(X * W_2 + b_2)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TimeBlock, self).__init__()
        
        # Conv1d expects (batch, channels, spatial_dim, sequence_length)
        # We will reshape our input (bat, nodes, feat, seq) -> (bat*nodes, feat, seq) for 1D conv
        
        # 1D Padding to keep sequence length same (if stride=1)
        padding = (kernel_size - 1) // 2
        
        # 2 Convolution lines for GLU gating mechanism
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        
        # Residual mapping if feature dimensions change
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, X):
        # Expected X: (batch * num_nodes, in_channels, seq_len)
        res = self.residual(X)
        
        # GLU Operation
        x1 = self.conv1(X)
        x2 = torch.sigmoid(self.conv2(X))
        
        return res + (x1 * x2)

class SpatialBlock(nn.Module):
    """
    Standard Spatial Graph Convolution logic using the Adjacency Matrix
    Paper: Θ*x = \sum (θ_k (L^k) x) -> Reduced to 1-hop: H = σ(D^-1/2 A D^-1/2 X W)
    """
    def __init__(self, in_channels, out_channels, num_nodes):
        super(SpatialBlock, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, X, A):
        # X: (batch, seq_len, num_nodes, in_channels)
        # A: (num_nodes, num_nodes)
        
        # XW -> (bat, seq, nodes, out_channels)
        xW = torch.matmul(X, self.weight)
        
        # A * XW -> Aggregate neighbors
        out = torch.einsum('bsni,nm->bsmi', xW, A)
        
        return F.relu(out)

class STGCNBlock(nn.Module):
    """
    Spatio-Temporal Convolutional Block
    Sandwich Structure: Temporal -> Spatial -> Temporal -> BatchNorm
    """
    def __init__(self, in_channels, spatial_channels, out_channels, num_nodes):
        super(STGCNBlock, self).__init__()
        
        self.num_nodes = num_nodes
        
        self.temp_conv1 = TimeBlock(in_channels, spatial_channels)
        self.spatial_conv = SpatialBlock(spatial_channels, spatial_channels, num_nodes)
        self.temp_conv2 = TimeBlock(spatial_channels, out_channels)
        
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        
    def forward(self, X, A):
        # Original Input X: (batch, seq_len, num_nodes, features)
        b, t, n, c = X.shape
        
        # --- 1. Temporal Conv 1 ---
        # Reshape to (batch * nodes, features, seq_len) for 1D Conv
        x_t1 = X.permute(0, 2, 3, 1).contiguous().view(b * n, c, t)
        x_t1 = self.temp_conv1(x_t1) # (b*n, spat_c, t)
        
        # --- 2. Spatial Conv ---
        # Reshape back to (batch, seq_len, nodes, spat_c)
        x_s = x_t1.view(b, n, -1, t).permute(0, 3, 1, 2).contiguous()
        x_s = self.spatial_conv(x_s, A) # (b, t, n, spat_c)
        
        # --- 3. Temporal Conv 2 ---
        x_t2 = x_s.permute(0, 2, 3, 1).contiguous().view(b * n, -1, t)
        x_t2 = self.temp_conv2(x_t2) # (b*n, out_c, t)
        
        # --- 4. Layer Norm/BatchNorm ---
        out = x_t2.view(b, n, -1, t).permute(0, 1, 2, 3).contiguous() # (bat, nodes, out_c, seq_len)
        out = self.batch_norm(out)
        
        # Return format expected for next block: (batch, seq_len, nodes, out_c)
        out = out.permute(0, 3, 1, 2).contiguous()
        return out

class STGCN(nn.Module):
    """
    STGCN: Spatio-Temporal Graph Convolutional Networks (rxiv:1709.04875)
    Architecture: ST-Conv Block x2 -> Output Layer (Temporal bottleneck -> FC)
    """
    def __init__(self, num_nodes, num_features, seq_len, pred_len):
        super(STGCN, self).__init__()
        
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        
        # The Paper uses two ST-Conv Blocks
        # In_Feat -> (STBlock 1) -> 64 -> 16 -> 64 -> (STBlock 2) -> 64 -> 16 -> 64
        self.block1 = STGCNBlock(in_channels=num_features, spatial_channels=16, out_channels=64, num_nodes=num_nodes)
        self.block2 = STGCNBlock(in_channels=64, spatial_channels=16, out_channels=64, num_nodes=num_nodes)
        
        # Output Layer: 
        # 1. Bottleneck Temporal Conv to reduce temporal dimension to 1 channel safely
        self.out_temp = TimeBlock(in_channels=64, out_channels=128)
        
        # 2. Fully connected layer over the flattened temporal dimension
        self.fc = nn.Linear(128 * seq_len, pred_len)

    def forward(self, x, adj, x_future_weather=None):
        # x is originally (batch, seq_len, num_nodes, num_features)
        
        # Pass through ST-Blocks
        x_st1 = self.block1(x, adj) # (bat, seq_len, nodes, 64)
        x_st2 = self.block2(x_st1, adj) # (bat, seq_len, nodes, 64)
        
        b, t, n, c = x_st2.shape
        
        # Reshape for Temporal Bottleneck
        x_out = x_st2.permute(0, 2, 3, 1).contiguous().view(b * n, c, t)
        x_out = self.out_temp(x_out) # (b*n, 128, t)
        
        # Flatten sequence and channel for FC
        x_flat = x_out.view(b * n, -1) # (b*n, 128 * t)
        
        # Project to Future Horizons
        out = self.fc(x_flat) # (b*n, pred_len)
        
        # Reshape exactly to (batch_size, pred_len, num_nodes) for Evaluation
        out = out.view(b, n, -1).permute(0, 2, 1).contiguous()
        
        return out
