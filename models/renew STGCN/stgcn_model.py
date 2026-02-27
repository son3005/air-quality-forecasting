import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeBlock(nn.Module):
    """
    Neural networks block that applies a temporal convolution to each node of
    a graph individually.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        X: (batch_size, num_features, num_nodes, seq_len)
        return: (batch_size, out_channels, num_nodes, seq_len - kernel_size + 1)
        """
        # Gating Mechanism (GLU)
        v1 = self.conv1(X)
        v2 = torch.sigmoid(self.conv2(X))
        residual = self.conv3(X)
        return residual + v1 * v2


class STGCNBlock(nn.Module):
    """
    Neural networks block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels, num_nodes):
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels, out_channels=out_channels)
        self.theta = nn.Parameter(torch.FloatTensor(out_channels, spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels, out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta)

    def forward(self, X, A_hat):
        """
        X: (batch_size, in_channels, num_nodes, seq_len)
        A_hat: (num_nodes, num_nodes) Graph normalized adjacency matrix
        return: (batch_size, out_channels, num_nodes, seq_len - 2 * kernel_size + 2)
        """
        # Temporal Conv 1
        t = self.temporal1(X)  # (batch_size, out_channels, num_nodes, seq_len-2)

        # Spatial Graph Conv
        # t: (batch_size, out_channels, num_nodes, seq_len-2) -> Permute -> (batch_size, seq_len-2, num_nodes, out_channels)
        t_permuted = t.permute(0, 3, 2, 1)

        # Matmul with A_hat
        # t_permuted: (batch_size, seq_len-2, num_nodes, out_channels)
        # A_hat: (num_nodes, num_nodes)
        # Result -> (batch_size, seq_len-2, num_nodes, out_channels)
        out = torch.einsum('btnd,nm->btmd', t_permuted, A_hat)

        # Matmul with Theta
        out = torch.relu(torch.matmul(out, self.theta))
        
        # Permute back: (batch_size, seq_len-2, num_nodes, spatial_channels) -> (batch_size, spatial_channels, num_nodes, seq_len-2)
        out = out.permute(0, 3, 2, 1)

        # Temporal Conv 2
        t2 = self.temporal2(out) # (batch_size, out_channels, num_nodes, seq_len-4)

        # BatchNorm: input expects (batch_size, num_nodes, out_channels, seq_len-4)
        out = self.batch_norm(t2.permute(0, 2, 1, 3))
        
        # Permute back to standard format
        return out.permute(0, 2, 1, 3)

class STGCN(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network.
    Designed for MULTIVARIATE input.
    """

    def __init__(self, num_nodes, num_features, num_timesteps_input, num_timesteps_output):
        """
        :param num_nodes: Number of stations (e.g., 32)
        :param num_features: Number of input features per station (e.g., AQI, PM2.5, Temp, Wind... -> 10 features)
        :param num_timesteps_input: Length of historical sequence (e.g., 24 hours)
        :param num_timesteps_output: Length of prediction sequence (e.g., 1 hour or 12 hours)
        """
        super(STGCN, self).__init__()
        
        # We define the network architecture manually for flexibility
        self.block1 = STGCNBlock(in_channels=num_features, spatial_channels=16, out_channels=64, num_nodes=num_nodes)
        
        self.block2 = STGCNBlock(in_channels=64, spatial_channels=16, out_channels=64, num_nodes=num_nodes)
        
        # The output of block2 will have sequence length: num_timesteps_input - 4 - 4 = num_timesteps_input - 8
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        # Sequence length now: num_timesteps_input - 8 - 2 = num_timesteps_input - 10
        
        # Fully connected layer to map channels to 1 (prediction output) and sequences to required output length
        self.fully_connected_1 = nn.Linear((num_timesteps_input - 10) * 64, 256)
        self.fully_connected_2 = nn.Linear(256, num_timesteps_output)

    def forward(self, A_hat, X):
        """
        X: (batch_size, seq_len, num_nodes, num_features)  <-- Standard Data Loader format
        A_hat: (num_nodes, num_nodes) Normalized adjacency matrix
        """
        # Convert standard shape to PyTorch Conv2D shape: (batch_size, num_features, num_nodes, seq_len)
        X = X.permute(0, 3, 2, 1)

        out1 = self.block1(X, A_hat)
        out2 = self.block2(out1, A_hat)
        out3 = self.last_temporal(out2)
        
        # out3 shape: (batch_size, out_channels=64, num_nodes, seq_len_left)
        batch_size, channels, num_nodes, seq_len_left = out3.shape
        
        # Flatten the temporal and channel dimensions for the fully connected layer
        out4 = out3.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, channels * seq_len_left)
        
        # Dense layer
        out5 = F.relu(self.fully_connected_1(out4))
        out6 = self.fully_connected_2(out5)
        
        # Output shape: (batch_size, num_nodes, num_timesteps_output)
        return out6
