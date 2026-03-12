import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        # x: (batch, num_nodes, in_channels)
        # adj: (num_nodes, num_nodes)
        xW = torch.matmul(x, self.weight) # (batch, num_nodes, out_channels)
        out = torch.matmul(adj, xW) + self.bias # Adjacency operation
        return F.relu(out)

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, spatial_channels, out_channels, num_nodes):
        super(STGCNBlock, self).__init__()
        # temporal convolution over time sequence
        self.tconv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        
        # spatial graph convolution
        self.gcn = GraphConv(out_channels, spatial_channels)
        
        # temporal convolution again
        self.tconv2 = nn.Conv2d(spatial_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        
        # 1x1 convolution for residual connection if dimensions differ
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.residual_conv = None
            
        # batch norm over nodes
        self.batch_norm = nn.BatchNorm2d(num_nodes)

    def forward(self, x, adj):
        # x: (batch, in_channels, num_nodes, seq_len)
        res = x
        
        x = F.relu(self.tconv1(x)) # (batch, out_channels, num_nodes, seq_len)
        
        # reshape for graph convolution
        # we iterate over seq_len dimension by folding it with batch dimension
        b, c, n, t = x.shape
        x_gcn = x.permute(0, 3, 2, 1).contiguous().view(b * t, n, c) 
        
        x_gcn = self.gcn(x_gcn, adj) # (batch * seq_len, num_nodes, spatial_channels)
        
        x_gcn = x_gcn.view(b, t, n, -1).permute(0, 3, 2, 1) # (batch, spatial_channels, num_nodes, seq_len)
        
        x = F.relu(self.tconv2(x_gcn)) # (batch, out_channels, num_nodes, seq_len)
        
        x = x.permute(0, 2, 1, 3).contiguous() # (batch, num_nodes, out_channels, seq_len)
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1, 3).contiguous() # (batch, out_channels, num_nodes, seq_len)
        
        if self.residual_conv is not None:
            res = self.residual_conv(res)
            
        x = x + res
        return F.relu(x)

class STGCN(nn.Module):
    def __init__(self, num_nodes, num_features, seq_len, pred_len):
        super(STGCN, self).__init__()
        
        self.stgcn_blocks = nn.ModuleList([
            STGCNBlock(in_channels=num_features, spatial_channels=64, out_channels=32, num_nodes=num_nodes),
            STGCNBlock(in_channels=32, spatial_channels=64, out_channels=32, num_nodes=num_nodes)
        ])
        
        # fully convolutional output to get (pred_len) from the sequence
        self.output_conv = nn.Conv2d(32, pred_len, kernel_size=(1, seq_len))
        
    def forward(self, x, adj):
        # x: (batch, seq_len, num_nodes, num_features)
        x = x.permute(0, 3, 2, 1) # (batch, num_features, num_nodes, seq_len)
        
        for block in self.stgcn_blocks:
            x = block(x, adj)
            
        # x is (batch, out_channels=32, num_nodes, seq_len)
        out = self.output_conv(x) # (batch, pred_len, num_nodes, 1)
        out = out.squeeze(-1) # (batch, pred_len, num_nodes)
        
        # To match regular targets, permute to (batch, pred_len, num_nodes)
        # Actually it's already there
        return out
