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
        xW = torch.matmul(x, self.weight) # (batch, num_nodes, out_channels)
        out = torch.matmul(adj, xW) + self.bias
        return F.relu(out)

class PureGCN(nn.Module):
    def __init__(self, num_nodes, num_features):
        super(PureGCN, self).__init__()
        self.gcn1 = GraphConv(num_features, 64)
        self.gcn2 = GraphConv(64, 32)
        
        # Max-Pooling requirement
        # Pool across the feature dimension to aggregate
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(16, 1)

    def forward(self, x, adj):
        # x: (batch, num_nodes, num_features)
        out = self.gcn1(x, adj)
        out = self.gcn2(out, adj)
        
        b, n, c = out.shape
        out = out.reshape(b * n, c).unsqueeze(1) # (b*n, 1, c)
        
        out = self.pool(out) # (b*n, 1, c/2)
        out = out.squeeze(1) # (b*n, c/2)
        
        out = self.fc(out) # (b*n, 1)
        out = out.view(b, n) # (batch, num_nodes)
        
        return out
