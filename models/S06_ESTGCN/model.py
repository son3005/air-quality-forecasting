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
        # x: (batch * seq_len, num_nodes, in_channels)
        # adj: (num_nodes, num_nodes)
        xW = torch.matmul(x, self.weight) # (batch * seq_len, num_nodes, out_channels)
        out = torch.matmul(adj, xW) + self.bias # Adjacency operation
        return F.relu(out)

class ESTGCN(nn.Module):
    def __init__(self, num_nodes, num_features, seq_len, pred_len):
        super(ESTGCN, self).__init__()
        
        # Spatial Graph Convolution
        self.gcn1 = GraphConv(num_features, 64)
        self.bn1 = nn.LayerNorm(64)
        self.gcn2 = GraphConv(64, 32)
        self.bn2 = nn.LayerNorm(32)
        
        # Temporal LSTM Sequence tracking
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True, num_layers=2)
        
        # Output projection to pred_len
        self.fc = nn.Sequential(
            nn.Linear(64 + num_features, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, pred_len)
        )
        
    def forward(self, x, adj):
        # x: (batch, seq_len, num_nodes, num_features)
        b, t, n, f = x.shape
        
        # Fold batch and time to process spatial graphs independently
        x_reshaped = x.view(b * t, n, f)
        
        # GCN Encoder for spatial features
        x_gcn = self.gcn1(x_reshaped, adj)
        x_gcn = self.bn1(x_gcn)
        x_gcn = self.gcn2(x_gcn, adj) # (b*t, n, 32)
        x_gcn = self.bn2(x_gcn)
        
        # Unfold to sequence and permute for LSTM per node
        x_seq = x_gcn.view(b, t, n, -1) # (b, t, n, 32)
        x_seq = x_seq.permute(0, 2, 1, 3).contiguous().view(b * n, t, -1) # (b*n, t, 32)
        
        # LSTM layer
        lstm_out, (hn, cn) = self.lstm(x_seq) # hn is (num_layers, b*n, 64)
        last_hidden = hn[-1] # (b*n, 64)
        
        # Skip connection: include the direct observation of the last time step
        x_last = x[:, -1, :, :].contiguous().view(b * n, f)
        
        hidden_cat = torch.cat([last_hidden, x_last], dim=-1)
        
        # Decode the sequence using MLP
        out = self.fc(hidden_cat) # (b*n, pred_len)
        
        # Reshape to standard output (batch, pred_len, num_nodes)
        out = out.view(b, n, -1).permute(0, 2, 1).contiguous()
        
        return out
