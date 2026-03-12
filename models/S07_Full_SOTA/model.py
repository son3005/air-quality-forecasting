import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, num_nodes, num_timesteps):
        super(SpatialAttention, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_timesteps))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_timesteps))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_nodes, num_nodes))
        self.Vs = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W2)
        nn.init.uniform_(self.W1)
        nn.init.uniform_(self.W3)
        nn.init.zeros_(self.bs)
        nn.init.xavier_uniform_(self.Vs)

    def forward(self, x):
        # x: (B, N, F, T)
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (B, N, T)
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)       # (B, T, N)
        S = torch.matmul(lhs, rhs)                             # (B, N, N)
        S = S + self.bs
        S = torch.sigmoid(S)                                    
        S = torch.matmul(self.Vs, S)                            
        S = F.softmax(S, dim=1)                                
        return S

class TemporalAttention(nn.Module):
    def __init__(self, in_channels, num_nodes, num_timesteps):
        super(TemporalAttention, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_nodes))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_nodes))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels))
        self.be = nn.Parameter(torch.FloatTensor(1, num_timesteps, num_timesteps))
        self.Ve = nn.Parameter(torch.FloatTensor(num_timesteps, num_timesteps))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.U2)
        nn.init.uniform_(self.U1)
        nn.init.uniform_(self.U3)
        nn.init.zeros_(self.be)
        nn.init.xavier_uniform_(self.Ve)

    def forward(self, x):
        # x: (B, N, F, T) -> x.transpose(1, 3) is (B, T, F, N)
        lhs = torch.matmul(torch.matmul(x.transpose(1, 3), self.U1), self.U2) # (B, T, N)
        rhs = torch.matmul(self.U3, x) # (B, N, T)
        E = torch.matmul(lhs, rhs) # (B, T, T)
        E = E + self.be
        E = torch.sigmoid(E)
        E = torch.matmul(self.Ve, E)
        E = F.softmax(E, dim=1)
        return E

class ASTGCNBlock(nn.Module):
    def __init__(self, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, num_nodes, num_timesteps):
        super(ASTGCNBlock, self).__init__()
        self.TAt = TemporalAttention(in_channels, num_nodes, num_timesteps)
        self.SAt = SpatialAttention(in_channels, num_nodes, num_timesteps)
        self.cheb_conv = nn.Linear(K * in_channels, nb_chev_filter)
        self.K = K
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)

    def forward(self, x, cheb_polynomials):
        # x: (B, N, C, T)
        B, N, C, T = x.shape
        E = self.TAt(x) # (B, T, T)
        x_tat = torch.matmul(x.reshape(B, N*C, T), E).reshape(B, N, C, T)
        S = self.SAt(x_tat) # (B, N, N)

        spatial_gcn = []
        for k in range(self.K):
            T_k = cheb_polynomials[k] # (N, N)
            T_k_with_S = T_k.unsqueeze(0) * S # (B, N, N)
            x_step = x_tat.permute(0, 3, 2, 1) # (B, T, C, N)
            T_kw_S_aug = T_k_with_S.unsqueeze(1).transpose(-1, -2) # (B, 1, N, N)
            x_k = torch.matmul(x_step, T_kw_S_aug) # (B, T, C, N)
            x_k = x_k.permute(0, 3, 2, 1) # (B, N, C, T)
            spatial_gcn.append(x_k)
            
        x_gcn = torch.cat(spatial_gcn, dim=2) # (B, N, K*F, T)
        x_gcn = x_gcn.permute(0, 3, 1, 2) # (B, T, N, K*F)
        x_gcn = self.cheb_conv(x_gcn) # (B, T, N, nb_chev_filter)
        x_gcn = F.relu(x_gcn)
        x_gcn = x_gcn.permute(0, 3, 2, 1) # (B, nb_chev_filter, N, T)
        
        x_tcn = self.time_conv(x_gcn) # (B, nb_time_filter, N, T)
        x_res = self.residual_conv(x.permute(0, 2, 1, 3)) # (B, nb_time_filter, N, T)
        x_out = x_tcn + x_res
        
        x_out = x_out.permute(0, 3, 2, 1) # (B, T, N, nb_time_filter)
        x_out = self.ln(x_out)
        x_out = x_out.permute(0, 2, 3, 1) # (B, N, nb_time_filter, T)
        return x_out

class ASTGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, out_channels=1, nb_block=2, K=3, nb_chev_filter=64, nb_time_filter=64, seq_len=24, pred_len=24):
        super(ASTGCN, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(nb_block):
            in_c = in_channels if i == 0 else nb_time_filter
            self.blocks.append(ASTGCNBlock(in_c, K, nb_chev_filter, nb_time_filter, 1, num_nodes, seq_len))
            
        self.fc1 = nn.Linear(nb_time_filter * seq_len, 128)
        self.fc2 = nn.Linear(128, pred_len * out_channels)
        self.out_channels = out_channels
        self.pred_len = pred_len

    def forward(self, x, cheb_polynomials):
        # x: (B, seq_len, N, C) -> (B, N, C, seq_len)
        x = x.permute(0, 2, 3, 1)
        
        for block in self.blocks:
            x = block(x, cheb_polynomials)
            
        B, N, C, T = x.shape
        x = x.reshape(B, N, C * T)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # (B, N, pred_len)
        
        x = x.reshape(B, N, self.pred_len, self.out_channels) # (B, N, pred_len, 1)
        x = x.permute(0, 2, 1, 3).squeeze(-1) # (B, pred_len, N)
        return x

def get_polynomials(adj, k_hop=3):
    out_degree = adj.sum(dim=1, keepdim=True).clamp(min=1e-5)
    P_f = adj / out_degree
    supports = [torch.eye(adj.shape[0]).to(adj.device), P_f]
    for _ in range(2, k_hop):
        P_k = torch.matmul(P_f, supports[-1])
        supports.append(P_k)
    return supports
