import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConv(nn.Module):
    """
    Tương đương lớp GraphConv trong E_STGCN_Code_Share.ipynb 
    - Tính representation: X * W
    - Aggregate neighbor: Aggregated = A * representation
    - Combine: (concat hoặc add) rồi qua ReLU
    """
    def __init__(self, in_feat, out_feat, combination_type="concat"):
        super(GraphConv, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.combination_type = combination_type
        
        # Theo code Keras gốc: self.weight có shape (in_feat, out_feat)
        self.weight = nn.Parameter(torch.FloatTensor(in_feat, out_feat))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, features, adj):
        # features: (..., num_nodes, in_feat)
        # adj: (num_nodes, num_nodes)
        
        # 1. Compute nodes representation (X * W)
        nodes_repr = torch.matmul(features, self.weight) # (..., num_nodes, out_feat)
        
        # 2. Aggregate messages (A * (X * W))
        aggregated_messages = torch.matmul(adj, nodes_repr)
        
        # 3. Update (Combine + ReLU)
        if self.combination_type == "concat":
            h = torch.cat([nodes_repr, aggregated_messages], dim=-1)
        elif self.combination_type == "add":
            h = nodes_repr + aggregated_messages
        else:
            raise ValueError("combination_type phải là 'concat' hoặc 'add'")
            
        return F.relu(h)

class LSTMGC(nn.Module):
    """
    Tương đương lớp LSTMGC trong bài báo gốc:
    GraphConv -> Reshape -> 1 LSTM Layer -> Lấy timestep cuối cùng -> Dense
    """
    def __init__(self, in_feat, out_feat, lstm_units, input_seq_len, output_seq_len, combination_type="concat"):
        super(LSTMGC, self).__init__()
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        
        self.graph_conv = GraphConv(in_feat, out_feat, combination_type=combination_type)
        
        # Nếu concat, đầu ra của GCN sẽ là out_feat * 2
        lstm_input_dim = out_feat * 2 if combination_type == "concat" else out_feat
        
        # Keras mặc định LSTM activation là tanh, ta dùng y xì của PyTorch
        self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=lstm_units, num_layers=1, batch_first=True)
        
        self.dense = nn.Linear(lstm_units, output_seq_len)
        
    def forward(self, inputs, adj):
        # inputs trong Keras của họ là (num_nodes, batch_size, seq_len, in_feat)
        # Đầu vào của hệ thống hiện tại S06 là (batch_size, seq_len, num_nodes, in_feat)
        b, t, n, f = inputs.shape
        
        # Gọi GCN
        gcn_out = self.graph_conv(inputs, adj) # (b, t, n, gcn_out_dim)
        
        # Chuyển đổi để feed vào LSTM: gom (batch * nodes) thành dòng để xử lý Temporal
        # GCN đang là (b, t, n, out_dim) -> (b, n, t, out_dim) -> (b*n, t, out_dim)
        gcn_out = gcn_out.permute(0, 2, 1, 3).contiguous().view(b * n, t, -1)
        
        # Gọi LSTM
        lstm_out, _ = self.lstm(gcn_out) # lstm_out: (b*n, t, lstm_units)
        
        # Đặc trưng của bài báo: Cắt cái vòi timestep, chỉ chừa đúng điểm CUỐI CÙNG
        lstm_last = lstm_out[:, -1, :] # (b*n, lstm_units)
        
        # Mở thẳng cổng bắn 1 tia dự báo bằng Dense layer
        dense_out = self.dense(lstm_last) # (b*n, pred_len)
        
        # Reshape về lại (batch_size, pred_len, num_nodes) của dataloader S06
        output = dense_out.view(b, n, -1).permute(0, 2, 1).contiguous()
        return output

class ESTGCN(nn.Module):
    """
    Wrapper ngoài cùng tương đương cách model.compile() trong notebook
    """
    def __init__(self, num_nodes, num_features, seq_len, pred_len):
        super(ESTGCN, self).__init__()
        
        # Các tham số mặc định từ file Keras gốc
        out_feat = 10 
        lstm_units = 64
        combination_type = "concat"
        
        self.core = LSTMGC(
            in_feat=num_features,
            out_feat=out_feat, 
            lstm_units=lstm_units,
            input_seq_len=seq_len,
            output_seq_len=pred_len,
            combination_type=combination_type
        )
        
    def forward(self, x, adj, x_future_weather=None):
        # Tác giả bài báo không sử dụng future weather đùn vào network
        out = self.core(x, adj)
        return out
