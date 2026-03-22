import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    """Tính khoảng cách dọc theo bề mặt Trái Đất giữa 2 tọa độ (km)"""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Tính góc phương vị (radian) từ điểm 1 đến điểm 2"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - (sin(lat1) * cos(lat2) * cos(dlon))
    initial_bearing = atan2(x, y)
    return (initial_bearing + 2*np.pi) % (2*np.pi)

def get_base_matrices(info_csv_path, selected_station_ids):
    """
    Tính toán 2 ma trận nền tĩnh: Khoảng cách (H_base) và Góc phương vị (B)
    """
    df = pd.read_csv(info_csv_path)
    df = df[df['station'].isin(selected_station_ids)].copy()
    
    # Sort IDs numerically ensures matrix coordinates match feature coordinates uniformly
    df['station_int'] = df['station'].apply(lambda x: int(str(x).replace('S','')) if isinstance(x, str) and str(x).startswith('S') else int(x))
    df = df.sort_values('station_int').reset_index(drop=True)
    N = len(df)
    
    lats = df['latitude'].values
    lons = df['longitude'].values
    
    dist_km = np.zeros((N, N))
    bearings = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            if i != j:
                dist_km[i, j] = haversine(lats[i], lons[i], lats[j], lons[j])
                bearings[i, j] = calculate_bearing(lats[i], lons[i], lats[j], lons[j])
                
    return dist_km, bearings

class DynamicGraphBuilder(nn.Module):
    """
    Module PyTorch DÙNG CHUNG cho tất cả các mô hình STGCN.
    Sinh ra Dynamic Adjacency Matrix dựa trên cơ sở vật lý (khoảng cách + hướng gió)
    với tham số alpha tối ưu tự động (Optimal Alpha) được phân tích từ Machine Learning.
    """
    def __init__(self, dist_km, bearings, alpha=0.4858, device='cpu'):
        super().__init__()
        self.alpha = alpha
        self.N = dist_km.shape[0]
        
        # 1. Tính toán ma trận trọng số không gian Gaussian (H)
        dist_tensor = torch.tensor(dist_km, dtype=torch.float32)
        mask = dist_tensor > 0
        sigma2 = dist_tensor[mask].std() ** 2 if mask.any() else 1.0
        
        H = torch.exp(-(dist_tensor**2) / (sigma2 + 1e-8))
        H.fill_diagonal_(1.0)
        
        # Đưa ma trận không đổi vào device
        self.register_buffer('H', H.to(device))
        self.register_buffer('B', torch.tensor(bearings, dtype=torch.float32).to(device))

    def forward(self, wind_dir_batch):
        """
        Biến đổi Batch dữ liệu gió thành các Adjacency Matrices không gian định hướng.
        
        Args:
            wind_dir_batch: Tensor kích thước (Batch, N) mang dữ liệu Hướng Gió (radian)
            
        Returns:
            adj: Tensor kích thước (Batch, N, N) biểu diễn Graph động ở mỗi time step
        """
        B_size = wind_dir_batch.size(0)
        
        # Mở rộng 차 chiều (Broadcasting)
        # B: (N, N) -> (Batch, N, N)
        bearings_exp = self.B.unsqueeze(0).expand(B_size, self.N, self.N)
        
        # wind_dir: (Batch, N) -> (Batch, N, 1) -> (Batch, N, N)
        wind_dir_exp = wind_dir_batch.unsqueeze(2).expand(B_size, self.N, self.N)
        
        # Yếu tố Gió (W): Tính sự tương đồng lượng giác giữa hướng liên kết và luồng gió
        angle_diff = bearings_exp - wind_dir_exp
        W = (torch.cos(angle_diff) + 1.0) / 2.0
        
        # Pha trộn (Mixture): Áp dụng tham số Alpha tối ưu (0.4858)
        adj = self.H * ((1.0 - self.alpha) + self.alpha * W)
        
        return adj
