import numpy as np
import pandas as pd
import os

def haversine(lat1, lon1, lat2, lon2):
    """
    Tính khoảng cách đường chim bay giữa 2 điểm (kinh độ, vĩ độ) bằng công thức Haversine.
    Kết quả trả về theo đơn vị: Kilometer.
    """
    R = 6371.0 # Bán kính Trái Đất (km)
    
    # Chuyển đổi sang radian
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    distance = R * c
    return distance

def generate_graph_matrix(info_path, out_path, sigma2=10.0, epsilon=0.5):
    """
    Tạo Ma trận Kề (Adjacency Matrix) dựa trên khoảng cách địa lý.
    Sử dụng hàm Weight Matrix chuẩn của STGCN:
        W_{ij} = exp(- (dist_{ij}^2) / sigma2) nếu W_{ij} >= epsilon, ngược lại 0
    sigma2: kiểm soát độ rộng của gaussian (km^2). 
            Các trạm xa nhau hơn độ lan này sẽ có trọng số nhỏ.
    epsilon: Ngưỡng cắt (threshold) để tạo tính thưa (sparsity) cho đồ thị.
    """
    print(f"Reading station info from: {info_path}")
    df = pd.read_csv(info_path)
    
    # Đảm bảo sort theo đúng thứ tự station 1 -> 32
    df = df.sort_values(by='station').reset_index(drop=True)
    num_nodes = len(df)
    
    lats = df['latitude'].values
    lons = df['longitude'].values
    
    # 1. Tính ma trận khoảng cách (Distance Matrix)
    dist_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                dist_matrix[i, j] = haversine(lats[i], lons[i], lats[j], lons[j])
                
    # 2. Xây dựng ma trận trọng số (Weighted Adjacency Matrix) theo thuật toán STGCN
    W = np.zeros((num_nodes, num_nodes))
    
    # Tính bình phương khoảng cách
    dist2 = dist_matrix ** 2
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                weight = np.exp(-dist2[i, j] / sigma2)
                if weight >= epsilon:
                    W[i, j] = weight
                    
    # Lưu ra file CSV
    np.savetxt(out_path, W, delimiter=',')
    
    print(f"\nGraph Matrix generated and saved to: {out_path}")
    print(f"Shape: {W.shape}")
    
    # Phân tích một chút về đồ thị vừa tạo
    num_edges = np.count_nonzero(W)
    sparsity = 1.0 - (num_edges / (num_nodes * (num_nodes - 1)))
    print(f"Number of directed edges: {num_edges}")
    print(f"Matrix Sparsity: {sparsity:.2%}")
    
    return W

if __name__ == "__main__":
    # Get the directory where generate_graph.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Calculate absolute path to data dir
    # models/renew STGCN -> models -> CODE -> data
    data_dir = os.path.join(current_dir, "..", "..", "data")
    
    INFO_FILE = os.path.join(data_dir, "info.csv")
    OUT_FILE = os.path.join(data_dir, "graph_adj_matrix.csv")
    
    # Chú ý: Ở Việt Nam, các tỉnh cách nhau khá xa. 
    # Nếu để sigma2 nhỏ (như STGCN gốc 0.1 cho mạng lưới nhỏ trong 1 thành phố) 
    # thì đồ thị sẽ bị đứt gãy hoàn toàn (chỉ có các trạm cùng thành phố mới nối với nhau).
    # Chúng ta thử set sigma2 = 2500 (tương đương std dev 50km)
    # và epsilon = 0.1 để giữ lại nhiều kết nối hơn.
    
    W = generate_graph_matrix(
        info_path=INFO_FILE, 
        out_path=OUT_FILE,
        sigma2=2500.0, # (50km)^2
        epsilon=0.1
    )
