import os
import numpy as np
import pandas as pd

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0 # Earth radius in kilometers
    
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi / 2.0) ** 2 + \
        np.cos(phi1) * np.cos(phi2) * \
        np.sin(delta_lambda / 2.0) ** 2
        
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def calculate_bearing(lat1, lon1, lat2, lon2):
    # Tính góc phương vị từ điểm 1 đến điểm 2
    # Trả về giá trị trong khoảng [0, 2pi) radians
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    lambda1, lambda2 = np.radians(lon1), np.radians(lon2)
    
    y = np.sin(lambda2 - lambda1) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(lambda2 - lambda1)
    
    theta = np.arctan2(y, x)
    return (theta + 2 * np.pi) % (2 * np.pi)

def get_base_matrices(info_path='data/info.csv', selected_stations=None):
    """
    Tính toán 2 Ma trận tĩnh nền tảng: Cự ly và Góc phương vị.
    Phục vụ cho việc sinh Ma Trận Đồ Thị Động bên trong PyTorch (model.forward).
    """
    if selected_stations is None:
        selected_stations = [1, 3, 4, 7, 9, 12, 13, 15, 16, 17, 18, 24, 27, 29, 31, 32]
        
    df_info = pd.read_csv(info_path)
    df_selected = df_info[df_info['station'].isin(selected_stations)].copy()
    
    # Sort to enforce the exact order of selected_stations
    df_selected['station'] = pd.Categorical(df_selected['station'], categories=selected_stations, ordered=True)
    df_selected = df_selected.sort_values('station').reset_index(drop=True)
    
    num_nodes = len(selected_stations)
    
    distances = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    bearings = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    
    lats = df_selected['latitude'].values
    lons = df_selected['longitude'].values
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distances[i, j] = haversine_distance(lats[i], lons[i], lats[j], lons[j])
                bearings[i, j] = calculate_bearing(lats[i], lons[i], lats[j], lons[j])
            else:
                distances[i, j] = 0.0
                bearings[i, j] = 0.0
                
    return distances, bearings

if __name__ == '__main__':
    dist, bear = get_base_matrices('../../data/info.csv')
    print("Distance Matrix Shape:", dist.shape)
    print("Bearing Matrix Shape:", bear.shape)
    print("Max Distance (km):", dist.max())
