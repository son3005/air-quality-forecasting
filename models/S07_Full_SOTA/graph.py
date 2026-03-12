import os
import numpy as np
import pandas as pd

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0 # Earth radius in kilometers
    
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi / 2.0) ** 2 + \
        np.cos(phi1) * np.cos(phi2) * \
        np.sin(delta_lambda / 2.0) ** 2
        
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

def calculate_bearing(lat1, lon1, lat2, lon2):
    # Calculate bearing angle between two coordinates
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_lambda = np.radians(lon2 - lon1)
    y = np.sin(delta_lambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda)
    theta = np.arctan2(y, x)
    return theta # [-pi, pi]

def get_wind_directed_adjacency(info_path='../../data/info.csv', norm_dir='../../data/normalized', selected_stations=None):
    if selected_stations is None:
        selected_stations = [1, 3, 4, 7, 9, 12, 13, 15, 16, 17, 18, 24, 27, 29, 31, 32]
        
    df_info = pd.read_csv(info_path)
    df_selected = df_info[df_info['station'].isin(selected_stations)].copy()
    
    station_coords = {}
    for _, row in df_selected.iterrows():
        station_coords[row['station']] = (row['latitude'], row['longitude'])
        
    num_nodes = len(selected_stations)
    
    # Calculate average wind vector for each station
    wind_vectors = {}
    for sid in selected_stations:
        df = pd.read_csv(os.path.join(norm_dir, f'norm_station_{sid}.csv'))
        df_train = df[df['split'] == 'train']
        avg_sin = df_train['wind_sin'].mean()
        avg_cos = df_train['wind_cos'].mean()
        wind_vectors[sid] = np.arctan2(avg_sin, avg_cos)
        
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    distances = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                lat1, lon1 = station_coords[selected_stations[i]]
                lat2, lon2 = station_coords[selected_stations[j]]
                distances[i, j] = haversine_distance(lat1, lon1, lat2, lon2)
                
    dist_std = distances[distances > 0].std()
    sigma_squared = dist_std ** 2
    
    # Construct Directed Adjacency Matrix
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                adj_matrix[i, j] = 1.0 # identity connection
            else:
                lat_i, lon_i = station_coords[selected_stations[i]]
                lat_j, lon_j = station_coords[selected_stations[j]]
                
                # Direction from station i to station j
                bearing_ij = calculate_bearing(lat_i, lon_i, lat_j, lon_j)
                
                # Predominant wind at station i
                wind_angle_i = wind_vectors[selected_stations[i]]
                
                # Alignment difference in radians [0, pi]
                angle_diff = np.abs(bearing_ij - wind_angle_i)
                angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
                
                # Normalize alignment to [0, 1] where 1 is pure tailwind and 0 is pure headwind
                alignment = (np.cos(angle_diff) + 1.0) / 2.0
                
                # Geographic distance factor (Gaussian kernel)
                dist_factor = np.exp(- (distances[i, j] ** 2) / sigma_squared)
                
                # Weighted mixture
                adj_matrix[i, j] = dist_factor * (0.1 + 0.9 * alignment)
                
    return adj_matrix

if __name__ == '__main__':
    adj = get_wind_directed_adjacency()
    print("Adjacency Matrix Shape:", adj.shape)
    print("Min W:", adj.min(), "Max W:", adj.max())
