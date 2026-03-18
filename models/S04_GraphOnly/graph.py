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

def get_adjacency_matrix(info_path='data/info.csv', selected_stations=None):
    if selected_stations is None:
        selected_stations = [1, 3, 4, 7, 9, 12, 13, 15, 16, 17, 18, 24, 27, 29, 31, 32]
        
    df_info = pd.read_csv(info_path)
    
    df_selected = df_info[df_info['station'].isin(selected_stations)].copy()
    
    station_coords = {}
    for _, row in df_selected.iterrows():
        station_coords[row['station']] = (row['latitude'], row['longitude'])
        
    num_nodes = len(selected_stations)
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
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                adj_matrix[i, j] = 1.0
            else:
                adj_matrix[i, j] = np.exp(- (distances[i, j] ** 2) / sigma_squared)
                
    return adj_matrix
