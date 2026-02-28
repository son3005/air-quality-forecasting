import os
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

DATA_DIR = "data/KnowAir-V2"
OUT_DIR = "data/clean"

def haversine_dist(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on the earth (specified in decimal degrees)"""
    # convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # haversine formula 
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r

def process_region(region_name):
    print(f"\nProcessing {region_name.upper()} region...")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. Load Dataset
    nc_path = os.path.join(DATA_DIR, f"dataset_{region_name}.nc")
    csv_path = os.path.join(DATA_DIR, f"stations_{region_name}.csv")
    
    ds = xr.open_dataset(nc_path)
    stations_df = pd.read_csv(csv_path)
    
    num_stations = len(stations_df)
    
    # Target feature first, then meteorology
    feature_vars = ["PM2.5", "O3", "t2m", "d2m", "sp", "tp", "blh", "msdwswrf", "u100", "v100"]
    num_features = len(feature_vars)
    
    time_len = ds.dims['time']
    print(f"Data shape will be: (time={time_len}, stations={num_stations}, features={num_features})")
    
    # Create final numpy array
    # Memory efficient allocation
    out_array = np.zeros((time_len, num_stations, num_features), dtype=np.float32)
    
    print("Extracting features from NetCDF...")
    for f_idx, var_name in enumerate(tqdm(feature_vars)):
        # Data in NC is (time, station)
        val = ds[var_name].values
        # Assign to our out_array
        out_array[:, :, f_idx] = val
        
    out_npy = os.path.join(OUT_DIR, f"knowair_{region_name}.npy")
    print(f"Saving data array to {out_npy}...")
    np.save(out_npy, out_array)
    
    # 2. Build Adjacency Matrix
    print("Building adjacency matrix...")
    lats = stations_df['lat'].values
    lons = stations_df['lon'].values
    
    dist_mat = np.zeros((num_stations, num_stations))
    for i in range(num_stations):
        for j in range(i+1, num_stations):
            d = haversine_dist(lats[i], lons[i], lats[j], lons[j])
            dist_mat[i, j] = d
            dist_mat[j, i] = d
            
    # Gaussian kernel thresholding as in STGCN
    sigma2 = 2500.0
    epsilon = 0.1
    
    W = np.zeros((num_stations, num_stations))
    for i in range(num_stations):
        for j in range(num_stations):
            if i != j:
                w = np.exp(-(dist_mat[i, j] ** 2) / sigma2)
                if w >= epsilon:
                    W[i, j] = w
                    
    # Normalized Laplacian
    A = W + np.eye(num_stations)
    D = np.diag(np.power(A.sum(1), -0.5))
    D[np.isinf(D)] = 0.
    adj = D @ A @ D
    
    out_adj = os.path.join(OUT_DIR, f"adj_mat_knowair_{region_name}.npy")
    print(f"Saving adjacency matrix to {out_adj}...")
    np.save(out_adj, adj)
    
    print(f"Done processing {region_name}!")

if __name__ == "__main__":
    process_region("bthsa")  # Beijing-Tianjin-Hebei
    # process_region("yrd")  # Yangtze River Delta (optional, not enabled now to save RAM)
