import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

SELECTED_STATIONS = [1, 3, 4, 7, 9, 12, 13, 15, 16, 17, 18, 24, 27, 29, 31, 32]
EXCLUDE_COLS = ['timestamp', 'split', 'station_id', 'province', 'district']

# Các cột thời tiết được mô hình S06 phép biết trước
FUTURE_WEATHER_COLS = [
    'temp', 'rh', 'precip', 'wind_spd', 'wind_sin', 'wind_cos'
]

class AirQualityDataset(Dataset):
    def __init__(self, data, seq_len=48, pred_len=24, weather_indices=None):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.weather_indices = weather_indices

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len] # (seq_len, num_nodes, num_features)
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len, :, 0] # (pred_len, num_nodes)
        
        if self.weather_indices is not None and len(self.weather_indices) > 0:
            x_future_weather = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len, :, self.weather_indices]
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(x_future_weather, dtype=torch.float32)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def load_data(normalized_dir='data/normalized'):
    station_data = {}
    timestamps = None
    features_list = None
    
    for sid in SELECTED_STATIONS:
        path = os.path.join(normalized_dir, f'norm_station_{sid}.csv')
        df = pd.read_csv(path, parse_dates=['timestamp']).sort_values('timestamp')
        
        if features_list is None:
            cols = list(df.columns)
            valid_cols = [c for c in cols if c not in EXCLUDE_COLS]
            if 'pm25' in valid_cols:
                valid_cols.remove('pm25')
            features_list = ['pm25'] + valid_cols
            timestamps = df['timestamp'].values
            
        features = df[features_list].values
        station_data[sid] = features
            
    num_samples = len(timestamps)
    num_nodes = len(SELECTED_STATIONS)
    num_features = len(features_list)
    
    X = np.zeros((num_samples, num_nodes, num_features), dtype=np.float32)
    for i, sid in enumerate(SELECTED_STATIONS):
        X[:, i, :] = station_data[sid]
        
    ts = pd.to_datetime(timestamps)
    train_mask = ts < '2025-01-01'
    val_mask = (ts >= '2025-01-01') & (ts < '2025-05-01')
    test_mask = ts >= '2025-05-01'
    
    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    test_indices = np.where(test_mask)[0]
    
    train_data = X[train_indices]
    val_data = X[val_indices]
    test_data = X[test_indices]
    
    weather_indices = []
    for w in FUTURE_WEATHER_COLS:
        if w in features_list:
            weather_indices.append(features_list.index(w))
            
    return train_data, val_data, test_data, weather_indices, num_features

def get_dataloaders(normalized_dir='data/normalized', seq_len=48, pred_len=24, batch_size=32):
    train_data, val_data, test_data, weather_indices, num_features = load_data(normalized_dir)
    print(f"[*] Features dimension: {num_features} (pm25 @ index 0)")
    
    train_set = AirQualityDataset(train_data, seq_len, pred_len, weather_indices)
    val_set = AirQualityDataset(val_data, seq_len, pred_len, weather_indices)
    test_set = AirQualityDataset(test_data, seq_len, pred_len, weather_indices)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, test_loader, num_features
