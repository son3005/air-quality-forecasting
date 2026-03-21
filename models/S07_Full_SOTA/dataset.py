import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

SELECTED_STATIONS = [1, 3, 4, 7, 9, 12, 13, 15, 16, 17, 18, 24, 27, 29, 31, 32]

def load_data(normalized_dir='../../data/normalized'):
    station_data = {}
    splits = None
    
    for sid in SELECTED_STATIONS:
        path = os.path.join(normalized_dir, f'norm_station_{sid}.csv')
        df = pd.read_csv(path)
        
        # Calculate cyclical time features
        timestamps = pd.to_datetime(df['timestamp'])
        df['hour_sin'] = np.sin(2 * np.pi * timestamps.dt.hour / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * timestamps.dt.hour / 24.0)
        df['dayofweek_sin'] = np.sin(2 * np.pi * timestamps.dt.dayofweek / 7.0)
        df['dayofweek_cos'] = np.cos(2 * np.pi * timestamps.dt.dayofweek / 7.0)
        
        # Exclude metadata columns
        features_to_drop = ['timestamp', 'split', 'station_id', 'province', 'district']
        feature_cols = [c for c in df.columns if c not in features_to_drop]
        
        features = df[feature_cols].values
        station_data[sid] = features
        
        if splits is None:
            splits = df['split'].values
            
    num_samples = len(splits)
    num_nodes = len(SELECTED_STATIONS)
    num_features = station_data[SELECTED_STATIONS[0]].shape[1]
    
    # Construct X matrix: (num_samples, num_nodes, num_features)
    X = np.zeros((num_samples, num_nodes, num_features), dtype=np.float32)
    for i, sid in enumerate(SELECTED_STATIONS):
        X[:, i, :] = station_data[sid]
        
    train_mask = (splits == 'train')
    val_mask = (splits == 'val')
    test_mask = (splits == 'test')
    
    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    test_indices = np.where(test_mask)[0]
    
    train_data = X[train_indices]
    val_data = X[val_indices]
    test_data = X[test_indices]
    
    return train_data, val_data, test_data, num_features

class AirQualityDataset(Dataset):
    def __init__(self, data, seq_len=24, pred_len=24):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len] # (seq_len, num_nodes, num_features)
        # Target PM2.5 is at index 0 of features
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len, :, 0] # (pred_len, num_nodes)
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def get_dataloaders(normalized_dir='../../data/normalized', seq_len=24, pred_len=24, batch_size=16):
    train_data, val_data, test_data, num_features = load_data(normalized_dir)
    
    train_set = AirQualityDataset(train_data, seq_len, pred_len)
    val_set = AirQualityDataset(val_data, seq_len, pred_len)
    test_set = AirQualityDataset(test_data, seq_len, pred_len)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, test_loader, num_features
