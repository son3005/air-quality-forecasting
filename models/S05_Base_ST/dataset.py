import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

SELECTED_STATIONS = [1, 3, 4, 7, 9, 12, 13, 15, 16, 17, 18, 24, 27, 29, 31, 32]
RAW_FEATURES = [
    'pm25', 'pm10', 'co', 'o3', 'no2', 'so2', 'temp', 'rh', 'dewpt', 
    'precip', 'clouds', 'wind_spd', 'wind_gusts', 'soil_temp_0_7', 'soil_moist_0_7'
]

class AirQualityDataset(Dataset):
    def __init__(self, data, seq_len=24, pred_len=1):
        """
        data: (num_samples, num_nodes, num_features)
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len] # (seq_len, num_nodes, num_features)
        # Target PM2.5 is at index 0 of features
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len, :, 0] # (pred_len, num_nodes)
        
        # We might want to shape target as (pred_len, num_nodes) or similar
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def load_data(normalized_dir='../../data/normalized'):
    # Load all stations into a list
    station_data = {}
    splits = None
    
    for sid in SELECTED_STATIONS:
        path = os.path.join(normalized_dir, f'norm_station_{sid}.csv')
        df = pd.read_csv(path)
        
        # Extract the features
        features = df[RAW_FEATURES].values
        station_data[sid] = features
        
        if splits is None:
            splits = df['split'].values
            
    num_samples = len(splits)
    num_nodes = len(SELECTED_STATIONS)
    num_features = len(RAW_FEATURES)
    
    # Construct X matrix: (num_samples, num_nodes, num_features)
    X = np.zeros((num_samples, num_nodes, num_features), dtype=np.float32)
    for i, sid in enumerate(SELECTED_STATIONS):
        X[:, i, :] = station_data[sid]
        
    train_mask = (splits == 'train')
    val_mask = (splits == 'val')
    test_mask = (splits == 'test')
    
    # Use standard array indexing since train, val, test are contiguous
    # but let's actually just find the boundaries to prevent overlapping leaks
    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    test_indices = np.where(test_mask)[0]
    
    train_data = X[train_indices]
    val_data = X[val_indices]
    test_data = X[test_indices]
    
    return train_data, val_data, test_data

def get_dataloaders(normalized_dir='../../data/normalized', seq_len=24, pred_len=1, batch_size=32):
    train_data, val_data, test_data = load_data(normalized_dir)
    
    train_set = AirQualityDataset(train_data, seq_len, pred_len)
    val_set = AirQualityDataset(val_data, seq_len, pred_len)
    test_set = AirQualityDataset(test_data, seq_len, pred_len)
    
    # Note: drop_last=True for better batch processing with BatchNorms
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_dataloaders()
    for x, y in train_loader:
        print("X shape:", x.shape)
        print("Y shape:", y.shape)
        break
