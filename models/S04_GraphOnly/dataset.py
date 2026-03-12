import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

SELECTED_STATIONS = [1, 3, 4, 7, 9, 12, 13, 15, 16, 17, 18, 24, 27, 29, 31, 32]
EXCLUDE_COLS = ['timestamp', 'province', 'district', 'split', 'station_id']

class GraphDataset(Dataset):
    def __init__(self, data, seq_len=1, pred_len=1):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        # We need input t and output t+1
        # data: (num_samples, num_nodes, num_features)
        x = self.data[idx : idx + self.seq_len] # (1, num_nodes, num_features)
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len, :, 0] # (1, num_nodes)
        
        # Squeeze the seq_len and pred_len dimensions since they are 1
        x = x[0] # (num_nodes, num_features)
        y = y[0] # (num_nodes,)
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def load_data(normalized_dir='../../data/normalized'):
    station_data = {}
    splits = None
    features_list = None
    
    for sid in SELECTED_STATIONS:
        path = os.path.join(normalized_dir, f'norm_station_{sid}.csv')
        df = pd.read_csv(path)
        
        if features_list is None:
            # Reorder to make pm25 the first feature
            cols = list(df.columns)
            valid_cols = [c for c in cols if c not in EXCLUDE_COLS]
            if 'pm25' in valid_cols:
                valid_cols.remove('pm25')
            features_list = ['pm25'] + valid_cols
            
        features = df[features_list].values
        station_data[sid] = features
        
        if splits is None:
            splits = df['split'].values
            
    num_samples = len(splits)
    num_nodes = len(SELECTED_STATIONS)
    num_features = len(features_list)
    
    X = np.zeros((num_samples, num_nodes, num_features), dtype=np.float32)
    for i, sid in enumerate(SELECTED_STATIONS):
        X[:, i, :] = station_data[sid]
        
    train_mask = (splits == 'train')
    val_mask = (splits == 'val')
    test_mask = (splits == 'test')
    
    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    test_indices = np.where(test_mask)[0]
    
    return X[train_indices], X[val_indices], X[test_indices], num_features

def get_dataloaders(normalized_dir='../../data/normalized', batch_size=32):
    train_data, val_data, test_data, num_features = load_data(normalized_dir)
    
    train_set = GraphDataset(train_data, seq_len=1, pred_len=1)
    val_set = GraphDataset(val_data, seq_len=1, pred_len=1)
    test_set = GraphDataset(test_data, seq_len=1, pred_len=1)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, test_loader, num_features
