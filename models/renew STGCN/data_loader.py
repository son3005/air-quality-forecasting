import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob


class MultivariateAQIDataset(Dataset):
    def __init__(self, data_dir, seq_len=24, pre_len=1, target_col_idx=0, split='train'):
        """
        Args:
            data_dir (str): Directory containing the 32 station CSV files.
            seq_len (int): Length of historical sequence (e.g., 24 hours).
            pre_len (int): Prediction horizon (e.g., 1 hour).
            target_col_idx (int): Index of the target column to predict (e.g., 0 for 'ma_pm25_24').
            split (str): 'train', 'val', or 'test'.
        """
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.target_col_idx = target_col_idx
        
        # 1. Load data from all 32 stations
        # We need data in shape: (num_timestamps, num_nodes, num_features)
        
        file_paths = sorted(glob.glob(os.path.join(data_dir, "station_*.csv")),
                              key=lambda x: int(os.path.basename(x).replace('station_', '').replace('.csv', '')))
        
        station_data_list = []
        for file in file_paths:
            df = pd.read_csv(file)
            
            # The columns based on preprocessing_clean.py:
            # ['time', 'aqi', 'pm25', 'pm10', 'co', 'ma_pm25_24', 'rh', 'dewpt', 'temp', 'precip', 'wind_spd', 'wind_sin', 'wind_cos', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
            
            # Drop time column as it's not a numerical feature for the model 
            # (we already have cyclical encoded time features)
            df = df.drop(columns=['time'])
            
            # Convert to numpy array: shape (num_timestamps, num_features)
            station_data_list.append(df.values)
        
        # Combine into (num_timestamps, num_nodes, num_features)
        # All stations have exactly the same length (full hourly index)
        self.data_raw = np.stack(station_data_list, axis=1)
        
        # 2. Train / Val / Test Split
        total_len = len(self.data_raw)
        train_size = int(total_len * 0.7)
        val_size = int(total_len * 0.1)
        # test_size is the rest
        
        if split == 'train':
            self.data = self.data_raw[:train_size]
        elif split == 'val':
            self.data = self.data_raw[train_size:train_size + val_size]
        elif split == 'test':
            self.data = self.data_raw[train_size + val_size:]
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")
            
        # 3. Z-score Normalization (computed ONLY on training set to avoid data leakage)
        train_data = self.data_raw[:train_size]
        # Calculate mean and std over the timestamp and node dimensions -> result shape: (num_features,)
        self.mean = np.mean(train_data, axis=(0, 1))
        self.std = np.std(train_data, axis=(0, 1))
        
        # Normalize the current split data
        self.data_normalized = (self.data - self.mean) / (self.std + 1e-5)
        
        # Number of samples we can generate
        self.num_samples = len(self.data_normalized) - self.seq_len - self.pre_len + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        """
        Returns:
            X: (seq_len, num_nodes, num_features)
            y: (num_nodes,) -> the target feature at time t + pre_len
        """
        start_idx = index
        end_idx = index + self.seq_len
        
        # Historical features: (seq_len, num_nodes, num_features)
        X = self.data_normalized[start_idx:end_idx, :, :]
        
        # Future target sequence for multi-step forecasting: (pre_len, num_nodes)
        y = self.data_normalized[end_idx : end_idx + self.pre_len, :, self.target_col_idx]
        
        # PyTorch expects shape: (batch_size, num_nodes, pre_len) -> we swap axes below
        # Current y shape is (pre_len, num_nodes) -> transpose -> (num_nodes, pre_len)
        y = np.transpose(y, (1, 0))
        
        return torch.FloatTensor(X), torch.FloatTensor(y)
        
    def inverse_transform(self, normalized_target):
        """Convert predictions back to original scale."""
        return normalized_target * self.std[self.target_col_idx] + self.mean[self.target_col_idx]

def get_dataloaders(data_dir, batch_size=32, seq_len=24, pre_len=1, target_col_idx=4):
    """
    Args:
        target_col_idx: 4 is 'ma_pm25_24' if we drop 'time' from ['time', 'aqi', 'pm25', 'pm10', 'co', 'ma_pm25_24'...]
    """
    train_dataset = MultivariateAQIDataset(data_dir, seq_len, pre_len, target_col_idx, split='train')
    val_dataset = MultivariateAQIDataset(data_dir, seq_len, pre_len, target_col_idx, split='val')
    test_dataset = MultivariateAQIDataset(data_dir, seq_len, pre_len, target_col_idx, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_dataset
