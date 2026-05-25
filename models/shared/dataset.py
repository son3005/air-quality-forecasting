"""
models/shared/dataset.py

Unified Data Loading and Dataset Class for Multi-Station Deep Learning models.
Shared across iTransformer, Mamba, TFT, and PatchTST.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

EXCLUDE_COLS = ['timestamp', 'split', 'station_id', 'province', 'district',
                'is_frozen', 'is_pm25_sensor_error', 'is_weekend_holiday']
POLLUTANTS = ['pm25', 'pm10', 'co', 'o3', 'no2', 'so2']

def load_station_data(sids, split_name='train', data_dir='data/split/block7'):
    """
    Load normalized data for selected stations filtered by split.
    """
    station_data = {}
    for sid in sids:
        fpath = os.path.join(data_dir, f'station_{sid}.csv')
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Normalized station data not found: {fpath}")
        df = pd.read_csv(fpath)
        df_s = df[df['split'] == split_name].reset_index(drop=True)

        feat_cols = [c for c in df_s.columns if c not in EXCLUDE_COLS and c not in POLLUTANTS
                     and df_s[c].dtype in ['float64', 'float32', 'int64', 'int32']]

        # Đảm bảo các cột pollutant đều tồn tại, nếu thiếu điền 0
        target_df = pd.DataFrame()
        for pol in POLLUTANTS:
            if pol in df_s.columns:
                target_df[pol] = df_s[pol]
            else:
                target_df[pol] = 0.0

        station_data[sid] = {
            'features': df_s[feat_cols].fillna(0).values.astype(np.float32),
            'targets': target_df.fillna(0).values.astype(np.float32),
        }
    return station_data


class MultiStationDataset(Dataset):
    """
    Multi-station sequence dataset.
    
    x: (seq_len, num_variates) where num_variates = num_features + num_stations * 6
    y: (num_stations * 6,) — Targets at t + seq_len - 1 + horizon
    """
    def __init__(self, station_data, sids, seq_len=48, horizon=1):
        self.seq_len = seq_len
        self.horizon = horizon
        self.sids = sids
        num_nodes = len(sids)

        # Align all stations to same length
        min_len = min(len(v['features']) for v in station_data.values())

        # Stack: features for each station + targets for each station
        features_list = []
        target_list = []
        for sid in sids:
            features_list.append(station_data[sid]['features'][:min_len])
            target_list.append(station_data[sid]['targets'][:min_len])

        self.target_matrix = np.concatenate(target_list, axis=1)  # (T, N*6)
        self.shared_features = features_list[0]  # (T, F)
        self.num_features = self.shared_features.shape[1]

        # Variates = shared_features + each station's targets
        # Total variates = F + N*6
        self.data = np.concatenate([self.shared_features, self.target_matrix], axis=1)
        self.num_variates = self.data.shape[1]
        self.total_len = min_len
        self.valid_len = self.total_len - seq_len - horizon + 1

    def __len__(self):
        return max(0, self.valid_len)

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]  # (seq_len, num_variates)
        t_idx = idx + self.seq_len - 1 + self.horizon
        y = self.target_matrix[t_idx]  # (N*6,)
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32))
