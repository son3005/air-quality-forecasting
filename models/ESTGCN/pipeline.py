"""
models/ESTGCN/pipeline.py

E-STGCN Training + Evaluation Pipeline cho 12 trạm đã chọn.
Sử dụng shared graph_builder.py và shared metrics.py.
"""
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ══════════════════════════════════════════════════════════════════════════
# CONFIG — Chỉnh sửa tại đây
# ══════════════════════════════════════════════════════════════════════════
SELECTED_STATIONS = [1, 4, 5, 16, 17, 27, 7, 18, 24, 30, 31, 32]
REGIONS = {
    'north': [1, 4, 5, 16, 17, 27],
    'south': [7, 18, 24, 30, 31, 32],
}
PM25_COL = 'pm25'

# --- Block Split Selection ---
# Chọn 1 trong 3: 'block5', 'block7', 'block30'
BLOCK      = 'block7'
DATA_DIR   = f'data/split/{BLOCK}'
INFO_PATH  = 'data/info.csv'

EXCLUDE_COLS = ['timestamp', 'split', 'station_id', 'province', 'district',
                'is_frozen', 'is_pm25_sensor_error', 'is_weekend_holiday']

SEQ_LEN    = 48
PRED_LEN   = 24
HORIZONS   = [1, 3, 6, 12, 24]
BATCH_SIZE = 256
EPOCHS     = 100
LR         = 1e-3
PATIENCE   = 15
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ══════════════════════════════════════════════════════════════════════════
# IMPORTS — Shared + local model
# ══════════════════════════════════════════════════════════════════════════
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from metrics import inverse_pm25, get_metrics
from graph_builder import get_base_matrices

from model import ESTGCN

print(f"[*] Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════
# ADJACENCY MATRIX
# ══════════════════════════════════════════════════════════════════════════

def get_adjacency_matrix(sids):
    """Build normalized adjacency matrix from distance for specific stations."""
    dist_km, _ = get_base_matrices(INFO_PATH, sids)
    N = dist_km.shape[0]
    # Gaussian kernel
    sigma2 = dist_km[dist_km > 0].std() ** 2 if (dist_km > 0).any() else 1.0
    A = np.exp(-(dist_km ** 2) / (sigma2 + 1e-8))
    np.fill_diagonal(A, 1.0)
    # Row-normalize
    row_sum = A.sum(axis=1, keepdims=True)
    A = A / (row_sum + 1e-8)
    return A


# ══════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════

class STGCNDataset(Dataset):
    """
    Multi-station spatio-temporal dataset.
    Returns: (x, y) where
        x: (seq_len, num_nodes, num_features)
        y: (pred_len, num_nodes)  — PM2.5 only
    """
    def __init__(self, station_data, seq_len=48, pred_len=24):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_nodes = len(station_data)

        # Align all stations to same length
        min_len = min(len(v['features']) for v in station_data.values())
        self.features = np.stack([v['features'][:min_len] for v in station_data.values()], axis=1)
        self.targets = np.stack([v['pm25'][:min_len] for v in station_data.values()], axis=1)
        # features: (T, N, C), targets: (T, N)

        self.total_len = min_len
        self.valid_len = self.total_len - seq_len - pred_len + 1

    def __len__(self):
        return max(0, self.valid_len)

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_len]     # (seq_len, N, C)
        y = self.targets[idx + self.seq_len:idx + self.seq_len + self.pred_len]  # (pred_len, N)
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32))


def load_station_data(sids, split_name='train'):
    """Load normalized data for selected stations filtered by split."""
    station_data = {}
    for sid in sids:
        df = pd.read_csv(os.path.join(DATA_DIR, f'station_{sid}.csv'))

        # Sử dụng cột split đã chia sẵn theo block
        mask = df['split'] == split_name

        df_split = df[mask].reset_index(drop=True)

        # Extract features (drop non-numeric and excluded columns)
        feat_cols = [c for c in df_split.columns if c not in EXCLUDE_COLS and c != PM25_COL]
        feat_cols = [c for c in feat_cols if df_split[c].dtype in ['float64', 'float32', 'int64', 'int32']]

        station_data[sid] = {
            'features': df_split[feat_cols].fillna(0).values.astype(np.float32),
            'pm25': df_split[PM25_COL].fillna(0).values.astype(np.float32),
        }

    return station_data


def get_dataloaders(sids, seq_len=SEQ_LEN, pred_len=PRED_LEN, batch_size=BATCH_SIZE):
    """Create train/val/test DataLoaders for a set of stations."""
    train_data = load_station_data(sids, 'train')
    val_data = load_station_data(sids, 'val')
    test_data = load_station_data(sids, 'test')

    train_ds = STGCNDataset(train_data, seq_len, pred_len)
    val_ds = STGCNDataset(val_data, seq_len, pred_len)
    test_ds = STGCNDataset(test_data, seq_len, pred_len)

    num_features = train_ds.features.shape[2]

    print(f"    Dataset sizes: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    print(f"    Features: {num_features}, Nodes: {len(sids)}")

    train_loader = DataLoader(train_ds, batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader, num_features


# ══════════════════════════════════════════════════════════════════════════
# TRAIN + EVALUATE
# ══════════════════════════════════════════════════════════════════════════

def train_model(seq_len=SEQ_LEN, pred_len=PRED_LEN, epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE):
    print("=" * 70)
    print(f"  ESTGCN Pipeline — 12 Stations | Block: {BLOCK}")
    print(f"  Horizons: {HORIZONS}")
    print("=" * 70)

    all_results = []
    total_start = time.time()

    for r_name, sids in REGIONS.items():
        print(f"\n[{r_name.upper()}] Training {len(sids)} stations...")
        
        train_loader, val_loader, test_loader, num_features = get_dataloaders(sids, seq_len, pred_len, batch_size)

        adj = get_adjacency_matrix(sids)
        adj = torch.tensor(adj, dtype=torch.float32).to(DEVICE)

        num_nodes = len(sids)
        model = ESTGCN(num_nodes=num_nodes, num_features=num_features, seq_len=seq_len, pred_len=pred_len)
        model.to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"    Model params: {n_params:,}")

        criterion = nn.HuberLoss(delta=1.0)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

        save_dir = os.path.join('models_saved', BLOCK, 'ESTGCN')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'baseline_{r_name}.pth')

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            t0 = time.time()
            model.train()
            train_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                out = model(x, adj)
                loss = criterion(out, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                train_loss += loss.item() * x.size(0)
            train_loss /= max(len(train_loader.dataset), 1)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    out = model(x, adj)
                    loss = criterion(out, y)
                    val_loss += loss.item() * x.size(0)
            val_loss /= max(len(val_loader.dataset), 1)

            dt = time.time() - t0
            print(f"    Epoch {epoch+1:03d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | {dt:.1f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    Early stop at epoch {epoch+1}")
                    break

        # ══════════════════════════════════════════════════════════════════
        # EVALUATION PER REGION
        # ══════════════════════════════════════════════════════════════════
        model.load_state_dict(torch.load(save_path, weights_only=True))
        model.eval()

        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(DEVICE)
                out = model(x, adj)
                all_preds.append(out.cpu().numpy())
                all_targets.append(y.numpy())

        if all_preds:
            all_preds = np.concatenate(all_preds, axis=0)     # (samples, pred_len, num_nodes)
            all_targets = np.concatenate(all_targets, axis=0)
        else:
            print(f"[!] No test predictions for {r_name}!")
            continue

        # Inverse transform per station locally
        all_targets_inv = np.zeros_like(all_targets)
        all_preds_inv = np.zeros_like(all_preds)
        for node_idx, sid in enumerate(sids):
            for t_step in range(all_targets.shape[1]):
                all_targets_inv[:, t_step, node_idx] = inverse_pm25(all_targets[:, t_step, node_idx], sid)
                all_preds_inv[:, t_step, node_idx] = inverse_pm25(all_preds[:, t_step, node_idx], sid)

        for h in HORIZONS:
            a_arr = all_targets_inv[:, h-1, :].flatten()
            p_arr = all_preds_inv[:, h-1, :].flatten()
            rmse, mae, r2, mape = get_metrics(a_arr, p_arr)
            print(f"    T+{h:<2d} | RMSE={rmse:.2f} | MAE={mae:.2f} | R2={r2*100:.2f}% | MAPE={mape:.2f}%")
            all_results.append({
                'region': r_name, 'horizon': f'T+{h}',
                'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape,
                'n_test': len(a_arr), 'train_time': round(dt, 2)
            })

    # Summary
    print("\n" + "=" * 70)
    print("  FINAL BENCHMARK — ESTGCN")
    print("=" * 70)
    print(f"{'Region':<10} {'Horizon':<8} {'RMSE':>8} {'MAE':>8} {'R2 %':>8} {'MAPE %':>10}")
    print("-" * 55)
    for r in all_results:
        print(f"{r['region']:<10} {r['horizon']:<8} {r['RMSE']:>8.2f} {r['MAE']:>8.2f} {r['R2']*100:>7.2f}% {r['MAPE']:>9.2f}%")

    print("\n" + "-" * 55)
    print("AGGREGATED (weighted by test size):")
    for h in HORIZONS:
        hr = [r for r in all_results if r['horizon'] == f'T+{h}']
        if hr:
            total = sum(r['n_test'] for r in hr)
            agg = lambda key: sum(r[key]*r['n_test'] for r in hr) / total
            print(f"  T+{h:<3d}  RMSE={agg('RMSE'):.2f}  MAE={agg('MAE'):.2f}  "
                  f"R2={agg('R2')*100:.2f}%  MAPE={agg('MAPE'):.2f}%")

    total_time = time.time() - total_start
    print(f"\n  Total training time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print("=" * 70)

    return all_results, total_time


if __name__ == '__main__':
    train_model()
