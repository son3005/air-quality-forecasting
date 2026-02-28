"""
Hybrid STGCN + EVT-GPD — All-in-One Training Script.
Kiến trúc STGCN (Temporal Conv + Spectral GCN) + EVT-GPD Loss.

Pipeline:
  1. Load dữ liệu 32 trạm
  2. Xây dựng đồ thị (Haversine)
  3. Fit GPD parameters (thay R bằng Python)
  4. Train STGCN with EVT-GPD Loss (warmup + scheduler + early stop)
  5. Evaluate: RMSE, MAE, R²-Score
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import os
import glob
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from torch.utils.data import Dataset, DataLoader
from scipy.stats import genpareto

from model import STGCN, EVTGPDLoss


# ========================================================================
# 1. DATA
# ========================================================================
class AQIDataset(Dataset):
    def __init__(self, data_normalized, seq_len, pre_len, target_col_idx):
        self.data = data_normalized
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.target_col_idx = target_col_idx
        self.num_samples = len(self.data) - self.seq_len - self.pre_len + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        X = self.data[idx:idx + self.seq_len, :, :]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pre_len, :, self.target_col_idx]
        y = np.transpose(y, (1, 0))  # (nodes, pre_len)
        return torch.FloatTensor(X), torch.FloatTensor(y)


def load_data(data_dir, batch_size, seq_len, pre_len, target_col_idx):
    files = sorted(
        glob.glob(os.path.join(data_dir, "station_*.csv")),
        key=lambda x: int(os.path.basename(x).replace('station_', '').replace('.csv', ''))
    )
    data_raw = np.stack([pd.read_csv(f).drop(columns=['time']).values for f in files], axis=1)

    train_size = int(len(data_raw) * 0.7)
    val_size = int(len(data_raw) * 0.1)

    train_data = data_raw[:train_size]
    mean = np.mean(train_data, axis=(0, 1))
    std = np.std(train_data, axis=(0, 1))
    normed = (data_raw - mean) / (std + 1e-5)

    sets = [normed[:train_size], normed[train_size:train_size + val_size], normed[train_size + val_size:]]
    loaders = [
        DataLoader(AQIDataset(s, seq_len, pre_len, target_col_idx),
                   batch_size=batch_size, shuffle=(i == 0))
        for i, s in enumerate(sets)
    ]
    return loaders[0], loaders[1], loaders[2], mean, std


# ========================================================================
# 2. GRAPH
# ========================================================================
def build_graph(info_path, sigma2=2500.0, epsilon=0.1):
    df = pd.read_csv(info_path).sort_values('station').reset_index(drop=True)
    lats, lons = np.radians(df['latitude'].values), np.radians(df['longitude'].values)
    n = len(df)

    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dlat, dlon = lats[j] - lats[i], lons[j] - lons[i]
            a = np.sin(dlat / 2) ** 2 + np.cos(lats[i]) * np.cos(lats[j]) * np.sin(dlon / 2) ** 2
            d = 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            dist[i, j] = dist[j, i] = d

    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                w = np.exp(-(dist[i, j] ** 2) / sigma2)
                if w >= epsilon:
                    W[i, j] = w

    A_tilde = W + np.eye(n)
    d_inv = np.diag(np.power(A_tilde.sum(1), -0.5))
    d_inv[np.isinf(d_inv)] = 0.
    return d_inv @ A_tilde @ d_inv


# ========================================================================
# 3. POT FITTING (thay thế R)
# ========================================================================
def fit_gpd(data_dir, threshold, target_col, num_stations):
    xi_list, sig_list = [], []
    for i in range(1, num_stations + 1):
        vals = pd.read_csv(os.path.join(data_dir, f"station_{i}.csv"))[target_col].dropna().values
        train_vals = vals[:int(len(vals) * 0.7)]
        exc = train_vals[train_vals > threshold] - threshold

        if len(exc) < 10:
            xi_list.append(0.1); sig_list.append(10.0)
            print(f"  Station {i}: insufficient exceedances ({len(exc)}), using defaults")
        else:
            try:
                c, _, scale = genpareto.fit(exc, floc=0)
                xi_list.append(c); sig_list.append(scale)
                print(f"  Station {i}: xi={c:.4f}, sigma={scale:.4f} ({len(exc)} exc.)")
            except:
                xi_list.append(0.1); sig_list.append(10.0)

    return np.array(xi_list), np.array(sig_list)


# ========================================================================
# 4. EVALUATE
# ========================================================================
def evaluate(model, loader, A_hat, criterion, device, return_preds=False):
    model.eval()
    total_loss, preds, targs = 0.0, [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(A_hat, X)
            if out.shape[-1] == 1:
                out = out.squeeze(-1)
            loss = criterion(out, y)
            total_loss += loss.item() * X.size(0)
            if return_preds:
                preds.append(out.cpu().numpy())
                targs.append(y.cpu().numpy())
    avg = total_loss / len(loader.dataset)
    if return_preds:
        return avg, np.concatenate(preds), np.concatenate(targs)
    return avg


# ========================================================================
# 5. TRAIN
# ========================================================================
def train(args):
    # Fix random seed cho kết quả ổn định
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
    clean_dir = os.path.join(root, "clean")
    info_path = os.path.join(root, "info.csv")

    # col_map: maps feature index → name (matches OUTPUT_SELECTED_COLS in preprocessing_clean.py)
    col_map = {
        0: 'aqi', 1: 'pm25', 2: 'pm10', 3: 'co', 4: 'no2', 5: 'o3',
        6: 'temp', 7: 'rh', 8: 'dewpt', 9: 'precip', 10: 'wind_spd',
        11: 'wind_sin', 12: 'wind_cos',
        13: 'ah', 14: 'dpd', 15: 'is_stagnant',
        16: 'rush_hour',
        17: 'hour_sin', 18: 'hour_cos', 19: 'month_sin', 20: 'month_cos',
        21: 'ma_pm25_24',
        22: 'delta_pm25', 23: 'pm25_lag_1', 24: 'pm25_lag_24', 25: 'rain_sum_6',
    }
    target_name = col_map.get(args.target_col_idx, 'aqi')

    # ---- Data ----
    print("\n[1/4] Loading data...")
    train_ld, val_ld, test_ld, mean, std = load_data(
        clean_dir, args.batch_size, args.seq_len, args.pre_len, args.target_col_idx
    )
    num_features = mean.shape[0]
    print(f"  Features: {num_features}, Target: {target_name}")
    print(f"  Train/Val/Test: {len(train_ld.dataset)}/{len(val_ld.dataset)}/{len(test_ld.dataset)}")

    # ---- Graph ----
    print("\n[2/4] Building graph...")
    A_hat_np = build_graph(info_path)
    num_nodes = A_hat_np.shape[0]
    A_hat = torch.FloatTensor(A_hat_np).to(device)
    print(f"  Nodes: {num_nodes}")

    # ---- POT ----
    print(f"\n[3/4] Fitting GPD (threshold={args.threshold})...")
    xi_vals, sig_vals = fit_gpd(clean_dir, args.threshold, target_name, num_nodes)

    target_mean = float(mean[args.target_col_idx])
    target_std = float(std[args.target_col_idx])

    evt_loss = EVTGPDLoss(
        xi=xi_vals, sigma=sig_vals,
        mean_vals=target_mean, std_vals=target_std,
        threshold=args.threshold,
        beta_1=args.beta1, beta_2=args.beta2
    ).to(device)

    mse_loss = nn.MSELoss()

    # ---- Model ----
    print(f"\n[4/4] Building Hybrid STGCN + EVT-GPD...")
    model = STGCN(
        num_nodes=num_nodes,
        num_features=num_features,
        num_timesteps_input=args.seq_len,
        num_timesteps_output=args.pre_len
    ).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {params:,}")
    print(f"  GPD Warmup: {args.warmup} epochs | β₁={args.beta1}, β₂={args.beta2}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5, min_lr=1e-6)

    best_val, patience_cnt = float('inf'), 0
    os.makedirs("checkpoints", exist_ok=True)

    print(f"\n{'='*70}")
    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val MSE':>10} | {'LR':>10} | {'Time':>6} | Status")
    print(f"{'='*70}")

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()

        # Warmup control
        use_evt = epoch >= args.warmup
        evt_loss.set_warmup(use_evt)
        label = "EVT" if use_evt else "MSE"

        epoch_loss = 0.0
        for X, y in train_ld:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(A_hat, X)
            if out.shape[-1] == 1:
                out = out.squeeze(-1)
            loss = evt_loss(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item() * X.size(0)

        epoch_loss /= len(train_ld.dataset)
        val_loss = evaluate(model, val_ld, A_hat, mse_loss, device)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        if val_loss < best_val:
            best_val = val_loss
            patience_cnt = 0
            torch.save(model.state_dict(), 'checkpoints/best_hybrid.pth')
            status = f"★ Best ({best_val:.4f})"
        else:
            patience_cnt += 1
            status = f"wait {patience_cnt}/{args.patience}"

        print(f"{epoch+1:>6} | {label:>4} {epoch_loss:>7.4f} | {val_loss:>10.4f} | {lr:>10.6f} | {elapsed:>5.1f}s | {status}")

        if patience_cnt >= args.patience:
            print(f"\n  Early stopping at epoch {epoch + 1}!")
            break

    # ---- Test ----
    print(f"\n{'='*70}")
    print("TESTING — Hybrid STGCN + EVT-GPD")
    print(f"{'='*70}")

    model.load_state_dict(torch.load('checkpoints/best_hybrid.pth', weights_only=True))
    test_loss, p_norm, t_norm = evaluate(model, test_ld, A_hat, mse_loss, device, return_preds=True)

    p_real = p_norm * target_std + target_mean
    t_real = t_norm * target_std + target_mean
    pf, tf = p_real.flatten(), t_real.flatten()

    rmse = np.sqrt(mean_squared_error(tf, pf))
    mae = mean_absolute_error(tf, pf)
    r2 = r2_score(tf, pf)

    print(f"  Test MSE (norm)  : {test_loss:.4f}")
    print(f"  Test RMSE ({target_name:>10}): {rmse:.2f}")
    print(f"  Test MAE  ({target_name:>10}): {mae:.2f}")
    print(f"  Test R²-Score    : {r2:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hybrid STGCN + EVT-GPD')
    parser.add_argument('--seq_len', type=int, default=72)
    parser.add_argument('--pre_len', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--target_col_idx', type=int, default=21, help='Target column index: 0=aqi, 1=pm25, 21=ma_pm25_24')
    parser.add_argument('--threshold', type=float, default=50.0, help='EVT threshold')
    parser.add_argument('--warmup', type=int, default=15, help='MSE-only warmup epochs')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--beta1', type=float, default=0.99, help='MSE weight')
    parser.add_argument('--beta2', type=float, default=0.01, help='GPD weight')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')

    args = parser.parse_args()
    train(args)
