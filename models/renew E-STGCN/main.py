"""
E-STGCN Main Training Script (PyTorch) - Phiên bản 2 (Tinh chỉnh).
Pipeline: Load Data → POT Fitting → Build Graph → Train (với Warmup + Scheduler) → Evaluate.

Cải thiện:
  - Learning Rate Scheduler (ReduceLROnPlateau)
  - Early Stopping (patience=10)
  - GPD Loss Warmup (bật GPD penalty sau 10 epochs thuần MSE)
  - Tăng capacity model (gcn=32, lstm=128)
  - Gradient Clipping
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

from estgcn_model import ESTGCN, EVTGPDLoss
from pot_fitting import fit_gpd_per_station


# ========================================================================
# 1. DATASET CLASS
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

    def __getitem__(self, index):
        start = index
        end = index + self.seq_len
        X = self.data[start:end, :, :]
        y = self.data[end:end + self.pre_len, :, self.target_col_idx]
        y = np.transpose(y, (1, 0))
        return torch.FloatTensor(X), torch.FloatTensor(y)


def load_all_stations(data_dir):
    file_paths = sorted(
        glob.glob(os.path.join(data_dir, "station_*.csv")),
        key=lambda x: int(os.path.basename(x).replace('station_', '').replace('.csv', ''))
    )
    station_data = []
    for f in file_paths:
        df = pd.read_csv(f).drop(columns=['time'])
        station_data.append(df.values)
    return np.stack(station_data, axis=1)


def get_dataloaders(data_dir, batch_size, seq_len, pre_len, target_col_idx):
    data_raw = load_all_stations(data_dir)
    total_len = len(data_raw)
    train_size = int(total_len * 0.7)
    val_size = int(total_len * 0.1)

    train_data = data_raw[:train_size]
    mean = np.mean(train_data, axis=(0, 1))
    std = np.std(train_data, axis=(0, 1))
    data_normalized = (data_raw - mean) / (std + 1e-5)

    train_norm = data_normalized[:train_size]
    val_norm = data_normalized[train_size:train_size + val_size]
    test_norm = data_normalized[train_size + val_size:]

    train_ds = AQIDataset(train_norm, seq_len, pre_len, target_col_idx)
    val_ds = AQIDataset(val_norm, seq_len, pre_len, target_col_idx)
    test_ds = AQIDataset(test_norm, seq_len, pre_len, target_col_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, mean, std


# ========================================================================
# 2. GRAPH CONSTRUCTION
# ========================================================================
def build_adjacency_matrix(info_path, sigma2=2500.0, epsilon=0.1):
    df = pd.read_csv(info_path).sort_values('station').reset_index(drop=True)
    lats, lons = df['latitude'].values, df['longitude'].values
    n = len(df)

    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                lat1, lon1 = np.radians(lats[i]), np.radians(lons[i])
                lat2, lon2 = np.radians(lats[j]), np.radians(lons[j])
                dlat, dlon = lat2 - lat1, lon2 - lon1
                a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
                dist[i, j] = 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                w = np.exp(-(dist[i, j] ** 2) / sigma2)
                if w >= epsilon:
                    W[i, j] = w

    A_tilde = W + np.eye(n)
    rowsum = np.sum(A_tilde, axis=1)
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D_inv_sqrt = np.diag(d_inv_sqrt)
    A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt

    return A_hat


# ========================================================================
# 3. EVALUATE FUNCTION
# ========================================================================
def evaluate(model, data_loader, A_hat, criterion, device, return_preds=False):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(A_hat, inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            if return_preds:
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)

    if return_preds:
        return avg_loss, np.concatenate(all_preds), np.concatenate(all_targets)
    return avg_loss


# ========================================================================
# 4. MAIN TRAINING FUNCTION
# ========================================================================
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(current_dir, "..", "..", "data")
    data_clean = os.path.join(data_root, "clean")
    info_path = os.path.join(data_root, "info.csv")

    # ---- Step 1: Load Data ----
    print("\n[1/4] Loading data...")
    train_loader, val_loader, test_loader, mean, std = get_dataloaders(
        data_dir=data_clean,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        pre_len=args.pre_len,
        target_col_idx=args.target_col_idx
    )
    num_features = mean.shape[0]
    print(f"  Num features: {num_features}")
    print(f"  Train/Val/Test samples: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")

    # ---- Step 2: Build Graph ----
    print("\n[2/4] Building adjacency matrix...")
    A_hat_np = build_adjacency_matrix(info_path, sigma2=2500.0, epsilon=0.1)
    num_nodes = A_hat_np.shape[0]
    A_hat = torch.FloatTensor(A_hat_np).to(device)
    print(f"  Num nodes: {num_nodes}")

    # ---- Step 3: POT Fitting ----
    print("\n[3/4] Fitting GPD parameters (POT)...")
    target_col_map = {0: 'aqi', 1: 'pm25', 2: 'pm10', 3: 'co', 4: 'ma_pm25_24'}
    target_col_name = target_col_map.get(args.target_col_idx, 'aqi')

    pot_results = fit_gpd_per_station(
        data_dir=data_clean,
        threshold=args.threshold,
        target_col=target_col_name,
        num_stations=num_nodes
    )

    target_mean = float(mean[args.target_col_idx])
    target_std = float(std[args.target_col_idx])

    evt_loss = EVTGPDLoss(
        xi=pot_results[:, 0],
        sigma=pot_results[:, 1],
        mean_vals=target_mean,
        std_vals=target_std,
        target_col_idx=args.target_col_idx,
        threshold=args.threshold,
        beta_1=0.99,
        beta_2=0.01
    ).to(device)

    mse_loss = nn.MSELoss()

    # ---- Step 4: Build Model & Train ----
    print(f"\n[4/4] Building E-STGCN model (v2 - tuned)...")
    model = ESTGCN(
        num_nodes=num_nodes,
        num_features=num_features,
        input_seq_len=args.seq_len,
        output_seq_len=args.pre_len,
        gcn_out_feat=args.gcn_out_feat,
        lstm_units=args.lstm_units,
        combination_type="concat",
        dropout=args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable parameters: {total_params:,}")
    print(f"  Dropout: {args.dropout}")
    print(f"  GCN out feat: {args.gcn_out_feat}, LSTM units: {args.lstm_units}")
    print(f"  GPD Warmup: {args.warmup_epochs} epochs (pure MSE first)")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Learning Rate Scheduler: giảm LR khi Val Loss không cải thiện
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )

    best_val_loss = float('inf')
    patience_counter = 0
    os.makedirs("checkpoints", exist_ok=True)

    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 80)

    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0

        # Warmup: chỉ bật GPD sau warmup_epochs đầu tiên
        if epoch >= args.warmup_epochs:
            evt_loss.set_warmup(True)
            loss_label = "EVT"
        else:
            evt_loss.set_warmup(False)
            loss_label = "MSE"

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(A_hat, inputs)

            loss = evt_loss(outputs, targets)
            loss.backward()

            # Gradient clipping để ổn định training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        val_loss = evaluate(model, val_loader, A_hat, mse_loss, device)

        # Scheduler step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        elapsed = time.time() - start_time
        print(f'Epoch [{epoch + 1}/{args.epochs}], '
              f'Train ({loss_label}): {train_loss:.4f}, '
              f'Val (MSE): {val_loss:.4f}, '
              f'LR: {current_lr:.6f}, '
              f'Time: {elapsed:.1f}s', end='')

        # Early Stopping & Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'checkpoints/best_estgcn.pth')
            print(f'  -> Best! (Val: {best_val_loss:.4f})')
        else:
            patience_counter += 1
            print(f'  (no improve {patience_counter}/{args.patience})')
            if patience_counter >= args.patience:
                print(f'\nEarly stopping at epoch {epoch + 1}!')
                break

    # ---- Testing ----
    print("\n" + "=" * 60)
    print("TESTING")
    print("=" * 60)
    model.load_state_dict(torch.load('checkpoints/best_estgcn.pth', weights_only=True))
    test_loss, preds_norm, targets_norm = evaluate(
        model, test_loader, A_hat, mse_loss, device, return_preds=True
    )

    print(f'Test Loss (MSE normalized): {test_loss:.4f}')

    preds_actual = preds_norm * target_std + target_mean
    targets_actual = targets_norm * target_std + target_mean
    preds_flat = preds_actual.flatten()
    targets_flat = targets_actual.flatten()

    rmse = np.sqrt(mean_squared_error(targets_flat, preds_flat))
    mae = mean_absolute_error(targets_flat, preds_flat)
    r2 = r2_score(targets_flat, preds_flat)

    print(f'Test RMSE (Original Scale {target_col_name}): {rmse:.2f}')
    print(f'Test MAE  (Original Scale {target_col_name}): {mae:.2f}')
    print(f'Test R2-Score                              : {r2:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='E-STGCN Training (Tuned v2)')
    parser.add_argument('--seq_len', type=int, default=24, help='Input sequence length (hours)')
    parser.add_argument('--pre_len', type=int, default=96, help='Prediction horizon (hours)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--target_col_idx', type=int, default=0, help='Target column index (0=aqi, 1=pm25...)')
    parser.add_argument('--threshold', type=float, default=60.0, help='EVT threshold for GPD penalty')
    parser.add_argument('--gcn_out_feat', type=int, default=32, help='GCN output features')
    parser.add_argument('--lstm_units', type=int, default=128, help='LSTM hidden units')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of MSE-only warmup epochs')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')

    args = parser.parse_args()
    train(args)
