"""
Hybrid STGCN + XLinear + EVT-GPD — Training Script.
Pipeline: Load Data → Graph → POT Fitting → Train → Evaluate.
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

from model import STGCN_XLinear, EVTGPDLoss


# ========================================================================
# DATA
# ========================================================================
class AQIDataset(Dataset):
    def __init__(self, data, seq_len, pre_len, target_idx):
        self.data, self.seq_len, self.pre_len, self.target_idx = data, seq_len, pre_len, target_idx
        self.n = len(data) - seq_len - pre_len + 1

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        X = self.data[i:i + self.seq_len, :, :]
        y = self.data[i + self.seq_len:i + self.seq_len + self.pre_len, :, self.target_idx]
        return torch.FloatTensor(X), torch.FloatTensor(y.T)


def load_data(data_npy_path, batch_size, seq_len, pre_len, target_idx, limit_years=None):
    print(f"Loading data from {data_npy_path}")
    # raw is [time, num_nodes, num_features]
    raw = np.load(data_npy_path)
    
    if limit_years is not None:
        # 1 year = 365 * 24 = 8760 hours
        max_len = limit_years * 8760
        if raw.shape[0] > max_len:
            raw = raw[-max_len:] # Take last N years to save memory
            print(f"Limited data to last {limit_years} years: shape {raw.shape}")

    ts = int(len(raw) * 0.7)
    vs = int(len(raw) * 0.1)

    mean = np.mean(raw[:ts], axis=(0, 1))
    std = np.std(raw[:ts], axis=(0, 1))
    normed = (raw - mean) / (std + 1e-5)

    loaders = [
        DataLoader(AQIDataset(normed[s:e], seq_len, pre_len, target_idx),
                   batch_size=batch_size, shuffle=(i == 0))
        for i, (s, e) in enumerate([(0, ts), (ts, ts + vs), (ts + vs, len(raw))])
    ]
    return loaders[0], loaders[1], loaders[2], mean, std


# ========================================================================
# GRAPH
# ========================================================================
def build_graph(adj_npy_path):
    print(f"Loading adjacency matrix from {adj_npy_path}")
    return np.load(adj_npy_path)


# ========================================================================
# POT FITTING
# ========================================================================
def fit_gpd(data_dir, threshold, target_col, num_stations):
    xi_list, sig_list = [], []
    for i in range(1, num_stations + 1):
        path = os.path.join(data_dir, f"station_{i}.csv")
        try:
            df = pd.read_csv(path)
            if target_col not in df.columns:
                print(f"  Station {i}: column '{target_col}' missing, using defaults")
                xi_list.append(0.1); sig_list.append(10.0)
                continue
            vals = df[target_col].dropna().values
            exc = vals[:int(len(vals) * 0.7)]
            exc = exc[exc > threshold] - threshold
            if len(exc) < 10:
                xi_list.append(0.1); sig_list.append(10.0)
            else:
                c, _, s = genpareto.fit(exc, floc=0)
                xi_list.append(c); sig_list.append(s)
                print(f"  Station {i}: xi={c:.4f}, sigma={s:.4f} ({len(exc)} exc.)")
        except Exception as e:
            print(f"  Station {i}: error ({e}), using defaults")
            xi_list.append(0.1); sig_list.append(10.0)
    return np.array(xi_list), np.array(sig_list)


# ========================================================================
# EVALUATE
# ========================================================================
def evaluate(model, loader, A_hat, criterion, device, return_preds=False):
    model.eval()
    total_loss, preds, targs = 0., [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(A_hat, X)
            total_loss += criterion(out, y).item() * X.size(0)
            if return_preds:
                preds.append(out.cpu().numpy())
                targs.append(y.cpu().numpy())
    avg = total_loss / len(loader.dataset)
    if return_preds:
        return avg, np.concatenate(preds), np.concatenate(targs)
    return avg


# ========================================================================
# TRAIN
# ========================================================================
def train(args):
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
    data_npy = os.path.join(root, "clean", "knowair_bthsa.npy")
    adj_npy = os.path.join(root, "clean", "adj_mat_knowair_bthsa.npy")

    # Data
    print("\n[1/4] Loading data...")
    train_ld, val_ld, test_ld, mean, std = load_data(
        data_npy, args.batch_size, args.seq_len, args.pre_len, args.target_idx, args.limit_years
    )
    raw = np.load(data_npy)
    if args.limit_years is not None:
        max_len = args.limit_years * 8760
        if raw.shape[0] > max_len:
            raw = raw[-max_len:]
            
    nf = mean.shape[0]
    print(f"  Features: {nf}, Target idx: {args.target_idx}")
    print(f"  Train/Val/Test: {len(train_ld.dataset)}/{len(val_ld.dataset)}/{len(test_ld.dataset)}")

    # Graph
    print("\n[2/4] Building graph...")
    A_hat_np = build_graph(adj_npy)
    nn_nodes = A_hat_np.shape[0]
    A_hat = torch.FloatTensor(A_hat_np).to(device)
    print(f"  Nodes: {nn_nodes}")

    # POT
    print(f"\n[3/4] Fitting GPD (threshold={args.threshold})...")
    params = []
    for i in range(nn_nodes):
        valid = raw[:, i, args.target_idx]
        valid = valid[~np.isnan(valid)]
        exc = valid[valid > args.threshold] - args.threshold
        if len(exc) > 10:
            c, loc, scale = genpareto.fit(exc, floc=0)
            params.append(float(c))
            params.append(float(scale))
        else:
            params.append(0.1)
            params.append(10.0)
            
    # reshaping for tensor
    xi_arr = np.array(params[0::2], dtype=np.float32)
    sig_arr = np.array(params[1::2], dtype=np.float32)
    xi = torch.FloatTensor(xi_arr)
    sig = torch.FloatTensor(sig_arr)
    
    t_mean, t_std = float(mean[args.target_idx]), float(std[args.target_idx])

    evt_loss = EVTGPDLoss(
        xi=xi, sigma=sig, mean_vals=t_mean, std_vals=t_std,
        threshold=args.threshold, beta_1=args.beta1, beta_2=args.beta2
    ).to(device)
    mse_loss = nn.MSELoss()

    # Model
    print(f"\n[4/4] Building Hybrid STGCN + XLinear + EVT-GPD v2...")
    model = STGCN_XLinear(
        num_nodes=nn_nodes,
        num_features=nf,
        num_timesteps_input=args.seq_len,
        num_timesteps_output=args.pre_len,
        gating_ff=args.t_ff
    ).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {params:,}")
    print(f"  Gating FF: {args.t_ff}")
    print(f"  GPD Warmup: {args.warmup} epochs | beta1={args.beta1}, beta2={args.beta2}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5, min_lr=1e-6)

    best_val, patience_cnt = float('inf'), 0
    os.makedirs("checkpoints", exist_ok=True)

    print(f"\n{'='*75}")
    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val MSE':>10} | {'LR':>10} | {'Time':>6} | Status")
    print(f"{'='*75}")

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        use_evt = epoch >= args.warmup
        evt_loss.set_warmup(use_evt)
        label = "EVT" if use_evt else "MSE"

        ep_loss = 0.
        for X, y in train_ld:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(A_hat, X)
            loss = evt_loss(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            ep_loss += loss.item() * X.size(0)

        ep_loss /= len(train_ld.dataset)
        val = evaluate(model, val_ld, A_hat, mse_loss, device)
        scheduler.step(val)
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        if val < best_val:
            best_val = val
            patience_cnt = 0
            torch.save(model.state_dict(), 'checkpoints/best_stgcn_xlinear.pth')
            status = f"★ Best ({best_val:.4f})"
        else:
            patience_cnt += 1
            status = f"wait {patience_cnt}/{args.patience}"

        print(f"{epoch+1:>6} | {label:>4} {ep_loss:>7.4f} | {val:>10.4f} | {lr:>10.6f} | {elapsed:>5.1f}s | {status}")

        if patience_cnt >= args.patience:
            print(f"\n  Early stopping at epoch {epoch + 1}!")
            break

    # Test
    print(f"\n{'='*75}")
    print("TESTING — Hybrid STGCN + XLinear + EVT-GPD")
    print(f"{'='*75}")

    model.load_state_dict(torch.load('checkpoints/best_stgcn_xlinear.pth', weights_only=True))
    tl, pn, tn = evaluate(model, test_ld, A_hat, mse_loss, device, return_preds=True)

    pr = pn * t_std + t_mean
    tr = tn * t_std + t_mean
    pf, tf_ = pr.flatten(), tr.flatten()

    rmse = np.sqrt(mean_squared_error(tf_, pf))
    mae = mean_absolute_error(tf_, pf)
    r2 = r2_score(tf_, pf)

    print(f"  Test MSE (norm)  : {tl:.4f}")
    print(f"  Test RMSE ({target_name:>10}): {rmse:.2f}")
    print(f"  Test MAE  ({target_name:>10}): {mae:.2f}")
    print(f"  Test R²-Score    : {r2:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hybrid STGCN + XLinear + EVT-GPD')
    parser.add_argument('--seq_len', type=int, default=72)
    parser.add_argument('--pre_len', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--target_idx', type=int, default=0, help='0 = PM2.5 in KnowAir')
    parser.add_argument('--limit_years', type=int, default=2, help='Only use last N years to save RAM')
    parser.add_argument('--threshold', type=float, default=50.0)
    parser.add_argument('--warmup', type=int, default=15)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--beta1', type=float, default=0.99)
    parser.add_argument('--beta2', type=float, default=0.01)
    parser.add_argument('--d_model', type=int, default=64, help='XLinear temporal projection dim')
    parser.add_argument('--t_ff', type=int, default=128, help='Gating hidden dim')

    args = parser.parse_args()
    train(args)
