"""
XLinear — Single Station Training Script
Dùng XLinear gốc với chế độ exogenous:
  Endogenous (target): ma_pm25_24 (cột cuối)
  Exogenous: 25 features còn lại

Chạy: python main.py --station 1
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Import XLinear model gốc
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'XLinear'))
from models.XLinear import Model as XLinear


# ============================================================
# Config object (thay thế argparse.Namespace cho XLinear)
# ============================================================
class XLinearConfig:
    def __init__(self, seq_len, pred_len, enc_in, d_model=64, t_ff=128, c_ff=128,
                 embed_dropout=0.1, head_dropout=0.1, t_dropout=0.1, c_dropout=0.1,
                 usenorm=True, features='MS'):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in        # tổng số channels (bao gồm endogenous + exogenous)
        self.d_model = d_model
        self.t_ff = t_ff
        self.c_ff = c_ff
        self.embed_dropout = embed_dropout
        self.head_dropout = head_dropout
        self.t_dropout = t_dropout
        self.c_dropout = c_dropout
        self.usenorm = usenorm
        self.features = features    # 'MS': exogenous mode (last col = target)


# ============================================================
# Dataset
# ============================================================
class StationDataset(Dataset):
    """
    Single-station time series dataset.
    X: [seq_len, num_features]
    y: [pred_len, 1]  — chỉ target column
    """
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int, target_idx: int):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_idx = target_idx
        # Reorder columns so target is LAST (XLinear 'MS' mode: last col = endogenous)
        others = [i for i in range(data.shape[1]) if i != target_idx]
        self.data = data[:, others + [target_idx]]   # [T, C] with target last
        self.n = len(self.data) - seq_len - pred_len + 1

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        X = self.data[i:i + self.seq_len]                            # [seq_len, C]
        y = self.data[i + self.seq_len:i + self.seq_len + self.pred_len, -1:]  # [pred_len, 1]
        return torch.FloatTensor(X), torch.FloatTensor(y)


# ============================================================
# Main
# ============================================================
def train(args):
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---- Load data ----
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'clean')
    path = os.path.join(data_dir, f'station_{args.station}.csv')
    df = pd.read_csv(path).drop(columns=['time'])
    cols = list(df.columns)
    target_idx = cols.index(args.target_col)
    print(f"\nStation {args.station} | Features: {len(cols)} | Target: {args.target_col} (idx={target_idx})")

    data = df.values.astype(np.float32)
    T = len(data)
    ts = int(T * 0.7)
    vs = int(T * 0.1)
    te = ts + vs

    # Normalize on train set
    mean = data[:ts].mean(axis=0)
    std  = data[:ts].std(axis=0) + 1e-5
    normed = (data - mean) / std

    train_ds = StationDataset(normed[:ts],      args.seq_len, args.pred_len, target_idx)
    val_ds   = StationDataset(normed[ts:te],    args.seq_len, args.pred_len, target_idx)
    test_ds  = StationDataset(normed[te:],      args.seq_len, args.pred_len, target_idx)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    val_ld   = DataLoader(val_ds,  batch_size=args.batch_size, shuffle=False)
    test_ld  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    print(f"Train/Val/Test samples: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    # ---- Model ----
    cfg = XLinearConfig(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        enc_in=len(cols),            # total channels
        d_model=args.d_model,
        t_ff=args.t_ff,
        c_ff=args.c_ff,
        embed_dropout=args.dropout,
        head_dropout=args.dropout,
        t_dropout=args.dropout,
        c_dropout=args.dropout,
        usenorm=True,
        features='MS'                # exogenous mode: last col = endogenous
    )
    model = XLinear(cfg).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {params:,} | d_model={args.d_model} | t_ff={args.t_ff}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-6)

    os.makedirs('checkpoints', exist_ok=True)
    ckpt_path = f'checkpoints/xlinear_s{args.station}.pth'

    best_val, patience_cnt = float('inf'), 0

    print(f"\n{'='*65}")
    print(f"{'Epoch':>6} | {'Train MSE':>10} | {'Val MSE':>10} | {'LR':>10} | Status")
    print(f"{'='*65}")

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        ep_loss = 0.
        for X, y in train_ld:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            # XLinear forward: input [B, seq_len, C] → out [B, pred_len, 1]
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            ep_loss += loss.item() * X.size(0)
        ep_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0.
        with torch.no_grad():
            for X, y in val_ld:
                X, y = X.to(device), y.to(device)
                out = model(X)
                val_loss += criterion(out, y).item() * X.size(0)
        val_loss /= len(val_ds)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']

        if val_loss < best_val:
            best_val = val_loss
            patience_cnt = 0
            torch.save(model.state_dict(), ckpt_path)
            status = f"★ Best ({best_val:.4f})"
        else:
            patience_cnt += 1
            status = f"wait {patience_cnt}/{args.patience}"

        print(f"{epoch+1:>6} | {ep_loss:>10.4f} | {val_loss:>10.4f} | {lr:>10.6f} | {status}")

        if patience_cnt >= args.patience:
            print(f"\n  Early stopping at epoch {epoch+1}!")
            break

    # ---- Test ----
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()
    preds, targs = [], []
    with torch.no_grad():
        for X, y in test_ld:
            X = X.to(device)
            out = model(X).cpu().numpy()
            preds.append(out); targs.append(y.numpy())

    pn = np.concatenate(preds)  # [N, pred_len, 1]
    tn = np.concatenate(targs)

    # Denormalize
    t_mean, t_std = float(mean[target_idx]), float(std[target_idx])
    pr = pn.flatten() * t_std + t_mean
    tr = tn.flatten() * t_std + t_mean

    rmse = np.sqrt(mean_squared_error(tr, pr))
    mae  = mean_absolute_error(tr, pr)
    r2   = r2_score(tr, pr)

    print(f"\n{'='*65}")
    print(f"TEST RESULTS — XLinear (Station {args.station})")
    print(f"{'='*65}")
    print(f"  RMSE  : {rmse:.4f}")
    print(f"  MAE   : {mae:.4f}")
    print(f"  R²    : {r2:.4f}")

    # Machine-readable output for batch runner
    if getattr(args, 'json_out', False):
        print(json.dumps({"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--station',    type=int,   default=1,        help='Station ID (1-32)')
    parser.add_argument('--target_col', type=str,   default='ma_pm25_24')
    parser.add_argument('--seq_len',    type=int,   default=72)
    parser.add_argument('--pred_len',   type=int,   default=24,       help='Forecast horizon (24=1 day)')
    parser.add_argument('--batch_size', type=int,   default=64)
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--patience',   type=int,   default=20)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--wd',         type=float, default=1e-3,     help='Weight decay')
    parser.add_argument('--d_model',    type=int,   default=64)
    parser.add_argument('--t_ff',       type=int,   default=128)
    parser.add_argument('--c_ff',       type=int,   default=128)
    parser.add_argument('--dropout',    type=float, default=0.3)
    parser.add_argument('--json_out',   action='store_true', help='Print JSON result for batch parsing')
    args = parser.parse_args()
    train(args)
