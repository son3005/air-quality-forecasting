"""
run_all_stations.py — XLinear Batch Evaluation (in-process)
Chạy XLinear trên tất cả 32 stations trong cùng process,
lưu kết quả vào results.csv
"""
import os
import sys
import csv
import numpy as np

# Add XLinear to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'XLinear'))
sys.path.insert(0, os.path.dirname(__file__))

from main import XLinearConfig, StationDataset
from models.XLinear import Model as XLinear

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

STATIONS = [1,2,3,5,6,7,9,10,12,13,14,15,16,17,18,20,21,23,24,25,26,27,28,29,30,31,32]
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.csv")
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'clean')
CKPT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')


def run_station(sid, device):
    torch.manual_seed(42); np.random.seed(42)

    df = pd.read_csv(os.path.join(DATA_DIR, f'station_{sid}.csv')).drop(columns=['time'])
    cols = list(df.columns)
    target_idx = cols.index('ma_pm25_24')

    data = df.values.astype(np.float32)
    T = len(data)
    ts = int(T * 0.7); te = ts + int(T * 0.1)

    mean_ = data[:ts].mean(axis=0)
    std_  = data[:ts].std(axis=0) + 1e-5
    normed = (data - mean_) / std_

    train_ds = StationDataset(normed[:ts],   72, 24, target_idx)
    val_ds   = StationDataset(normed[ts:te], 72, 24, target_idx)
    test_ds  = StationDataset(normed[te:],   72, 24, target_idx)

    train_ld = DataLoader(train_ds, 64, shuffle=True, drop_last=True)
    val_ld   = DataLoader(val_ds,   64, shuffle=False)
    test_ld  = DataLoader(test_ds,  64, shuffle=False)

    cfg = XLinearConfig(seq_len=72, pred_len=24, enc_in=len(cols),
                        d_model=64, t_ff=128, c_ff=128,
                        embed_dropout=0.3, head_dropout=0.3,
                        t_dropout=0.3, c_dropout=0.3,
                        usenorm=True, features='MS')
    model = XLinear(cfg).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-6)

    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt = os.path.join(CKPT_DIR, f'xlinear_s{sid}.pth')

    best_val, wait = float('inf'), 0
    for epoch in range(100):
        model.train()
        for X, y in train_ld:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(X), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = sum(criterion(model(X.to(device)), y.to(device)).item() * X.size(0)
                           for X, y in val_ld) / len(val_ds)
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss; wait = 0
            torch.save(model.state_dict(), ckpt)
        else:
            wait += 1
            if wait >= 20: break

    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()
    preds, targs = [], []
    with torch.no_grad():
        for X, y in test_ld:
            preds.append(model(X.to(device)).cpu().numpy())
            targs.append(y.numpy())

    pr = np.concatenate(preds).flatten() * std_[target_idx] + mean_[target_idx]
    tr = np.concatenate(targs).flatten() * std_[target_idx] + mean_[target_idx]

    return {
        "station": sid,
        "r2":   float(r2_score(tr, pr)),
        "rmse": float(np.sqrt(mean_squared_error(tr, pr))),
        "mae":  float(mean_absolute_error(tr, pr)),
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    print(f"{'Station':>8} | {'R2':>8} | {'RMSE':>8} | {'MAE':>8}")
    print("-" * 42)

    rows = []
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["station", "r2", "rmse", "mae"])
        writer.writeheader()

        for sid in STATIONS:
            try:
                res = run_station(sid, device)
            except Exception as e:
                print(f"  Station {sid} ERROR: {e}")
                res = {"station": sid, "r2": None, "rmse": None, "mae": None}

            rows.append(res)
            writer.writerow(res)
            f.flush()

            r2s   = f"{res['r2']:.4f}"   if res['r2']   is not None else "ERROR"
            rmses = f"{res['rmse']:.4f}" if res['rmse'] is not None else "ERROR"
            maes  = f"{res['mae']:.4f}"  if res['mae']  is not None else "ERROR"
            print(f"{sid:>8} | {r2s:>8} | {rmses:>8} | {maes:>8}")

    valid = [r for r in rows if r['r2'] is not None]
    if valid:
        avg_r2   = np.mean([r['r2']   for r in valid])
        avg_rmse = np.mean([r['rmse'] for r in valid])
        avg_mae  = np.mean([r['mae']  for r in valid])
        print("-" * 42)
        print(f"{'AVG':>8} | {avg_r2:>8.4f} | {avg_rmse:>8.4f} | {avg_mae:>8.4f}")

    print(f"\nResults saved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
