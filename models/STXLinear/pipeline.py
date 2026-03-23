"""
models/STXLinear/pipeline.py

ST-XLinear Pipeline — Per-Horizon + Spatial Global Token.
Train riêng cho mỗi horizon (T+1, T+3...) × mỗi vùng (North/South).
"""
import os, sys, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════
SELECTED_STATIONS = [1, 4, 5, 16, 17, 27, 7, 18, 24, 30, 31, 32]
REGIONS = {
    'north': [1, 4, 5, 16, 17, 27],
    'south': [7, 18, 24, 30, 31, 32],
}
PM25_COL = 'pm25'

BLOCK      = 'block30'
DATA_DIR   = f'data/split/{BLOCK}'
INFO_PATH  = 'data/info.csv'

EXCLUDE_COLS = ['timestamp', 'split', 'station_id', 'province', 'district',
                'is_frozen', 'is_pm25_sensor_error', 'is_weekend_holiday']

SEQ_LEN    = 48
HORIZONS   = [1, 3, 6, 12, 24]
BATCH_SIZE = 64
EPOCHS     = 50
LR         = 5e-4
PATIENCE   = 10
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Horizon × Region specific model size (v7)
MODEL_CONFIG = {
    # horizon: {region: (d_model, t_ff, c_ff)}
    1:  {'north': (48,  96,  96),  'south': (64,  128, 128)},
    3:  {'north': (48,  96,  96),  'south': (64,  128, 128)},
    6:  {'north': (64,  128, 128), 'south': (96,  192, 192)},
    12: {'north': (96,  192, 192), 'south': (96,  192, 192)},
    24: {'north': (96,  192, 192), 'south': (128, 256, 256)},
}
DROPOUT    = 0.2

# ══════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from metrics import inverse_pm25, get_metrics
from graph_builder import get_base_matrices
from model import STXLinear

print(f"[*] Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════
# ADJACENCY
# ══════════════════════════════════════════════════════════════════════════
def get_adjacency_matrix(sids):
    dist_km, _ = get_base_matrices(INFO_PATH, sids)
    sigma2 = dist_km[dist_km > 0].std() ** 2 if (dist_km > 0).any() else 1.0
    A = np.exp(-(dist_km ** 2) / (sigma2 + 1e-8))
    np.fill_diagonal(A, 1.0)
    A = A / (A.sum(axis=1, keepdims=True) + 1e-8)
    return A


# ══════════════════════════════════════════════════════════════════════════
# DATASET — Per-Horizon, Multi-Station
# ══════════════════════════════════════════════════════════════════════════
class STPerHorizonDataset(Dataset):
    """
    x: (seq_len, num_nodes, num_features)
    y: (num_nodes,)  — PM2.5 at exactly horizon_h steps ahead
    """
    def __init__(self, station_data, seq_len=48, horizon_h=1):
        self.seq_len = seq_len
        self.horizon = horizon_h

        sids = list(station_data.keys())
        min_len = min(len(station_data[s]['features']) for s in sids)

        feats = np.stack([station_data[s]['features'][:min_len] for s in sids], axis=1)
        pm25 = np.stack([station_data[s]['pm25'][:min_len] for s in sids], axis=1)

        # Append PM2.5 as last channel
        self.features = np.concatenate([feats, pm25[:, :, np.newaxis]], axis=2).astype(np.float32)
        self.targets = pm25.astype(np.float32)

        self.valid_len = min_len - seq_len - horizon_h

    def __len__(self):
        return max(0, self.valid_len)

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_len]            # (seq_len, N, F+1)
        y = self.targets[idx + self.seq_len - 1 + self.horizon]  # (N,)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def load_station_data(sids, split_name='train'):
    data = {}
    for sid in sids:
        df = pd.read_csv(os.path.join(DATA_DIR, f'station_{sid}.csv'))
        df = df[df['split'] == split_name].reset_index(drop=True)
        feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c != PM25_COL]
        feat_cols = [c for c in feat_cols if df[c].dtype in ['float64','float32','int64','int32']]
        data[sid] = {
            'features': df[feat_cols].fillna(0).values.astype(np.float32),
            'pm25': df[PM25_COL].fillna(0).values.astype(np.float32),
        }
    return data


# ══════════════════════════════════════════════════════════════════════════
# LOSS
# ══════════════════════════════════════════════════════════════════════════
class CombinedLoss(nn.Module):
    def __init__(self, mse_w=0.7, delta=1.0):
        super().__init__()
        self.mse_w = mse_w
        self.mse = nn.MSELoss()
        self.huber = nn.HuberLoss(delta=delta)
    def forward(self, p, t):
        return self.mse_w * self.mse(p, t) + (1-self.mse_w) * self.huber(p, t)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def run():
    print("=" * 70)
    print(f"  ST-XLinear Pipeline — Per-Horizon | Block: {BLOCK}")
    print(f"  Horizons: {HORIZONS}")
    print("=" * 70)

    all_results = []

    for h in HORIZONS:
        print(f"\n{'='*60}")
        print(f"  HORIZON T+{h}h")
        print(f"{'='*60}")

        for r_name, sids in REGIONS.items():
            print(f"\n  [{r_name.upper()}] T+{h}h ({len(sids)} stations)...")

            # Load data
            tr = load_station_data(sids, 'train')
            va = load_station_data(sids, 'val')
            te = load_station_data(sids, 'test')

            tr_ds = STPerHorizonDataset(tr, SEQ_LEN, h)
            va_ds = STPerHorizonDataset(va, SEQ_LEN, h)
            te_ds = STPerHorizonDataset(te, SEQ_LEN, h)

            num_features = tr_ds.features.shape[2]
            num_nodes = len(sids)

            tr_dl = DataLoader(tr_ds, BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
            va_dl = DataLoader(va_ds, BATCH_SIZE, shuffle=False, pin_memory=True)
            te_dl = DataLoader(te_ds, BATCH_SIZE, shuffle=False, pin_memory=True)

            print(f"    Train={len(tr_ds)}, Val={len(va_ds)}, Test={len(te_ds)} | Features={num_features}")

            # Adjacency
            adj = torch.tensor(get_adjacency_matrix(sids), dtype=torch.float32).to(DEVICE)

           
            _dm, _tf, _cf = MODEL_CONFIG[h][r_name]

            model = STXLinear(
                num_nodes=num_nodes, num_features=num_features,
                seq_len=SEQ_LEN, pred_len=1,
                d_model=_dm, t_ff=_tf, c_ff=_cf, dropout=DROPOUT
            ).to(DEVICE)

            n_params = sum(p.numel() for p in model.parameters())
            print(f"    Params: {n_params:,}")

            optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
            criterion = CombinedLoss()

            save_path = f"models_saved/stxlinear_{r_name}_t{h}.pth"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            best_val = float('inf')
            patience_cnt = 0

            for ep in range(EPOCHS):
                t0 = time.time()
                model.train()
                tl = 0.0
                for bx, by in tr_dl:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    optimizer.zero_grad()
                    out = model(bx, adj).squeeze(1)  # (batch, num_nodes)
                    loss = criterion(out, by)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    tl += loss.item() * bx.size(0)
                tl /= max(len(tr_dl.dataset), 1)

                model.eval()
                vl = 0.0
                with torch.no_grad():
                    for bx, by in va_dl:
                        bx, by = bx.to(DEVICE), by.to(DEVICE)
                        out = model(bx, adj).squeeze(1)
                        vl += criterion(out, by).item() * bx.size(0)
                vl /= max(len(va_dl.dataset), 1)

                scheduler.step()
                dt = time.time() - t0
                print(f"    Ep {ep+1:02d}/{EPOCHS} | T: {tl:.4f} | V: {vl:.4f} | {dt:.1f}s")

                if vl < best_val:
                    best_val = vl
                    torch.save(model.state_dict(), save_path)
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= PATIENCE:
                        print(f"    Early stop ep {ep+1}")
                        break

            # Evaluate
            model.load_state_dict(torch.load(save_path, weights_only=True))
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for bx, by in te_dl:
                    bx = bx.to(DEVICE)
                    out = model(bx, adj).squeeze(1).cpu().numpy()
                    preds.append(out)
                    trues.append(by.numpy())

            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)

            # Inverse per station
            for ni, sid in enumerate(sids):
                preds[:, ni] = inverse_pm25(preds[:, ni], sid)
                trues[:, ni] = inverse_pm25(trues[:, ni], sid)

            y_p = preds.flatten()
            y_t = trues.flatten()
            rmse, mae, r2, mape = get_metrics(y_t, y_p)
            print(f"    [{r_name.upper()}] T+{h:2d} | RMSE={rmse:.2f} | MAE={mae:.2f} | R2={r2*100:.2f}% | MAPE={mape:.2f}%")

            all_results.append({
                'region': r_name, 'horizon': f'T+{h}',
                'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape,
                'n_test': len(y_t)
            })

    # Summary
    print("\n" + "=" * 70)
    print("  FINAL BENCHMARK — ST-XLinear (Per-Horizon + Spatial Token)")
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
            print(f"  T+{h:<3d}  RMSE={sum(r['RMSE']*r['n_test'] for r in hr)/total:.2f}  "
                  f"MAE={sum(r['MAE']*r['n_test'] for r in hr)/total:.2f}  "
                  f"R2={sum(r['R2']*r['n_test'] for r in hr)/total*100:.2f}%  "
                  f"MAPE={sum(r['MAPE']*r['n_test'] for r in hr)/total:.2f}%")
    print("=" * 70)


if __name__ == '__main__':
    run()
