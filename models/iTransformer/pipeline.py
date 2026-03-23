"""
models/iTransformer/pipeline.py

iTransformer Pipeline — Per-Horizon Training with Region-based Evaluation.
Sử dụng iTransformer (ICLR 2024) cho PM2.5 multi-station forecasting.

Cách tiếp cận:
  - Mỗi station là 1 VARIATE → attention học cross-station correlations
  - Per-horizon training (pred_len=1 per model)
  - Region-based (north/south)
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

# iTransformer hyperparameters (per horizon × region)
# Format: (d_model, d_ff, n_heads, e_layers)
MODEL_CONFIG = {
    # Shorter horizons → smaller model (less overfitting)
    1:  {'north': (64,  128, 4, 2), 'south': (64,  128, 4, 2)},
    3:  {'north': (64,  128, 4, 2), 'south': (64,  128, 4, 2)},
    6:  {'north': (64,  128, 4, 2), 'south': (128, 256, 4, 2)},
    12: {'north': (128, 256, 4, 2), 'south': (128, 256, 4, 2)},
    24: {'north': (128, 256, 4, 2), 'south': (128, 256, 4, 3)},
}

# ══════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from metrics import inverse_pm25, get_metrics

from model import iTransformer

print(f"[*] Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════

def load_station_data(sids, split_name='train'):
    """Load normalized data for selected stations filtered by split."""
    station_data = {}
    for sid in sids:
        df = pd.read_csv(os.path.join(DATA_DIR, f'station_{sid}.csv'))
        df_s = df[df['split'] == split_name].reset_index(drop=True)

        feat_cols = [c for c in df_s.columns if c not in EXCLUDE_COLS and c != PM25_COL
                     and df_s[c].dtype in ['float64', 'float32', 'int64', 'int32']]

        station_data[sid] = {
            'features': df_s[feat_cols].fillna(0).values.astype(np.float32),
            'pm25': df_s[PM25_COL].fillna(0).values.astype(np.float32),
        }
    return station_data


class iTransformerDataset(Dataset):
    """
    Multi-station dataset for iTransformer.

    Variate design: each FEATURE CHANNEL across ALL STATIONS = 1 variate.
    x: (seq_len, num_features) — all features stacked across stations
    y: (num_nodes,) — PM2.5 prediction for each station

    Effectively: num_variates = num_features (per-station, stacked)
    This treats the problem like standard multivariate forecasting.
    """
    def __init__(self, station_data, sids, seq_len=48, horizon=1):
        self.seq_len = seq_len
        self.horizon = horizon
        self.sids = sids
        num_nodes = len(sids)

        # Align all stations to same length
        min_len = min(len(v['features']) for v in station_data.values())

        # Stack: features for each station + PM2.5 for each station
        # Final shape: (T, N_features + N_nodes_pm25)
        features_list = []
        pm25_list = []
        for sid in sids:
            features_list.append(station_data[sid]['features'][:min_len])
            pm25_list.append(station_data[sid]['pm25'][:min_len].reshape(-1, 1))

        # Each station's PM2.5 is a separate variate for iTransformer
        # Shape: (T, N_stations) — PM2.5 channels
        self.pm25_matrix = np.concatenate(pm25_list, axis=1)  # (T, N)

        # Use first station's features as shared features (normalized similarly)
        # + all stations' PM2.5 as separate variates
        self.shared_features = features_list[0]  # (T, F)
        self.num_features = self.shared_features.shape[1]

        # Variates = shared_features + each station's PM2.5
        # Total variates = F + N
        self.data = np.concatenate([self.shared_features, self.pm25_matrix], axis=1)
        self.num_variates = self.data.shape[1]
        self.total_len = min_len
        self.valid_len = self.total_len - seq_len - horizon + 1

    def __len__(self):
        return max(0, self.valid_len)

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]  # (seq_len, num_variates)
        # Target: PM2.5 at idx + seq_len - 1 + horizon for all stations
        t_idx = idx + self.seq_len - 1 + self.horizon
        y = self.pm25_matrix[t_idx]  # (N,)
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32))


# ══════════════════════════════════════════════════════════════════════════
# LOSS
# ══════════════════════════════════════════════════════════════════════════

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.huber = nn.HuberLoss(delta=1.0)
        self.mae = nn.L1Loss()

    def forward(self, pred, target):
        return self.alpha * self.huber(pred, target) + (1 - self.alpha) * self.mae(pred, target)


# ══════════════════════════════════════════════════════════════════════════
# TRAIN + EVALUATE
# ══════════════════════════════════════════════════════════════════════════

def run():
    print("=" * 70)
    print(f"  iTransformer Pipeline — Per-Horizon | Block: {BLOCK}")
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
            train_data = load_station_data(sids, 'train')
            val_data = load_station_data(sids, 'val')
            test_data = load_station_data(sids, 'test')

            train_ds = iTransformerDataset(train_data, sids, SEQ_LEN, h)
            val_ds = iTransformerDataset(val_data, sids, SEQ_LEN, h)
            test_ds = iTransformerDataset(test_data, sids, SEQ_LEN, h)

            print(f"    Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)} | "
                  f"Variates={train_ds.num_variates}")

            train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
            val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, pin_memory=True)
            test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, pin_memory=True)

            # Model config
            d_model, d_ff, n_heads, e_layers = MODEL_CONFIG[h][r_name]
            num_variates = train_ds.num_variates
            num_nodes = len(sids)

            model = iTransformer(
                seq_len=SEQ_LEN,
                pred_len=1,
                enc_in=num_variates,
                d_model=d_model,
                n_heads=n_heads,
                e_layers=e_layers,
                d_ff=d_ff,
                dropout=0.15,
                activation='gelu',
                use_norm=True,
            ).to(DEVICE)

            n_params = sum(p.numel() for p in model.parameters())
            print(f"    Params: {n_params:,}")

            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
            criterion = CombinedLoss(alpha=0.7)

            # Training
            best_val = float('inf')
            patience_cnt = 0
            save_path = f"models_saved/itransformer_{r_name}_t{h}.pth"
            os.makedirs('models_saved', exist_ok=True)

            for epoch in range(EPOCHS):
                t0 = time.time()
                model.train()
                losses = []
                for bx, by in train_loader:
                    bx = bx.to(DEVICE, non_blocking=True)
                    by = by.to(DEVICE, non_blocking=True)

                    optimizer.zero_grad()
                    # out: (B, 1, num_variates) — forecast for all variates
                    out = model(bx)
                    # We only care about PM2.5 variates (last N columns)
                    pm25_pred = out[:, 0, -num_nodes:]  # (B, N)
                    loss = criterion(pm25_pred, by)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    losses.append(loss.item())
                scheduler.step()

                # Validation
                model.eval()
                vlosses = []
                with torch.no_grad():
                    for bx, by in val_loader:
                        bx = bx.to(DEVICE, non_blocking=True)
                        by = by.to(DEVICE, non_blocking=True)
                        out = model(bx)
                        pm25_pred = out[:, 0, -num_nodes:]
                        vlosses.append(criterion(pm25_pred, by).item())

                tl, vl = np.mean(losses), np.mean(vlosses)
                dt = time.time() - t0
                print(f"    Ep {epoch+1:02d}/{EPOCHS} | T: {tl:.4f} | V: {vl:.4f} | {dt:.1f}s")

                if vl < best_val:
                    best_val = vl
                    torch.save(model.state_dict(), save_path)
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= PATIENCE:
                        print(f"    Early stop ep {epoch+1}")
                        break

            # ═══════════════════════════════════════════════════════════
            # EVALUATION
            # ═══════════════════════════════════════════════════════════
            model.load_state_dict(torch.load(save_path, weights_only=True, map_location=DEVICE))
            model.eval()

            all_preds, all_trues = [], []
            with torch.no_grad():
                for bx, by in test_loader:
                    bx = bx.to(DEVICE, non_blocking=True)
                    out = model(bx)
                    pm25_pred = out[:, 0, -num_nodes:].cpu().numpy()  # (B, N)
                    pm25_true = by.numpy()  # (B, N)

                    # Inverse per station
                    for node_idx, sid in enumerate(sids):
                        p_inv = inverse_pm25(pm25_pred[:, node_idx], sid)
                        t_inv = inverse_pm25(pm25_true[:, node_idx], sid)
                        all_preds.extend(p_inv)
                        all_trues.extend(t_inv)

            y_true = np.array(all_trues)
            y_pred = np.array(all_preds)
            rmse, mae, r2, mape = get_metrics(y_true, y_pred)
            print(f"    [{r_name.upper()}] T+{h:2d} | RMSE={rmse:.2f} | MAE={mae:.2f} | "
                  f"R2={r2*100:.2f}% | MAPE={mape:.2f}%")

            all_results.append({
                'region': r_name, 'horizon': f'T+{h}',
                'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape,
                'n_test': len(y_true),
            })

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  FINAL BENCHMARK — iTransformer (Per-Horizon)")
    print("=" * 70)
    print(f"{'Region':<10} {'Horizon':<8} {'RMSE':>8} {'MAE':>8} {'R2 %':>8} {'MAPE %':>10}")
    print("-" * 55)
    for r in all_results:
        print(f"{r['region']:<10} {r['horizon']:<8} {r['RMSE']:>8.2f} {r['MAE']:>8.2f} "
              f"{r['R2']*100:>7.2f}% {r['MAPE']:>9.2f}%")

    print("\n" + "-" * 55)
    print("AGGREGATED (weighted by test size):")
    for h in HORIZONS:
        hr = [r for r in all_results if r['horizon'] == f'T+{h}']
        if hr:
            total = sum(r['n_test'] for r in hr)
            w = lambda key: sum(r[key]*r['n_test'] for r in hr) / total
            print(f"  T+{h:<3d}  RMSE={w('RMSE'):.2f}  MAE={w('MAE'):.2f}  "
                  f"R2={w('R2')*100:.2f}%  MAPE={w('MAPE'):.2f}%")
    print("=" * 70)


if __name__ == '__main__':
    run()
