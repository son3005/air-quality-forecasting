"""
models/XLinear/pipeline.py

XLinear Pipeline — Dataset-Driven with Region-based Training.
Refactored từ S09_XLinear_Clean với 12 trạm mới.
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
# CONFIG — Chỉnh sửa tại đây
# ══════════════════════════════════════════════════════════════════════════
SELECTED_STATIONS = [1, 4, 5, 16, 17, 27, 7, 18, 24, 30, 31, 32]
REGIONS = {
    'north': [1, 4, 5, 16, 17, 27],
    'south': [7, 18, 24, 30, 31, 32],
}
SID_TO_IDX = {sid: i for i, sid in enumerate(SELECTED_STATIONS)}
PM25_COL = 'pm25'

# --- Block Split Selection ---
# Chọn 1 trong 3: 'block5', 'block7', 'block30'
BLOCK      = 'block7'
DATA_DIR   = f'data/split/{BLOCK}'
INFO_PATH  = 'data/info.csv'

DROP_FEATURES = [
    'is_frozen', 'is_pm25_sensor_error', 'is_weekend_holiday',
    'station_id', 'province', 'district', 'timestamp', 'split',
]
PRECURSOR_COLS = ['pm10', 'no2', 'so2', 'co', 'o3']
PRECURSOR_LAGS = [1, 3, 6, 12, 24]
FUTURE_WEATHER_COLS = ['temp', 'wind_spd', 'precip', 'rh']
NEIGHBOR_LAGS = [1, 3, 6, 12, 24]

SEQ_LEN    = 48
HORIZONS   = [1, 3, 6, 12, 24]
BATCH_SIZE = 128
EPOCHS     = 20
LR         = 5e-4
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CORR_THRESHOLD = 0.05

# ══════════════════════════════════════════════════════════════════════════
# IMPORTS — Shared modules
# ══════════════════════════════════════════════════════════════════════════
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from metrics import inverse_pm25, get_metrics
from graph_builder import get_base_matrices
from XLinear import Model as XLinearModel

print(f"[*] Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════
# ASYMMETRIC HUBER LOSS
# ══════════════════════════════════════════════════════════════════════════

class AsymmetricHuberLoss(nn.Module):
    def __init__(self, delta=1.0, alpha=2.0):
        super().__init__()
        self.delta = delta
        self.alpha = alpha

    def forward(self, pred, true):
        error = true - pred
        abs_error = torch.abs(error)
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        base_loss = 0.5 * quadratic**2 + self.delta * linear
        weight = torch.where(error > 0, self.alpha, 1.0)
        return (weight * base_loss).mean()


# ══════════════════════════════════════════════════════════════════════════
# REGION KNN
# ══════════════════════════════════════════════════════════════════════════

def build_region_knn(region_sids, k=5):
    dist_km, _ = get_base_matrices(INFO_PATH, SELECTED_STATIONS)
    knn, dist_map = {}, {}
    for sid in region_sids:
        i = SID_TO_IDX[sid]
        candidates = [SID_TO_IDX[s] for s in region_sids if s != sid]
        if not candidates:
            knn[i] = []; dist_map[i] = {}; continue
        dists = [(c, dist_km[i, c]) for c in candidates]
        dists.sort(key=lambda x: x[1])
        neighbors = [c for c, _ in dists[:k]]
        knn[i] = neighbors
        dist_map[i] = {c: d for c, d in dists[:k]}
    return knn, dist_km


# ══════════════════════════════════════════════════════════════════════════
# CHRONOLOGICAL SPLIT
# ══════════════════════════════════════════════════════════════════════════

def create_split_from_column(df):
    """Use pre-computed split column from data/split/{block}/."""
    splits = df['split'].values.astype('U5')
    n_tr = (splits == 'train').sum()
    n_va = (splits == 'val').sum()
    n_te = (splits == 'test').sum()
    print(f"    Block split ({BLOCK}): train={n_tr}, val={n_va}, test={n_te}")
    return splits


# ══════════════════════════════════════════════════════════════════════════
# FEATURE ENRICHMENT
# ══════════════════════════════════════════════════════════════════════════

def compute_precursor_lags(df):
    feats = {}
    for col in PRECURSOR_COLS:
        if col in df.columns:
            for lag in PRECURSOR_LAGS:
                feats[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return pd.DataFrame(feats, index=df.index)


def compute_future_weather(df, horizon_h):
    feats = {}
    for col in FUTURE_WEATHER_COLS:
        if col in df.columns:
            feats[f'{col}_fut_h{horizon_h}'] = df[col].shift(-horizon_h)
    return pd.DataFrame(feats, index=df.index)


def compute_neighbor_lagged_stats(s_idx, pm25_arrays, knn_indices, distances_matrix):
    neighbor_idxs = knn_indices.get(s_idx, [])
    if not neighbor_idxs:
        return pd.DataFrame()
    neighbor_pm25 = [pm25_arrays[n] for n in neighbor_idxs]
    neighbor_dists = [distances_matrix[s_idx, n] for n in neighbor_idxs]
    stacked = np.column_stack(neighbor_pm25)
    inv_dists = np.array([1.0 / (d + 1e-3) for d in neighbor_dists])
    inv_dists_norm = inv_dists / inv_dists.sum()
    result = {}
    for lag in NEIGHBOR_LAGS:
        lagged = pd.DataFrame(stacked).shift(lag).values
        result[f'nbr_mean_lag{lag}'] = np.nanmean(lagged, axis=1)
        result[f'nbr_max_lag{lag}'] = np.nanmax(lagged, axis=1)
        result[f'nbr_std_lag{lag}'] = np.nanstd(lagged, axis=1)
        result[f'nbr_wmean_lag{lag}'] = (lagged * inv_dists_norm).sum(axis=1)
    if len(neighbor_pm25) > 0:
        nearest = pd.Series(neighbor_pm25[0])
        result['nbr_nearest_lag1'] = nearest.shift(1).values
        result['nbr_nearest_lag6'] = nearest.shift(6).values
        result['nbr_nearest_lag24'] = nearest.shift(24).values
    return pd.DataFrame(result)


# ══════════════════════════════════════════════════════════════════════════
# FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════

def select_features(feat_df, target_series, threshold=CORR_THRESHOLD):
    num_cols = feat_df.select_dtypes(include='number').columns
    corrs = feat_df[num_cols].corrwith(target_series).abs()
    keep = corrs[corrs >= threshold].index.tolist()
    dropped = corrs[corrs < threshold].index.tolist()
    if dropped:
        print(f"    Dropped {len(dropped)} low-corr features: {dropped[:5]}{'...' if len(dropped)>5 else ''}")
    return keep


# ══════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════

class XLinearDataset(Dataset):
    def __init__(self, station_features, station_targets, station_ids,
                 seq_len=48, horizon=1):
        self.seq_len = seq_len
        self.horizon = horizon
        self.index = []
        self.features = station_features
        self.targets = station_targets
        self.sids = station_ids

        for s_local, (feat, tgt) in enumerate(zip(station_features, station_targets)):
            max_start = len(feat) - seq_len - horizon
            for i in range(max(0, max_start + 1)):
                t_idx = i + seq_len - 1 + horizon
                if t_idx < len(tgt) and not np.isnan(tgt[t_idx]):
                    self.index.append((s_local, i))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        s_local, start = self.index[idx]
        x = self.features[s_local][start:start + self.seq_len]
        y = self.targets[s_local][start + self.seq_len - 1 + self.horizon]
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor([y], dtype=torch.float32),
                self.sids[s_local])


# ══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════

class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_xlinear_config(num_features, region_name='north'):
    cfg = DotDict()
    cfg.seq_len = SEQ_LEN
    cfg.pred_len = 1
    cfg.enc_in = num_features
    cfg.usenorm = False
    cfg.embed_dropout = 0.1
    cfg.head_dropout = 0.1
    cfg.t_dropout = 0.1
    cfg.c_dropout = 0.1
    cfg.features = 'MS'
    if region_name == 'south':
        cfg.d_model = 256; cfg.t_ff = 512; cfg.c_ff = 512
    else:
        cfg.d_model = 128; cfg.t_ff = 256; cfg.c_ff = 256
    return cfg


# ══════════════════════════════════════════════════════════════════════════
# PREPARE DATA
# ══════════════════════════════════════════════════════════════════════════

def prepare_region_data(region_name, region_sids, horizon_h):
    print(f"\n  [{region_name.upper()}] Preparing features for T+{horizon_h}h ({len(region_sids)} stations)...")

    all_dfs = {}
    for sid in SELECTED_STATIONS:
        all_dfs[sid] = pd.read_csv(os.path.join(DATA_DIR, f'station_{sid}.csv'))

    pm25_arrays = {}
    for s_idx, sid in enumerate(SELECTED_STATIONS):
        pm25_arrays[s_idx] = all_dfs[sid][PM25_COL].values

    knn, distances = build_region_knn(region_sids, k=min(5, len(region_sids) - 1))
    sample_df = all_dfs[region_sids[0]]
    strat_splits = create_split_from_column(sample_df)

    result = {'train': {}, 'val': {}, 'test': {}}
    selected_cols = None

    for sid in region_sids:
        df = all_dfs[sid]
        s_idx = SID_TO_IDX[sid]
        splits = strat_splits

        base_cols = [c for c in df.columns if c not in DROP_FEATURES and c != PM25_COL]
        base_feats = df[base_cols].copy()

        feat_df = base_feats.copy()
        feat_df = feat_df.fillna(0.0)

        if selected_cols is None:
            train_mask = splits == 'train'
            n_train = train_mask.sum()
            if n_train > 100:
                train_idx = np.where(train_mask)[0]
                feat_train = feat_df.iloc[train_idx]
                target_train = df[PM25_COL].iloc[train_idx]
                selected_cols = select_features(feat_train, target_train)
            else:
                selected_cols = feat_df.select_dtypes(include='number').columns.tolist()
            print(f"    Selected {len(selected_cols)} raw features (from {len(feat_df.columns)})")

        valid_cols = [c for c in selected_cols if c in feat_df.columns]
        features = feat_df[valid_cols].values.astype(np.float32)
        pm25_arr = df[PM25_COL].values.reshape(-1, 1).astype(np.float32)
        features = np.concatenate([features, pm25_arr], axis=1)
        features = np.nan_to_num(features, nan=0.0)
        target = df[PM25_COL].values.astype(np.float32)

        for split_name in ['train', 'val', 'test']:
            mask = splits == split_name
            result[split_name][sid] = (features[mask], target[mask])

    num_features = features.shape[1]
    print(f"    Total features: {num_features}")
    for sn in ['train', 'val', 'test']:
        total = sum(len(result[sn][s][0]) for s in region_sids)
        print(f"    {sn}: {total:,} rows")
    return result, num_features


# ══════════════════════════════════════════════════════════════════════════
# TRAIN + EVALUATE
# ══════════════════════════════════════════════════════════════════════════

def train_region(region_name, region_sids, data, num_features, horizon_h):
    train_ds = XLinearDataset(
        [data['train'][s][0] for s in region_sids],
        [data['train'][s][1] for s in region_sids],
        region_sids, SEQ_LEN, horizon_h)
    val_ds = XLinearDataset(
        [data['val'][s][0] for s in region_sids],
        [data['val'][s][1] for s in region_sids],
        region_sids, SEQ_LEN, horizon_h)
    test_ds = XLinearDataset(
        [data['test'][s][0] for s in region_sids],
        [data['test'][s][1] for s in region_sids],
        region_sids, SEQ_LEN, horizon_h)

    if len(train_ds) == 0 or len(val_ds) == 0:
        print(f"    [SKIP] Not enough data for {region_name}")
        return None

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, pin_memory=True)

    print(f"    Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    cfg = get_xlinear_config(num_features, region_name=region_name)
    model = XLinearModel(cfg).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Model params: {n_params:,} | Features: {num_features}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = AsymmetricHuberLoss(delta=1.0, alpha=2.0)

    best_val = float('inf')
    patience_cnt = 0
    model_path = f'models_saved/xlinear_{region_name}_t{horizon_h}.pth'

    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        losses = []
        for bx, by, _ in train_loader:
            bx = bx.to(DEVICE, non_blocking=True)
            by = by.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            out = model(bx)[:, -1, :]
            loss = criterion(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        model.eval()
        vlosses = []
        with torch.no_grad():
            for bx, by, _ in val_loader:
                bx = bx.to(DEVICE, non_blocking=True)
                by = by.to(DEVICE, non_blocking=True)
                out = model(bx)[:, -1, :]
                vlosses.append(criterion(out, by).item())

        tl, vl = np.mean(losses), np.mean(vlosses)
        dt = time.time() - t0
        print(f"    Epoch {epoch+1:02d}/{EPOCHS} | Train: {tl:.4f} | Val: {vl:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | {dt:.1f}s")

        if vl < best_val:
            best_val = vl
            os.makedirs('models_saved', exist_ok=True)
            torch.save(model.state_dict(), model_path)
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= 5:
                print(f"    Early stop!")
                break

    # === EVALUATION ===
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    preds_all, trues_all = [], []
    with torch.no_grad():
        for bx, by, sids in test_loader:
            bx = bx.to(DEVICE, non_blocking=True)
            p = model(bx)[:, -1, :].cpu().numpy()
            t = by.numpy()
            for i in range(len(sids)):
                sid = sids[i].item()
                preds_all.append(inverse_pm25(p[i], sid)[0])
                trues_all.append(inverse_pm25(t[i], sid)[0])

    y_true, y_pred = np.array(trues_all), np.array(preds_all)
    rmse, mae, r2, mape = get_metrics(y_true, y_pred)
    print(f"    [{region_name.upper()}] T+{horizon_h:2d} | RMSE={rmse:.2f} | MAE={mae:.2f} | R2={r2*100:.2f}% | MAPE={mape:.2f}%")

    return {'region': region_name, 'horizon': f'T+{horizon_h}',
            'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape,
            'n_test': len(y_true)}


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def run_xlinear():
    print("=" * 70)
    print(f"  XLinear Pipeline — 12 Stations | Block: {BLOCK}")
    print("=" * 70)

    all_results = []
    for horizon_h in HORIZONS:
        print(f"\n{'='*60}")
        print(f"  HORIZON T+{horizon_h}h")
        print(f"{'='*60}")

        for region_name, region_sids in REGIONS.items():
            data, num_features = prepare_region_data(region_name, region_sids, horizon_h)
            result = train_region(region_name, region_sids, data, num_features, horizon_h)
            if result:
                all_results.append(result)

    print("\n" + "=" * 70)
    print("  FINAL BENCHMARK — XLinear")
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
    run_xlinear()
