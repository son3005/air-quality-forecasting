"""
S09 XLinear Clean Pipeline (v2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Dataset-driven improvements on XLinear original architecture.

Fixes v2:
  D1. Region-based training (North+Central / South) + Asymmetric Huber Loss
  D2. Chronological split (Train:2023-2024, Val:2025-01~04, Test:2025-05~12)
  D3. Feature cleaning (remove useless binary/duplicate features)
  D4. Feature enrichment (Future Weather, Precursor Lags, Neighbor Lagged Stats)
  D5. XLinear original model (d_model=256 for South)
  D6. Force-keep future weather features in selection
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
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.append('models/S09_XLinear_Clean')
from graph import get_base_matrices

# Import XLinear original
sys.path.append('tmp_xlinear')
from models.XLinear import Model as XLinearModel

# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════
DATA_DIR   = 'data/normalized'
SCALER_DIR = DATA_DIR
INFO_PATH  = 'data/info.csv'

SELECTED_STATIONS = [1, 3, 4, 7, 9, 12, 13, 15, 16, 17, 18, 24, 27, 29, 31, 32]
PM25_COL = 'pm25'

# D1: Region clusters — Central (13,15) merged into North
REGIONS = {
    'north':   [1, 3, 4, 13, 15, 16, 17, 27, 29],
    'south':   [7, 9, 12, 18, 24, 31, 32],
}

# Station index mapping: station_id -> position in SELECTED_STATIONS
SID_TO_IDX = {sid: i for i, sid in enumerate(SELECTED_STATIONS)}

# D3: Features to DROP (binary noise + constant)
DROP_FEATURES = [
    'is_frozen', 'is_pm25_sensor_error', 'is_weekend_holiday',
    'station_id', 'province', 'district',
    'timestamp', 'split',
]

# D4: Precursor columns for lag features
PRECURSOR_COLS = ['pm10', 'no2', 'so2', 'co', 'o3']
PRECURSOR_LAGS = [1, 3, 6, 12, 24]

# D4: Future weather columns
FUTURE_WEATHER_COLS = ['temp', 'wind_spd', 'precip', 'rh']

# D4: Neighbor lags
NEIGHBOR_LAGS = [1, 3, 6, 12, 24]

SEQ_LEN    = 48
HORIZONS   = [1, 3, 6, 12, 24]
BATCH_SIZE = 128
EPOCHS     = 20
LR         = 5e-4
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CORR_THRESHOLD = 0.05  # D3/D5: drop features with |r| < this

print(f"[*] Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════
# D1: ASYMMETRIC HUBER LOSS
# ══════════════════════════════════════════════════════════════════════════

class AsymmetricHuberLoss(nn.Module):
    """Penalize under-prediction of high PM2.5 more heavily."""
    def __init__(self, delta=1.0, alpha=2.0):
        super().__init__()
        self.delta = delta
        self.alpha = alpha  # penalty multiplier for under-prediction
    
    def forward(self, pred, true):
        error = true - pred  # positive = under-prediction
        abs_error = torch.abs(error)
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        base_loss = 0.5 * quadratic**2 + self.delta * linear
        weight = torch.where(error > 0, self.alpha, 1.0)
        return (weight * base_loss).mean()


# ══════════════════════════════════════════════════════════════════════════
# D1: REGION-SCOPED KNN
# ══════════════════════════════════════════════════════════════════════════

def build_region_knn(region_sids, k=5):
    """Build KNN indices ONLY within the same region."""
    distances, _ = get_base_matrices(info_path=INFO_PATH)
    
    region_idxs = [SID_TO_IDX[sid] for sid in region_sids]
    knn = {}
    dist_map = {}
    
    for sid in region_sids:
        i = SID_TO_IDX[sid]
        # Only consider neighbors within same region
        candidates = [SID_TO_IDX[s] for s in region_sids if s != sid]
        if not candidates:
            knn[i] = []
            dist_map[i] = {}
            continue
        
        dists = [(c, distances[i, c]) for c in candidates]
        dists.sort(key=lambda x: x[1])
        neighbors = [c for c, _ in dists[:k]]
        knn[i] = neighbors
        dist_map[i] = {c: d for c, d in dists[:k]}
    
    return knn, distances


# ══════════════════════════════════════════════════════════════════════════
# D2: CHRONOLOGICAL TEMPORAL SPLIT
# ══════════════════════════════════════════════════════════════════════════

def create_chronological_split(df):
    """
    Chronological split to avoid temporal leakage:
      Train: 2023-01-01 → 2024-12-31
      Val:   2025-01-01 → 2025-04-30
      Test:  2025-05-01 → 2025-12-01
    Returns numpy array of 'train'/'val'/'test' labels.
    """
    ts = pd.to_datetime(df['timestamp'])
    
    splits = np.full(len(df), 'test', dtype='U5')
    splits[ts < '2025-01-01'] = 'train'
    splits[(ts >= '2025-01-01') & (ts < '2025-05-01')] = 'val'
    # test = rest (2025-05-01 onwards)
    
    n_tr = (splits == 'train').sum()
    n_va = (splits == 'val').sum()
    n_te = (splits == 'test').sum()
    print(f"    [D2] Chronological split: train={n_tr}, val={n_va}, test={n_te}")
    return splits


# ══════════════════════════════════════════════════════════════════════════
# D4: FEATURE ENRICHMENT
# ══════════════════════════════════════════════════════════════════════════

def compute_precursor_lags(df):
    """D4: Create lag features for precursor pollutants."""
    feats = {}
    for col in PRECURSOR_COLS:
        if col in df.columns:
            for lag in PRECURSOR_LAGS:
                feats[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return pd.DataFrame(feats, index=df.index)


def compute_future_weather(df, horizon_h):
    """D4: Future weather at T+h (shift by -h). Known in forecasting context."""
    feats = {}
    for col in FUTURE_WEATHER_COLS:
        if col in df.columns:
            feats[f'{col}_fut_h{horizon_h}'] = df[col].shift(-horizon_h)
    return pd.DataFrame(feats, index=df.index)


def compute_neighbor_lagged_stats(s_idx, pm25_arrays, knn_indices, distances_matrix):
    """
    D4: Compute neighbor PM2.5 stats at multiple lags, with distance weighting.
    Returns DataFrame with ~23 features.
    """
    neighbor_idxs = knn_indices.get(s_idx, [])
    if not neighbor_idxs:
        return pd.DataFrame()
    
    neighbor_pm25 = [pm25_arrays[n] for n in neighbor_idxs]
    neighbor_dists = [distances_matrix[s_idx, n] for n in neighbor_idxs]
    
    stacked = np.column_stack(neighbor_pm25)
    
    # Inverse distance weights
    inv_dists = np.array([1.0 / (d + 1e-3) for d in neighbor_dists])
    inv_dists_norm = inv_dists / inv_dists.sum()
    
    result = {}
    for lag in NEIGHBOR_LAGS:
        lagged = pd.DataFrame(stacked).shift(lag).values
        result[f'nbr_mean_lag{lag}'] = np.nanmean(lagged, axis=1)
        result[f'nbr_max_lag{lag}'] = np.nanmax(lagged, axis=1)
        result[f'nbr_std_lag{lag}'] = np.nanstd(lagged, axis=1)
        # Distance-weighted mean
        result[f'nbr_wmean_lag{lag}'] = (lagged * inv_dists_norm).sum(axis=1)
    
    # Nearest neighbor raw value
    if len(neighbor_pm25) > 0:
        nearest = pd.Series(neighbor_pm25[0])
        result['nbr_nearest_lag1'] = nearest.shift(1).values
        result['nbr_nearest_lag6'] = nearest.shift(6).values
        result['nbr_nearest_lag24'] = nearest.shift(24).values
    
    return pd.DataFrame(result)


# ══════════════════════════════════════════════════════════════════════════
# D3/D5: FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════

def select_features(feat_df, target_series, threshold=CORR_THRESHOLD):
    """Remove features with |correlation| < threshold with target."""
    num_cols = feat_df.select_dtypes(include='number').columns
    corrs = feat_df[num_cols].corrwith(target_series).abs()
    keep = corrs[corrs >= threshold].index.tolist()
    dropped = corrs[corrs < threshold].index.tolist()
    if dropped:
        print(f"    [D5] Dropped {len(dropped)} low-corr features: {dropped[:5]}{'...' if len(dropped)>5 else ''}")
    return keep


# ══════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════

class S09Dataset(Dataset):
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

def inverse_pm25(y_norm, station_id, scaler_dir=SCALER_DIR):
    scaler_path = os.path.join(scaler_dir, f'scalers_{station_id}.pkl')
    if not os.path.exists(scaler_path): return y_norm
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    method_tuple = scalers.get('pm25')
    if not method_tuple: return y_norm
    method, sc = method_tuple
    y_inv = sc.inverse_transform(np.array(y_norm).reshape(-1, 1)).flatten()
    if 'log1p' in method: y_inv = np.expm1(y_inv)
    return y_inv

def compute_mape(y_true, y_pred):
    mask = y_true > 1.0
    if mask.sum() == 0: return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def get_metrics(y_true, y_pred):
    return (np.sqrt(mean_squared_error(y_true, y_pred)),
            mean_absolute_error(y_true, y_pred),
            r2_score(y_true, y_pred),
            compute_mape(y_true, y_pred))

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
    
    # D5: Larger capacity for South (complex distribution, skew=1.6)
    if region_name == 'south':
        cfg.d_model = 256
        cfg.t_ff = 512
        cfg.c_ff = 512
    else:
        cfg.d_model = 128
        cfg.t_ff = 256
        cfg.c_ff = 256
    return cfg


# ══════════════════════════════════════════════════════════════════════════
# MAIN: PREPARE DATA PER REGION + HORIZON
# ══════════════════════════════════════════════════════════════════════════

def prepare_region_data(region_name, region_sids, horizon_h):
    """
    Build feature matrices for one region at one horizon.
    Applies all D1-D5 improvements.
    """
    print(f"\n  [{region_name.upper()}] Preparing features for T+{horizon_h}h ({len(region_sids)} stations)...")
    
    # Load all station data
    all_dfs = {}
    for sid in SELECTED_STATIONS:  # Load all for neighbor computation
        all_dfs[sid] = pd.read_csv(os.path.join(DATA_DIR, f'norm_station_{sid}.csv'))
    
    # PM2.5 arrays for KNN (all stations, but KNN limited to region)
    pm25_arrays = {}
    for s_idx, sid in enumerate(SELECTED_STATIONS):
        pm25_arrays[s_idx] = all_dfs[sid][PM25_COL].values
    
    # D1: Region-scoped KNN
    knn, distances = build_region_knn(region_sids, k=min(5, len(region_sids) - 1))
    
    # D2: Chronological split (same for all stations)
    sample_df = all_dfs[region_sids[0]]
    strat_splits = create_chronological_split(sample_df)
    
    result = {'train': {}, 'val': {}, 'test': {}}
    selected_cols = None  # will be set by feature selection on first station
    
    for sid in region_sids:
        df = all_dfs[sid]
        s_idx = SID_TO_IDX[sid]
        T = len(df)
        
        # D2: Override split
        splits = strat_splits
        
        # === BASE FEATURES (D3: drop useless) ===
        base_cols = [c for c in df.columns if c not in DROP_FEATURES and c != PM25_COL]
        base_feats = df[base_cols].copy()
        
        # D3: Remove duplicate time encodings
        # Data already has hour_sin, hour_cos, month_sin, month_cos
        # Don't add new ones — just keep what exists
        
        # === D4: PRECURSOR LAGS ===
        precursor_df = compute_precursor_lags(df)
        
        # === D4: FUTURE WEATHER ===
        future_df = compute_future_weather(df, horizon_h)
        
        # === D4: NEIGHBOR LAGGED STATS ===
        nbr_df = compute_neighbor_lagged_stats(s_idx, pm25_arrays, knn, distances)
        
        # === COMBINE ALL FEATURES ===
        feat_df = pd.concat([base_feats, precursor_df, future_df, nbr_df], axis=1)
        feat_df = feat_df.fillna(0.0)
        
        # === D5: FEATURE SELECTION (on train split, first station) ===
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
                print(f"    [D5] WARNING: train too small ({n_train}), keeping all features")
            # D6: Force-keep critical features even if low corr
            must_keep = [
                'pm25_lag_1', 'pm25_roll_mean_6', 'pm25_lag_3',
                'pm25_lag_6', 'pm25_lag_12', 'pm25_lag_24',
            ]
            # Force-keep ALL future weather features (known at forecast time)
            for col in feat_df.columns:
                if '_fut_h' in col:
                    must_keep.append(col)
            for mk in must_keep:
                if mk in feat_df.columns and mk not in selected_cols:
                    selected_cols.append(mk)
            print(f"    [D5] Selected {len(selected_cols)} features (from {len(feat_df.columns)})")
            print(f"    [D5] Top features: {selected_cols[:10]}...")
        
        # Apply selection + add PM2.5 as LAST column (endogenous)
        valid_cols = [c for c in selected_cols if c in feat_df.columns]
        features = feat_df[valid_cols].values.astype(np.float32)
        pm25_arr = df[PM25_COL].values.reshape(-1, 1).astype(np.float32)
        features = np.concatenate([features, pm25_arr], axis=1)
        features = np.nan_to_num(features, nan=0.0)
        
        target = df[PM25_COL].values.astype(np.float32)
        
        # Split
        for split_name in ['train', 'val', 'test']:
            mask = splits == split_name
            result[split_name][sid] = (features[mask], target[mask])
    
    num_features = features.shape[1]
    print(f"    Total features: {num_features} (selected + PM2.5 endogenous)")
    
    # Print split balance
    for sn in ['train', 'val', 'test']:
        total = sum(len(result[sn][s][0]) for s in region_sids)
        print(f"    {sn}: {total:,} rows")
    
    return result, num_features


# ══════════════════════════════════════════════════════════════════════════
# TRAIN + EVALUATE ONE REGION
# ══════════════════════════════════════════════════════════════════════════

def train_region(region_name, region_sids, data, num_features, horizon_h):
    """Train and evaluate XLinear for one region at one horizon."""
    
    train_ds = S09Dataset(
        [data['train'][s][0] for s in region_sids],
        [data['train'][s][1] for s in region_sids],
        region_sids, SEQ_LEN, horizon_h)
    val_ds = S09Dataset(
        [data['val'][s][0] for s in region_sids],
        [data['val'][s][1] for s in region_sids],
        region_sids, SEQ_LEN, horizon_h)
    test_ds = S09Dataset(
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
    
    # XLinear original model
    cfg = get_xlinear_config(num_features, region_name=region_name)
    model = XLinearModel(cfg).to(DEVICE)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Model params: {n_params:,} | Features: {num_features}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = AsymmetricHuberLoss(delta=1.0, alpha=2.0)
    
    best_val = float('inf')
    patience_cnt = 0
    model_path = f'models_saved/s09_{region_name}_t{horizon_h}.pth'
    
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

def run_s09():
    print("=" * 70)
    print("  S09: Dataset-Driven XLinear Pipeline")
    print("  D1: Region-based   D2: Stratified split   D3: Feature clean")
    print("  D4: Feature enrich D5: XLinear original   D6: Asymmetric loss")
    print("=" * 70)
    
    all_results = []
    
    for horizon_h in HORIZONS:
        print(f"\n{'='*60}")
        print(f"  HORIZON T+{horizon_h}h")
        print(f"{'='*60}")
        
        horizon_preds_all = []
        horizon_trues_all = []
        
        for region_name, region_sids in REGIONS.items():
            data, num_features = prepare_region_data(region_name, region_sids, horizon_h)
            result = train_region(region_name, region_sids, data, num_features, horizon_h)
            if result:
                all_results.append(result)
        
        # Aggregate across regions for this horizon
        horizon_results = [r for r in all_results if r['horizon'] == f'T+{horizon_h}']
        if horizon_results:
            total_test = sum(r['n_test'] for r in horizon_results)
            # Weighted average by n_test
            avg_r2 = sum(r['R2'] * r['n_test'] for r in horizon_results) / total_test
            avg_rmse = sum(r['RMSE'] * r['n_test'] for r in horizon_results) / total_test
            avg_mae = sum(r['MAE'] * r['n_test'] for r in horizon_results) / total_test
            avg_mape = sum(r['MAPE'] * r['n_test'] for r in horizon_results) / total_test
            print(f"\n  >>> COMBINED T+{horizon_h} | RMSE={avg_rmse:.2f} | MAE={avg_mae:.2f} | R2={avg_r2*100:.2f}% | MAPE={avg_mape:.2f}%")
    
    # Final summary
    print("\n" + "=" * 70)
    print("  FINAL BENCHMARK -- S09 (Dataset-Driven XLinear)")
    print("=" * 70)
    print(f"{'Region':<10} {'Horizon':<8} {'RMSE':>8} {'MAE':>8} {'R2 %':>8} {'MAPE %':>10}")
    print("-" * 55)
    for r in all_results:
        print(f"{r['region']:<10} {r['horizon']:<8} {r['RMSE']:>8.2f} {r['MAE']:>8.2f} {r['R2']*100:>7.2f}% {r['MAPE']:>9.2f}%")
    
    # Aggregated per horizon
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
    run_s09()
