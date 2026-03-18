"""
S10 v3: iTransformer Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Replaces XLinear with iTransformer — attention across variables, not time.

Key architecture difference:
  XLinear:       Linear(seq→d) per variable → 1 Gating Block → single head
  iTransformer:  Linear(seq→d) per variable → N Transformer layers (attn across vars) → head

Uses S09 v2 proven approach:
  - Per-horizon training (4 separate models, not multi-output)
  - Region-based (North+Central / South)
  - Chronological split
  - Feature enrichment + cleaning
  - Asymmetric Huber Loss
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from graph import get_base_matrices

# ══════════════════════════════════════════════════════════════════════════
# iTransformer MODEL
# ══════════════════════════════════════════════════════════════════════════

class RevIN(nn.Module):
    """Reversible Instance Normalization — handles distribution shift."""
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            return (x - self._mean) / self._std
        elif mode == 'denorm':
            return x * self._std[:, :, -1:] + self._mean[:, :, -1:]


class iTransformerModel(nn.Module):
    """
    iTransformer: Inverted Transformer for time series forecasting.

    Instead of attention across time steps, applies attention across VARIABLES.
    Each variable's time series becomes a token via embedding.

    Input:  [B, seq_len, C]  (C = num variables, last = PM2.5)
    Output: [B, 1]           (PM2.5 prediction at horizon h)
    """
    def __init__(self, seq_len, pred_len, d_model, n_heads, e_layers,
                 d_ff, dropout, enc_in, use_revin=True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.use_revin = use_revin

        # RevIN for distribution shift
        if use_revin:
            self.revin = RevIN(enc_in)

        # Variable embedding: each variable's seq_len series → d_model
        self.embedding = nn.Linear(seq_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer encoder — attention across variables
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.norm = nn.LayerNorm(d_model)

        # Projection head: d_model → pred_len (only target variable)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, pred_len)
        )

    def forward(self, x):
        """
        x: [B, seq_len, C]
        returns: [B, pred_len, 1]
        """
        # RevIN normalize
        if self.use_revin:
            x = self.revin(x, mode='norm')

        # [B, seq_len, C] → [B, C, seq_len] — each variable is a token
        x = x.permute(0, 2, 1)

        # Embed each variable: [B, C, seq_len] → [B, C, d_model]
        x = self.embed_dropout(self.embedding(x))

        # Transformer: self-attention across C variables
        x = self.encoder(x)  # [B, C, d_model]
        x = self.norm(x)

        # Extract target variable (last = PM2.5): [B, 1, d_model]
        target_repr = x[:, -1:, :]

        # Project to prediction: [B, 1, d_model] → [B, 1, pred_len]
        out = self.head(target_repr)

        # [B, 1, pred_len] → [B, pred_len, 1]
        out = out.permute(0, 2, 1)

        # RevIN denormalize
        if self.use_revin:
            out = self.revin(out, mode='denorm')

        return out


# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════
DATA_DIR   = 'data/normalized'
SCALER_DIR = DATA_DIR
INFO_PATH  = 'data/info.csv'

SELECTED_STATIONS = [1, 3, 4, 7, 9, 12, 13, 15, 16, 17, 18, 24, 27, 29, 31, 32]
PM25_COL = 'pm25'

REGIONS = {
    'north': [1, 3, 4, 13, 15, 16, 17, 27, 29],
    'south': [7, 9, 12, 18, 24, 31, 32],
}

SID_TO_IDX = {sid: i for i, sid in enumerate(SELECTED_STATIONS)}
DROP_FEATURES = [
    'is_frozen', 'is_pm25_sensor_error', 'is_weekend_holiday',
    'station_id', 'province', 'district', 'timestamp', 'split',
]

PRECURSOR_COLS = ['pm10', 'no2', 'so2', 'co', 'o3']
PRECURSOR_LAGS = [1, 3, 6, 12, 24]
FUTURE_WEATHER_COLS = ['temp', 'wind_spd', 'precip', 'rh']
NEIGHBOR_LAGS = [1, 3, 6, 12, 24]

SEQ_LEN    = 48
HORIZONS   = [1, 6, 12, 24]
BATCH_SIZE = 128
EPOCHS     = 25
LR         = 3e-4     # Slightly lower for Transformer
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CORR_THRESHOLD = 0.05

print(f"[*] Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════
# LOSS
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
# DATA PROCESSING (reused from S09)
# ══════════════════════════════════════════════════════════════════════════

def build_region_knn(region_sids, k=5):
    distances, _ = get_base_matrices(info_path=INFO_PATH)
    knn = {}
    for sid in region_sids:
        i = SID_TO_IDX[sid]
        candidates = [SID_TO_IDX[s] for s in region_sids if s != sid]
        if not candidates: knn[i] = []; continue
        dists = [(c, distances[i, c]) for c in candidates]
        dists.sort(key=lambda x: x[1])
        knn[i] = [c for c, _ in dists[:k]]
    return knn, distances

def create_chronological_split(df):
    ts = pd.to_datetime(df['timestamp'])
    splits = np.full(len(df), 'test', dtype='U5')
    splits[ts < '2025-01-01'] = 'train'
    splits[(ts >= '2025-01-01') & (ts < '2025-05-01')] = 'val'
    n_tr = (splits == 'train').sum()
    n_va = (splits == 'val').sum()
    n_te = (splits == 'test').sum()
    print(f"    [Split] train={n_tr}, val={n_va}, test={n_te}")
    return splits

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
    if not neighbor_idxs: return pd.DataFrame()
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

def select_features(feat_df, target_series, threshold=CORR_THRESHOLD):
    num_cols = feat_df.select_dtypes(include='number').columns
    corrs = feat_df[num_cols].corrwith(target_series).abs()
    keep = corrs[corrs >= threshold].index.tolist()
    dropped = corrs[corrs < threshold].index.tolist()
    if dropped:
        print(f"    [FS] Dropped {len(dropped)} low-corr: {dropped[:5]}{'...' if len(dropped)>5 else ''}")
    return keep


# ══════════════════════════════════════════════════════════════════════════
# DATASET (per-horizon, same as S09)
# ══════════════════════════════════════════════════════════════════════════

class S10Dataset(Dataset):
    def __init__(self, station_features, station_targets, station_ids, seq_len, horizon):
        self.seq_len = seq_len
        self.horizon = horizon
        self.index = []
        self.features = station_features
        self.targets = station_targets
        self.sids = station_ids

        for s_local, (feat, tgt) in enumerate(zip(station_features, station_targets)):
            T = len(feat)
            for i in range(T - seq_len - horizon + 1):
                t_idx = i + seq_len - 1 + horizon
                if t_idx < T and not np.isnan(tgt[t_idx]):
                    self.index.append((s_local, i))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        s_local, start = self.index[idx]
        x = self.features[s_local][start:start + self.seq_len]
        y = self.targets[s_local][start + self.seq_len - 1 + self.horizon]
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
                self.sids[s_local])


# ══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def inverse_pm25(y_norm, station_id):
    scaler_path = os.path.join(SCALER_DIR, f'scalers_{station_id}.pkl')
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


def get_itransformer_config(num_features, region_name='north'):
    """iTransformer config per region."""
    if region_name == 'south':
        return dict(
            seq_len=SEQ_LEN, pred_len=1, d_model=256, n_heads=8,
            e_layers=3, d_ff=512, dropout=0.1, enc_in=num_features,
            use_revin=True
        )
    else:
        return dict(
            seq_len=SEQ_LEN, pred_len=1, d_model=128, n_heads=4,
            e_layers=2, d_ff=256, dropout=0.1, enc_in=num_features,
            use_revin=True
        )


# ══════════════════════════════════════════════════════════════════════════
# PREPARE DATA PER REGION + HORIZON (same as S09)
# ══════════════════════════════════════════════════════════════════════════

def prepare_region_data(region_name, region_sids, horizon_h):
    print(f"\n  [{region_name.upper()}] Preparing features for T+{horizon_h}h ({len(region_sids)} stations)...")
    all_dfs = {}
    for sid in SELECTED_STATIONS:
        all_dfs[sid] = pd.read_csv(os.path.join(DATA_DIR, f'norm_station_{sid}.csv'))
    pm25_arrays = {s_idx: all_dfs[sid][PM25_COL].values
                   for s_idx, sid in enumerate(SELECTED_STATIONS)}

    knn, distances = build_region_knn(region_sids, k=min(5, len(region_sids) - 1))
    splits = create_chronological_split(all_dfs[region_sids[0]])

    result = {'train': {}, 'val': {}, 'test': {}}
    selected_cols = None

    for sid in region_sids:
        df = all_dfs[sid]
        s_idx = SID_TO_IDX[sid]
        base_cols = [c for c in df.columns if c not in DROP_FEATURES and c != PM25_COL]
        base_feats = df[base_cols].copy()
        precursor_df = compute_precursor_lags(df)
        future_df = compute_future_weather(df, horizon_h)
        nbr_df = compute_neighbor_lagged_stats(s_idx, pm25_arrays, knn, distances)
        feat_df = pd.concat([base_feats, precursor_df, future_df, nbr_df], axis=1).fillna(0.0)

        if selected_cols is None:
            train_mask = splits == 'train'
            if train_mask.sum() > 100:
                train_idx = np.where(train_mask)[0]
                selected_cols = select_features(feat_df.iloc[train_idx], df[PM25_COL].iloc[train_idx])
            else:
                selected_cols = feat_df.select_dtypes(include='number').columns.tolist()
            must_keep = ['pm25_lag_1', 'pm25_roll_mean_6', 'pm25_lag_3',
                         'pm25_lag_6', 'pm25_lag_12', 'pm25_lag_24']
            for col in feat_df.columns:
                if '_fut_h' in col: must_keep.append(col)
            for mk in must_keep:
                if mk in feat_df.columns and mk not in selected_cols:
                    selected_cols.append(mk)
            print(f"    [FS] Selected {len(selected_cols)} features (from {len(feat_df.columns)})")

        valid_cols = [c for c in selected_cols if c in feat_df.columns]
        features = feat_df[valid_cols].values.astype(np.float32)
        pm25_arr = df[PM25_COL].values.reshape(-1, 1).astype(np.float32)
        features = np.concatenate([features, pm25_arr], axis=1)  # PM2.5 = last col
        features = np.nan_to_num(features, nan=0.0)
        target = df[PM25_COL].values.astype(np.float32)

        for sn in ['train', 'val', 'test']:
            mask = splits == sn
            result[sn][sid] = (features[mask], target[mask])

    num_features = features.shape[1]
    print(f"    Total features: {num_features} (PM2.5 as last channel)")
    for sn in ['train', 'val', 'test']:
        total = sum(len(result[sn][s][0]) for s in region_sids)
        print(f"    {sn}: {total:,} rows")
    return result, num_features


# ══════════════════════════════════════════════════════════════════════════
# TRAIN + EVALUATE (per-horizon, like S09)
# ══════════════════════════════════════════════════════════════════════════

def train_region(region_name, region_sids, data, num_features, horizon_h):
    print(f"\n    Training iTransformer for T+{horizon_h}h...")

    train_ds = S10Dataset(
        [data['train'][s][0] for s in region_sids],
        [data['train'][s][1] for s in region_sids],
        region_sids, SEQ_LEN, horizon_h)
    val_ds = S10Dataset(
        [data['val'][s][0] for s in region_sids],
        [data['val'][s][1] for s in region_sids],
        region_sids, SEQ_LEN, horizon_h)
    test_ds = S10Dataset(
        [data['test'][s][0] for s in region_sids],
        [data['test'][s][1] for s in region_sids],
        region_sids, SEQ_LEN, horizon_h)

    if len(train_ds) == 0 or len(val_ds) == 0:
        print(f"    [SKIP]"); return None

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, pin_memory=True)
    print(f"    Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    cfg = get_itransformer_config(num_features, region_name)
    model = iTransformerModel(**cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    iTransformer: d={cfg['d_model']}, heads={cfg['n_heads']}, layers={cfg['e_layers']}, "
          f"RevIN=ON, params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = AsymmetricHuberLoss(delta=1.0, alpha=2.0)

    best_val = float('inf')
    patience_cnt = 0
    model_path = f'models_saved/s10v3_{region_name}_t{horizon_h}.pth'

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        losses = []
        for bx, by, _ in train_loader:
            bx = bx.to(DEVICE, non_blocking=True)
            by = by.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            out = model(bx).squeeze()  # [B, 1, 1] → [B]
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
                out = model(bx).squeeze()
                vlosses.append(criterion(out, by).item())

        tl, vl = np.mean(losses), np.mean(vlosses)
        dt = time.time() - t0
        print(f"    Epoch {epoch:02d}/{EPOCHS} | Train: {tl:.4f} | Val: {vl:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | {dt:.1f}s")

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

    # === EVALUATE ===
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    all_preds, all_trues, all_sids = [], [], []
    with torch.no_grad():
        for bx, by, sids in test_loader:
            bx = bx.to(DEVICE, non_blocking=True)
            out = model(bx).squeeze().cpu().numpy()
            all_preds.append(out)
            all_trues.append(by.numpy())
            all_sids.extend(sids.numpy().tolist())

    preds = np.concatenate(all_preds)
    trues = np.concatenate(all_trues)
    sids_arr = np.array(all_sids)

    y_true_inv = np.zeros_like(trues)
    y_pred_inv = np.zeros_like(preds)
    for sid in region_sids:
        mask = sids_arr == sid
        if mask.sum() > 0:
            y_true_inv[mask] = inverse_pm25(trues[mask], sid)
            y_pred_inv[mask] = inverse_pm25(preds[mask], sid)

    rmse, mae, r2, mape = get_metrics(y_true_inv, y_pred_inv)
    print(f"    [{region_name.upper()}] T+{horizon_h:2d} | RMSE={rmse:.2f} | MAE={mae:.2f} | "
          f"R2={r2*100:.2f}% | MAPE={mape:.2f}%")
    return {
        'region': region_name, 'horizon': f'T+{horizon_h}',
        'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape,
        'n_test': len(y_true_inv)
    }


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def run_s10():
    print("=" * 70)
    print("  S10 v3: iTransformer + Per-Horizon Training")
    print("  Attention across variables (not time) + RevIN")
    print("  Per-horizon models (proven optimal from S09)")
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

        # Combined for this horizon
        hr = [r for r in all_results if r['horizon'] == f'T+{horizon_h}']
        if len(hr) >= 2:
            total = sum(r['n_test'] for r in hr)
            c_rmse = sum(r['RMSE']*r['n_test'] for r in hr) / total
            c_mae = sum(r['MAE']*r['n_test'] for r in hr) / total
            c_r2 = sum(r['R2']*r['n_test'] for r in hr) / total
            c_mape = sum(r['MAPE']*r['n_test'] for r in hr) / total
            print(f"\n  >>> COMBINED T+{horizon_h} | RMSE={c_rmse:.2f} | MAE={c_mae:.2f} | "
                  f"R2={c_r2*100:.2f}% | MAPE={c_mape:.2f}%")

    # Final summary
    print("\n" + "=" * 70)
    print("  FINAL BENCHMARK — S10 v3 (iTransformer)")
    print("=" * 70)
    print(f"{'Region':<10} {'Horizon':<8} {'RMSE':>8} {'MAE':>8} {'R2 %':>8} {'MAPE %':>10}")
    print("-" * 55)
    for r in all_results:
        print(f"{r['region']:<10} {r['horizon']:<8} {r['RMSE']:>8.2f} {r['MAE']:>8.2f} "
              f"{r['R2']*100:>7.2f}% {r['MAPE']:>9.2f}%")

    print("\n" + "-" * 55)
    print("AGGREGATED:")
    for h in HORIZONS:
        hr = [r for r in all_results if r['horizon'] == f'T+{h}']
        if hr:
            total = sum(r['n_test'] for r in hr)
            agg = lambda k: sum(r[k]*r['n_test'] for r in hr) / total
            print(f"  T+{h:<3d}  RMSE={agg('RMSE'):.2f}  MAE={agg('MAE'):.2f}  "
                  f"R2={agg('R2')*100:.2f}%  MAPE={agg('MAPE'):.2f}%")

    # vs S09 v2 + S07 XGBoost
    print("\n" + "-" * 55)
    print("vs BASELINES (same chronological split):")
    s09 = {1: (9.87, 74.11), 6: (14.39, 45.25), 12: (15.32, 37.94), 24: (15.85, 33.58)}
    s07 = {1: (10.96, 71.23), 6: (15.60, 41.71), 12: (16.15, 37.51), 24: (16.58, 34.24)}
    for h in HORIZONS:
        hr = [r for r in all_results if r['horizon'] == f'T+{h}']
        if hr:
            total = sum(r['n_test'] for r in hr)
            s10r = sum(r['RMSE']*r['n_test'] for r in hr) / total
            s10r2 = sum(r['R2']*r['n_test'] for r in hr) / total * 100
            s09r, s09r2 = s09[h]
            s07r, s07r2 = s07[h]
            print(f"  T+{h:<3d}  iTransformer: RMSE={s10r:.2f} R2={s10r2:.1f}%  |  "
                  f"S09 XLinear: {s09r:.2f}/{s09r2:.1f}%  |  S07 XGB: {s07r:.2f}/{s07r2:.1f}%")
    print("=" * 70)


if __name__ == '__main__':
    run_s10()
