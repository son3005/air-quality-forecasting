"""
S11: GCN + XLinear Hybrid Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Integrates Graph Convolution (spatial learning) into the S09 XLinear pipeline.

Key improvements over S09:
  - GCN layers learn spatial correlation between stations in same region
  - Multi-station input: each sample sees ALL stations in region simultaneously
  - GCN output is concatenated with original features before XLinear

Architecture:
  Input [B, T, N, C] → SpatialGCN [B, T, N, gcn_out]
                      → Extract target station + concat original
                      → XLinear (Gating Block)
                      → PM2.5 prediction [B, 1]
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
HORIZONS   = [1, 3, 6, 12, 24]
BATCH_SIZE = 128
EPOCHS     = 20
LR         = 5e-4
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CORR_THRESHOLD = 0.05

# GCN config
GCN_HIDDEN = 32
GCN_OUT    = 16
ADJ_SIGMA  = 100.0  # km, for Gaussian kernel

print(f"[*] Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════
# GRAPH CONVOLUTION LAYERS
# ══════════════════════════════════════════════════════════════════════════

class GraphConvLayer(nn.Module):
    """Single GCN layer: H' = σ(A_norm @ H @ W + b)"""
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_feat, out_feat))
        self.bias = nn.Parameter(torch.zeros(out_feat))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        # x: [B*T, N, in_feat], adj: [N, N]
        return torch.matmul(adj, torch.matmul(x, self.weight)) + self.bias


class SpatialGCN(nn.Module):
    """Two-layer GCN for spatial mixing across stations."""
    def __init__(self, in_feat, hidden_feat, out_feat, dropout=0.1):
        super().__init__()
        self.gcn1 = GraphConvLayer(in_feat, hidden_feat)
        self.gcn2 = GraphConvLayer(hidden_feat, out_feat)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_feat)

    def forward(self, x, adj):
        # x: [B, T, N, C]
        B, T, N, C = x.shape
        x_flat = x.reshape(B * T, N, C)
        h = F.relu(self.gcn1(x_flat, adj))
        h = self.dropout(h)
        h = self.gcn2(h, adj)
        h = self.norm(h)
        return h.reshape(B, T, N, -1)  # [B, T, N, out_feat]


# ══════════════════════════════════════════════════════════════════════════
# GCN + XLINEAR HYBRID MODEL
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
        cfg.d_model = 256
        cfg.t_ff = 512
        cfg.c_ff = 512
    else:
        cfg.d_model = 128
        cfg.t_ff = 256
        cfg.c_ff = 256
    return cfg


class GCNXLinearModel(nn.Module):
    """
    Hybrid: SpatialGCN → XLinear
    GCN learns spatial correlation, XLinear handles temporal gating.
    """
    def __init__(self, num_features, gcn_hidden, gcn_out, region_name='north'):
        super().__init__()
        self.gcn = SpatialGCN(num_features, gcn_hidden, gcn_out, dropout=0.1)

        # XLinear input = original features + GCN spatial context
        xlinear_in = num_features + gcn_out
        cfg = get_xlinear_config(xlinear_in, region_name)
        self.xlinear = XLinearModel(cfg)

        self.num_features = num_features
        self.gcn_out = gcn_out

    def forward(self, x_multi, adj, target_idx):
        """
        x_multi: [B, T, N, C] — all stations in region
        adj:     [N, N]        — adjacency matrix
        target_idx: [B]        — which station is the target per sample
        Returns: [B, pred_len, 1]
        """
        B, T, N, C = x_multi.shape

        # 1. Spatial GCN — learn inter-station relationships
        h = self.gcn(x_multi, adj)  # [B, T, N, gcn_out]

        # 2. Extract target station — permute to [B, N, T, ?] for gather
        x_perm = x_multi.permute(0, 2, 1, 3)  # [B, N, T, C]
        h_perm = h.permute(0, 2, 1, 3)         # [B, N, T, gcn_out]

        idx = target_idx.view(B, 1, 1, 1)
        target_orig = x_perm.gather(1, idx.expand(-1, -1, T, C)).squeeze(1)      # [B, T, C]
        target_gcn  = h_perm.gather(1, idx.expand(-1, -1, T, self.gcn_out)).squeeze(1)  # [B, T, gcn_out]

        # 3. Concat: [exogenous, spatial_context, PM2.5(endogenous)]
        #    PM2.5 is last col of target_orig — move to end after concat
        pm25 = target_orig[:, :, -1:]      # [B, T, 1]
        exo  = target_orig[:, :, :-1]      # [B, T, C-1]
        combined = torch.cat([exo, target_gcn, pm25], dim=-1)  # [B, T, C + gcn_out]

        # 4. XLinear for temporal gating + prediction
        out = self.xlinear(combined)  # [B, pred_len, 1]
        return out


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
# ADJACENCY MATRIX
# ══════════════════════════════════════════════════════════════════════════

def build_adj_matrix(region_sids, sigma=ADJ_SIGMA):
    """Gaussian-kernel adjacency matrix, symmetric-normalized, region-scoped."""
    distances, _ = get_base_matrices(info_path=INFO_PATH)
    region_global_idxs = [SID_TO_IDX[sid] for sid in region_sids]
    N = len(region_sids)

    sub_dist = np.zeros((N, N), dtype=np.float32)
    for i, gi in enumerate(region_global_idxs):
        for j, gj in enumerate(region_global_idxs):
            sub_dist[i, j] = distances[gi, gj]

    adj = np.exp(-sub_dist**2 / (2 * sigma**2))
    np.fill_diagonal(adj, 1.0)

    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    D = np.sum(adj, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-8))
    adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt

    return torch.tensor(adj_norm, dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (same as S09)
# ══════════════════════════════════════════════════════════════════════════

def build_region_knn(region_sids, k=5):
    distances, _ = get_base_matrices(info_path=INFO_PATH)
    knn = {}
    dist_map = {}
    for sid in region_sids:
        i = SID_TO_IDX[sid]
        candidates = [SID_TO_IDX[s] for s in region_sids if s != sid]
        if not candidates:
            knn[i] = []; dist_map[i] = {}; continue
        dists = [(c, distances[i, c]) for c in candidates]
        dists.sort(key=lambda x: x[1])
        neighbors = [c for c, _ in dists[:k]]
        knn[i] = neighbors
        dist_map[i] = {c: d for c, d in dists[:k]}
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


def select_features(feat_df, target_series, threshold=CORR_THRESHOLD):
    num_cols = feat_df.select_dtypes(include='number').columns
    corrs = feat_df[num_cols].corrwith(target_series).abs()
    keep = corrs[corrs >= threshold].index.tolist()
    dropped = corrs[corrs < threshold].index.tolist()
    if dropped:
        print(f"    [FS] Dropped {len(dropped)} low-corr: {dropped[:5]}{'...' if len(dropped)>5 else ''}")
    return keep


# ══════════════════════════════════════════════════════════════════════════
# MULTI-STATION DATASET
# ══════════════════════════════════════════════════════════════════════════

class S11Dataset(Dataset):
    """
    Multi-station Dataset: each sample contains ALL stations in the region.
    Returns: (x_multi[seq_len, N, C], target_station_idx, target_value, station_id)
    """
    def __init__(self, stacked_features, stacked_targets, station_ids,
                 seq_len=48, horizon=1):
        """
        stacked_features: [T, N, C] numpy array (all stations stacked)
        stacked_targets:  [T, N] numpy array (PM2.5 per station)
        station_ids:      list of station IDs (length N)
        """
        self.features = stacked_features
        self.targets = stacked_targets
        self.station_ids = station_ids
        self.seq_len = seq_len
        self.horizon = horizon

        T, N = stacked_targets.shape
        self.index = []
        for i in range(T - seq_len - horizon + 1):
            t_idx = i + seq_len - 1 + horizon
            if t_idx < T:
                for n in range(N):
                    if not np.isnan(stacked_targets[t_idx, n]):
                        self.index.append((i, n))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        start, n = self.index[idx]
        x = self.features[start:start + self.seq_len]  # [seq_len, N, C]
        y = self.targets[start + self.seq_len - 1 + self.horizon, n]
        return (torch.tensor(x, dtype=torch.float32),
                n,  # target station local index
                torch.tensor(y, dtype=torch.float32),
                self.station_ids[n])


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


# ══════════════════════════════════════════════════════════════════════════
# PREPARE DATA (multi-station stacking)
# ══════════════════════════════════════════════════════════════════════════

def prepare_region_data(region_name, region_sids, horizon_h):
    """Build stacked [T, N, C] tensor for all stations in region."""
    print(f"\n  [{region_name.upper()}] Preparing features for T+{horizon_h}h ({len(region_sids)} stations)...")

    # Load all station data
    all_dfs = {}
    for sid in SELECTED_STATIONS:
        all_dfs[sid] = pd.read_csv(os.path.join(DATA_DIR, f'norm_station_{sid}.csv'))

    # PM2.5 arrays for KNN
    pm25_arrays = {}
    for s_idx, sid in enumerate(SELECTED_STATIONS):
        pm25_arrays[s_idx] = all_dfs[sid][PM25_COL].values

    # Region-scoped KNN
    knn, distances = build_region_knn(region_sids, k=min(5, len(region_sids) - 1))

    # Chronological split
    sample_df = all_dfs[region_sids[0]]
    strat_splits = create_chronological_split(sample_df)

    selected_cols = None
    station_features = {}  # sid → [T, C]
    station_targets = {}   # sid → [T]

    for sid in region_sids:
        df = all_dfs[sid]
        s_idx = SID_TO_IDX[sid]

        # Base features
        base_cols = [c for c in df.columns if c not in DROP_FEATURES and c != PM25_COL]
        base_feats = df[base_cols].copy()

        # Feature enrichment
        precursor_df = compute_precursor_lags(df)
        future_df = compute_future_weather(df, horizon_h)
        nbr_df = compute_neighbor_lagged_stats(s_idx, pm25_arrays, knn, distances)

        feat_df = pd.concat([base_feats, precursor_df, future_df, nbr_df], axis=1).fillna(0.0)

        # Feature selection (on first station's train data)
        if selected_cols is None:
            train_mask = strat_splits == 'train'
            if train_mask.sum() > 100:
                train_idx = np.where(train_mask)[0]
                selected_cols = select_features(feat_df.iloc[train_idx], df[PM25_COL].iloc[train_idx])
            else:
                selected_cols = feat_df.select_dtypes(include='number').columns.tolist()
            # Force-keep critical features
            must_keep = ['pm25_lag_1', 'pm25_roll_mean_6', 'pm25_lag_3',
                         'pm25_lag_6', 'pm25_lag_12', 'pm25_lag_24']
            for col in feat_df.columns:
                if '_fut_h' in col:
                    must_keep.append(col)
            for mk in must_keep:
                if mk in feat_df.columns and mk not in selected_cols:
                    selected_cols.append(mk)
            print(f"    [FS] Selected {len(selected_cols)} features (from {len(feat_df.columns)})")

        # Apply selection + append PM2.5 as last column
        valid_cols = [c for c in selected_cols if c in feat_df.columns]
        features = feat_df[valid_cols].values.astype(np.float32)
        pm25_arr = df[PM25_COL].values.reshape(-1, 1).astype(np.float32)
        features = np.concatenate([features, pm25_arr], axis=1)
        features = np.nan_to_num(features, nan=0.0)
        target = df[PM25_COL].values.astype(np.float32)

        station_features[sid] = features  # [T, C]
        station_targets[sid] = target     # [T]

    num_features = station_features[region_sids[0]].shape[1]
    print(f"    Total features per station: {num_features} (selected + PM2.5)")

    # Stack into [T, N, C] and [T, N] per split
    result = {}
    for split_name in ['train', 'val', 'test']:
        mask = strat_splits == split_name
        stacked_feat = np.stack([station_features[sid][mask] for sid in region_sids], axis=1)  # [T_split, N, C]
        stacked_tgt  = np.stack([station_targets[sid][mask] for sid in region_sids], axis=1)   # [T_split, N]
        result[split_name] = (stacked_feat, stacked_tgt)
        print(f"    {split_name}: {stacked_feat.shape[0]:,} timesteps × {stacked_feat.shape[1]} stations")

    return result, num_features


# ══════════════════════════════════════════════════════════════════════════
# TRAIN + EVALUATE
# ══════════════════════════════════════════════════════════════════════════

def train_region(region_name, region_sids, data, num_features, horizon_h, adj):
    """Train and evaluate GCN+XLinear for one region at one horizon."""

    train_ds = S11Dataset(data['train'][0], data['train'][1], region_sids, SEQ_LEN, horizon_h)
    val_ds   = S11Dataset(data['val'][0],   data['val'][1],   region_sids, SEQ_LEN, horizon_h)
    test_ds  = S11Dataset(data['test'][0],  data['test'][1],  region_sids, SEQ_LEN, horizon_h)

    if len(train_ds) == 0 or len(val_ds) == 0:
        print(f"    [SKIP] Not enough data for {region_name}")
        return None

    def collate_fn(batch):
        xs, target_idxs, ys, sids = zip(*batch)
        return (torch.stack(xs), torch.tensor(target_idxs, dtype=torch.long),
                torch.stack(ys), torch.tensor(sids, dtype=torch.long))

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True,
                              pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, BATCH_SIZE, shuffle=False,
                              pin_memory=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, BATCH_SIZE, shuffle=False,
                              pin_memory=True, collate_fn=collate_fn)

    print(f"    Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    # Build model
    model = GCNXLinearModel(num_features, GCN_HIDDEN, GCN_OUT, region_name).to(DEVICE)
    adj_dev = adj.to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"    GCN+XLinear: gcn={GCN_HIDDEN}→{GCN_OUT}, total params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = AsymmetricHuberLoss(delta=1.0, alpha=2.0)

    best_val = float('inf')
    patience_cnt = 0
    model_path = f'models_saved/s11_{region_name}_t{horizon_h}.pth'

    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        losses = []
        for bx, b_tidx, by, _ in train_loader:
            bx    = bx.to(DEVICE, non_blocking=True)
            b_tidx = b_tidx.to(DEVICE, non_blocking=True)
            by    = by.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            out = model(bx, adj_dev, b_tidx)[:, -1, :]  # [B, 1]
            loss = criterion(out.squeeze(-1), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        model.eval()
        vlosses = []
        with torch.no_grad():
            for bx, b_tidx, by, _ in val_loader:
                bx    = bx.to(DEVICE, non_blocking=True)
                b_tidx = b_tidx.to(DEVICE, non_blocking=True)
                by    = by.to(DEVICE, non_blocking=True)
                out = model(bx, adj_dev, b_tidx)[:, -1, :]
                vlosses.append(criterion(out.squeeze(-1), by).item())

        tl, vl = np.mean(losses), np.mean(vlosses)
        dt = time.time() - t0
        print(f"    Epoch {epoch+1:02d}/{EPOCHS} | Train: {tl:.4f} | Val: {vl:.4f}"
              f" | LR: {scheduler.get_last_lr()[0]:.2e} | {dt:.1f}s")

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
        for bx, b_tidx, by, sids in test_loader:
            bx    = bx.to(DEVICE, non_blocking=True)
            b_tidx = b_tidx.to(DEVICE, non_blocking=True)
            p = model(bx, adj_dev, b_tidx)[:, -1, :].cpu().numpy()  # [B, 1]
            t = by.numpy()
            sids_np = sids.numpy()
            for i in range(len(sids_np)):
                sid = sids_np[i]
                preds_all.append(inverse_pm25(p[i], sid)[0])
                trues_all.append(inverse_pm25(t[i:i+1], sid)[0])

    y_true, y_pred = np.array(trues_all), np.array(preds_all)
    rmse, mae, r2, mape = get_metrics(y_true, y_pred)
    print(f"    [{region_name.upper()}] T+{horizon_h:2d} | RMSE={rmse:.2f} | MAE={mae:.2f}"
          f" | R2={r2*100:.2f}% | MAPE={mape:.2f}%")

    return {'region': region_name, 'horizon': f'T+{horizon_h}',
            'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape,
            'n_test': len(y_true)}


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def run_s11():
    print("=" * 70)
    print("  S11: GCN + XLinear Hybrid Pipeline")
    print("  Spatial GCN → XLinear Temporal Gating → PM2.5 Prediction")
    print("  Per-horizon models | Region-based | Chronological split")
    print("=" * 70)

    all_results = []

    for horizon_h in HORIZONS:
        print(f"\n{'='*60}")
        print(f"  HORIZON T+{horizon_h}h")
        print(f"{'='*60}")

        for region_name, region_sids in REGIONS.items():
            # Build adjacency matrix for this region
            adj = build_adj_matrix(region_sids)
            print(f"    [Graph] Adj matrix: {adj.shape[0]}×{adj.shape[1]} (σ={ADJ_SIGMA}km)")

            data, num_features = prepare_region_data(region_name, region_sids, horizon_h)
            result = train_region(region_name, region_sids, data, num_features, horizon_h, adj)
            if result:
                all_results.append(result)

        # Aggregate for this horizon
        horizon_results = [r for r in all_results if r['horizon'] == f'T+{horizon_h}']
        if horizon_results:
            total_test = sum(r['n_test'] for r in horizon_results)
            avg_rmse = sum(r['RMSE'] * r['n_test'] for r in horizon_results) / total_test
            avg_mae  = sum(r['MAE'] * r['n_test'] for r in horizon_results) / total_test
            avg_r2   = sum(r['R2'] * r['n_test'] for r in horizon_results) / total_test
            avg_mape = sum(r['MAPE'] * r['n_test'] for r in horizon_results) / total_test
            print(f"\n  >>> COMBINED T+{horizon_h} | RMSE={avg_rmse:.2f} | MAE={avg_mae:.2f}"
                  f" | R2={avg_r2*100:.2f}% | MAPE={avg_mape:.2f}%")

    # Final summary
    print("\n" + "=" * 70)
    print("  FINAL BENCHMARK — S11 (GCN + XLinear)")
    print("=" * 70)
    print(f"{'Region':<10} {'Horizon':<8} {'RMSE':>8} {'MAE':>8} {'R2 %':>8} {'MAPE %':>10}")
    print("-" * 55)
    for r in all_results:
        print(f"{r['region']:<10} {r['horizon']:<8} {r['RMSE']:>8.2f} {r['MAE']:>8.2f}"
              f" {r['R2']*100:>7.2f}% {r['MAPE']:>9.2f}%")

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
    run_s11()
