"""
models/Ensemble/pipeline.py

Ensemble: α × XLinear + (1-α) × ESTGCN
- Load saved models, get val+test predictions
- Grid search α on val set (per horizon × region)
- Report ensemble metrics on test set
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════
SELECTED_STATIONS = [1, 4, 5, 16, 17, 27, 7, 18, 24, 30, 31, 32]
REGIONS = {
    'north': [1, 4, 5, 16, 17, 27],
    'south': [7, 18, 24, 30, 31, 32],
}
PM25_COL = 'pm25'
BLOCK    = 'block7'
DATA_DIR = f'data/split/{BLOCK}'
INFO_PATH = 'data/info.csv'
HORIZONS = [1, 3, 6, 12, 24]
SEQ_LEN  = 48
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ══════════════════════════════════════════════════════════════════════════
# IMPORTS — Shared & models
# ══════════════════════════════════════════════════════════════════════════
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from metrics import inverse_pm25, get_metrics
from graph_builder import get_base_matrices

# Import ESTGCN
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ESTGCN'))
from model import ESTGCN

# Import XLinear
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'XLinear'))
from XLinear import Model as XLinearModel

print(f"[*] Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════
# ESTGCN PREDICTION HELPERS
# ══════════════════════════════════════════════════════════════════════════

ESTGCN_EXCLUDE = ['timestamp', 'split', 'station_id', 'province', 'district',
                  'is_frozen', 'is_pm25_sensor_error', 'is_weekend_holiday']

def estgcn_adjacency(sids):
    dist_km, _ = get_base_matrices(INFO_PATH, sids)
    sigma2 = dist_km[dist_km > 0].std() ** 2 if (dist_km > 0).any() else 1.0
    A = np.exp(-(dist_km ** 2) / (sigma2 + 1e-8))
    np.fill_diagonal(A, 1.0)
    row_sum = A.sum(axis=1, keepdims=True)
    return A / (row_sum + 1e-8)


def estgcn_load_split(sids, split_name):
    station_data = {}
    for sid in sids:
        df = pd.read_csv(os.path.join(DATA_DIR, f'station_{sid}.csv'))
        df_s = df[df['split'] == split_name].reset_index(drop=True)
        feat_cols = [c for c in df_s.columns if c not in ESTGCN_EXCLUDE and c != PM25_COL
                     and df_s[c].dtype in ['float64', 'float32', 'int64', 'int32']]
        station_data[sid] = {
            'features': df_s[feat_cols].fillna(0).values.astype(np.float32),
            'pm25': df_s[PM25_COL].fillna(0).values.astype(np.float32),
        }
    return station_data


class ESTGCNDataset(torch.utils.data.Dataset):
    def __init__(self, station_data, seq_len=48, pred_len=24):
        self.seq_len, self.pred_len = seq_len, pred_len
        min_len = min(len(v['features']) for v in station_data.values())
        self.features = np.stack([v['features'][:min_len] for v in station_data.values()], axis=1)
        self.targets = np.stack([v['pm25'][:min_len] for v in station_data.values()], axis=1)
        self.valid_len = min_len - seq_len - pred_len + 1

    def __len__(self): return max(0, self.valid_len)

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_len]
        y = self.targets[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def get_estgcn_predictions(sids, r_name, split_name):
    """Get ESTGCN predictions for a split. Returns (preds_inv, trues_inv) per horizon."""
    data = estgcn_load_split(sids, split_name)
    ds = ESTGCNDataset(data, SEQ_LEN, 24)
    dl = DataLoader(ds, 256, shuffle=False, pin_memory=True)

    adj_np = estgcn_adjacency(sids)
    adj = torch.tensor(adj_np, dtype=torch.float32).to(DEVICE)

    num_features = ds.features.shape[2]
    model = ESTGCN(num_nodes=len(sids), num_features=num_features,
                   seq_len=SEQ_LEN, pred_len=24)
    model.to(DEVICE)

    save_path = f"models_saved/estgcn_baseline_{r_name}.pth"
    model.load_state_dict(torch.load(save_path, weights_only=True, map_location=DEVICE))
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(DEVICE)
            out = model(x, adj)
            all_preds.append(out.cpu().numpy())
            all_targets.append(y.numpy())

    all_preds = np.concatenate(all_preds, axis=0)    # (samples, 24, nodes)
    all_targets = np.concatenate(all_targets, axis=0)

    # Inverse per station
    preds_inv = np.zeros_like(all_preds)
    trues_inv = np.zeros_like(all_targets)
    for node_idx, sid in enumerate(sids):
        for t in range(24):
            trues_inv[:, t, node_idx] = inverse_pm25(all_targets[:, t, node_idx], sid)
            preds_inv[:, t, node_idx] = inverse_pm25(all_preds[:, t, node_idx], sid)

    # Return per-horizon flattened
    result = {}
    for h in HORIZONS:
        result[h] = {
            'pred': preds_inv[:, h-1, :].flatten(),
            'true': trues_inv[:, h-1, :].flatten(),
        }
    return result


# ══════════════════════════════════════════════════════════════════════════
# XLINEAR PREDICTION HELPERS
# ══════════════════════════════════════════════════════════════════════════

SID_TO_IDX = {sid: i for i, sid in enumerate(SELECTED_STATIONS)}
DROP_FEATURES = ['is_frozen', 'is_pm25_sensor_error', 'is_weekend_holiday',
                 'station_id', 'province', 'district', 'timestamp', 'split']
PRECURSOR_COLS = ['pm10', 'no2', 'so2', 'co', 'o3']
PRECURSOR_LAGS = [1, 3, 6, 12, 24]
FUTURE_WEATHER_COLS = ['temp', 'wind_spd', 'precip', 'rh']
NEIGHBOR_LAGS = [1, 3, 6, 12, 24]
CORR_THRESHOLD = 0.05


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def xlinear_config(num_features, region_name='north'):
    cfg = DotDict()
    cfg.seq_len = SEQ_LEN; cfg.pred_len = 1; cfg.enc_in = num_features
    cfg.usenorm = False; cfg.embed_dropout = 0.1; cfg.head_dropout = 0.1
    cfg.t_dropout = 0.1; cfg.c_dropout = 0.1; cfg.features = 'MS'
    if region_name == 'south':
        cfg.d_model = 256; cfg.t_ff = 512; cfg.c_ff = 512
    else:
        cfg.d_model = 128; cfg.t_ff = 256; cfg.c_ff = 256
    return cfg


def build_region_knn(region_sids, k=5):
    dist_km, _ = get_base_matrices(INFO_PATH, SELECTED_STATIONS)
    knn, dist_map = {}, {}
    for sid in region_sids:
        i = SID_TO_IDX[sid]
        candidates = [SID_TO_IDX[s] for s in region_sids if s != sid]
        dists = sorted([(c, dist_km[i, c]) for c in candidates], key=lambda x: x[1])
        knn[i] = [c for c, _ in dists[:k]]
        dist_map[i] = {c: d for c, d in dists[:k]}
    return knn, dist_km


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
    return corrs[corrs >= threshold].index.tolist()


class XLinearDataset(torch.utils.data.Dataset):
    def __init__(self, station_features, station_targets, station_ids, seq_len=48, horizon=1):
        self.seq_len, self.horizon = seq_len, horizon
        self.index = []
        self.features, self.targets, self.sids = station_features, station_targets, station_ids
        for s_local, (feat, tgt) in enumerate(zip(station_features, station_targets)):
            max_start = len(feat) - seq_len - horizon
            for i in range(max(0, max_start + 1)):
                t_idx = i + seq_len - 1 + horizon
                if t_idx < len(tgt) and not np.isnan(tgt[t_idx]):
                    self.index.append((s_local, i))

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        s_local, start = self.index[idx]
        x = self.features[s_local][start:start + self.seq_len]
        y = self.targets[s_local][start + self.seq_len - 1 + self.horizon]
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor([y], dtype=torch.float32),
                self.sids[s_local])


def prepare_xlinear_data(region_name, region_sids, horizon_h):
    all_dfs = {sid: pd.read_csv(os.path.join(DATA_DIR, f'station_{sid}.csv'))
               for sid in SELECTED_STATIONS}
    pm25_arrays = {SID_TO_IDX[sid]: all_dfs[sid][PM25_COL].values for sid in SELECTED_STATIONS}
    knn, distances = build_region_knn(region_sids, k=min(5, len(region_sids) - 1))

    sample_df = all_dfs[region_sids[0]]
    splits = sample_df['split'].values.astype('U5')

    result = {'train': {}, 'val': {}, 'test': {}}
    selected_cols = None

    for sid in region_sids:
        df = all_dfs[sid]
        s_idx = SID_TO_IDX[sid]
        base_cols = [c for c in df.columns if c not in DROP_FEATURES and c != PM25_COL]
        feat_df = df[base_cols].fillna(0.0).copy()

        if selected_cols is None:
            train_mask = splits == 'train'
            feat_train = feat_df.iloc[np.where(train_mask)[0]]
            target_train = df[PM25_COL].iloc[np.where(train_mask)[0]]
            selected_cols = select_features(feat_train, target_train)

        valid_cols = [c for c in selected_cols if c in feat_df.columns]
        features = feat_df[valid_cols].values.astype(np.float32)
        pm25_arr = df[PM25_COL].values.reshape(-1, 1).astype(np.float32)
        features = np.concatenate([features, pm25_arr], axis=1)
        features = np.nan_to_num(features, nan=0.0)
        target = df[PM25_COL].values.astype(np.float32)

        for sn in ['train', 'val', 'test']:
            mask = splits == sn
            result[sn][sid] = (features[mask], target[mask])

    num_features = features.shape[1]
    return result, num_features


def get_xlinear_predictions(region_sids, r_name, horizon_h, split_name):
    """Get XLinear predictions for a split+horizon. Returns (preds_inv, trues_inv)."""
    data, num_features = prepare_xlinear_data(r_name, region_sids, horizon_h)

    ds = XLinearDataset(
        [data[split_name][s][0] for s in region_sids],
        [data[split_name][s][1] for s in region_sids],
        region_sids, SEQ_LEN, horizon_h)
    dl = DataLoader(ds, 128, shuffle=False, pin_memory=True)

    cfg = xlinear_config(num_features, r_name)
    model = XLinearModel(cfg).to(DEVICE)

    save_path = f"models_saved/xlinear_{r_name}_t{horizon_h}.pth"
    model.load_state_dict(torch.load(save_path, weights_only=True, map_location=DEVICE))
    model.eval()

    preds_all, trues_all = [], []
    with torch.no_grad():
        for bx, by, sids_batch in dl:
            bx = bx.to(DEVICE)
            p = model(bx)[:, -1, :].cpu().numpy()
            t = by.numpy()
            for i in range(len(sids_batch)):
                sid = sids_batch[i].item()
                preds_all.append(inverse_pm25(p[i], sid)[0])
                trues_all.append(inverse_pm25(t[i], sid)[0])

    return {'pred': np.array(preds_all), 'true': np.array(trues_all)}


# ══════════════════════════════════════════════════════════════════════════
# ENSEMBLE: GRID SEARCH α
# ══════════════════════════════════════════════════════════════════════════

def find_best_alpha(val_xlinear, val_estgcn, val_true, metric='mae'):
    """Grid search α in [0,1] to minimize MAE on val set."""
    best_alpha, best_score = 0.5, float('inf')
    for a in np.arange(0.0, 1.01, 0.01):
        ensemble = a * val_xlinear + (1 - a) * val_estgcn
        if metric == 'mae':
            score = np.mean(np.abs(ensemble - val_true))
        else:  # rmse
            score = np.sqrt(np.mean((ensemble - val_true) ** 2))
        if score < best_score:
            best_score = score
            best_alpha = a
    return best_alpha, best_score


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def run():
    print("=" * 70)
    print(f"  Ensemble Pipeline — α×XLinear + (1-α)×ESTGCN | Block: {BLOCK}")
    print(f"  Horizons: {HORIZONS}")
    print("=" * 70)

    all_results = []

    for r_name, sids in REGIONS.items():
        print(f"\n{'='*60}")
        print(f"  [{r_name.upper()}] — {len(sids)} stations")
        print(f"{'='*60}")

        # 1. Get ESTGCN predictions (val + test) — one model for all horizons
        print("\n  Loading ESTGCN predictions...")
        estgcn_val = get_estgcn_predictions(sids, r_name, 'val')
        estgcn_test = get_estgcn_predictions(sids, r_name, 'test')
        print(f"    ESTGCN val samples: {len(estgcn_val[1]['true'])}, test: {len(estgcn_test[1]['true'])}")

        for h in HORIZONS:
            print(f"\n  --- T+{h}h ---")

            # 2. Get XLinear predictions (val + test) — per horizon
            print(f"    Loading XLinear T+{h}h predictions...")
            xlinear_val = get_xlinear_predictions(sids, r_name, h, 'val')
            xlinear_test = get_xlinear_predictions(sids, r_name, h, 'test')

            # 3. Align predictions (use min length — different sample counts possible)
            n_val = min(len(xlinear_val['pred']), len(estgcn_val[h]['pred']))
            n_test = min(len(xlinear_test['pred']), len(estgcn_test[h]['pred']))

            xl_val = xlinear_val['pred'][:n_val]
            es_val = estgcn_val[h]['pred'][:n_val]
            true_val = xlinear_val['true'][:n_val]

            xl_test = xlinear_test['pred'][:n_test]
            es_test = estgcn_test[h]['pred'][:n_test]
            true_test = xlinear_test['true'][:n_test]

            # 4. Sanity: individual model metrics
            _, mae_xl, r2_xl, _ = get_metrics(true_test, xl_test)
            _, mae_es, r2_es, _ = get_metrics(true_test, es_test)
            print(f"    XLinear  alone: MAE={mae_xl:.2f}, R²={r2_xl*100:.2f}%")
            print(f"    ESTGCN   alone: MAE={mae_es:.2f}, R²={r2_es*100:.2f}%")

            # 5. Grid search α on val
            best_alpha, val_score = find_best_alpha(xl_val, es_val, true_val, metric='mae')
            print(f"    Optimal α={best_alpha:.2f} (val MAE={val_score:.2f})")

            # 6. Ensemble on test
            ensemble_pred = best_alpha * xl_test + (1 - best_alpha) * es_test
            rmse, mae, r2, mape = get_metrics(true_test, ensemble_pred)
            print(f"    ✅ ENSEMBLE: MAE={mae:.2f}, R²={r2*100:.2f}%, α={best_alpha:.2f}")

            # vs best individual
            better_than = []
            if mae <= mae_xl: better_than.append('XLinear')
            if mae <= mae_es: better_than.append('ESTGCN')
            if better_than:
                print(f"    → Beats: {', '.join(better_than)}")

            all_results.append({
                'region': r_name, 'horizon': f'T+{h}',
                'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape,
                'alpha': best_alpha, 'n_test': n_test,
                'MAE_xl': mae_xl, 'R2_xl': r2_xl,
                'MAE_es': mae_es, 'R2_es': r2_es,
            })

    # ══════════════════════════════════════════════════════════════════════
    # FINAL BENCHMARK
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  FINAL BENCHMARK — Ensemble (α×XLinear + (1-α)×ESTGCN)")
    print("=" * 70)
    print(f"{'Region':<10} {'Horizon':<8} {'α':<6} {'MAE':<8} {'R2 %':<8} {'MAE_XL':<8} {'MAE_ES':<8} {'RMSE':<8}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['region']:<10} {r['horizon']:<8} {r['alpha']:<6.2f} "
              f"{r['MAE']:<8.2f} {r['R2']*100:<7.2f}% "
              f"{r['MAE_xl']:<8.2f} {r['MAE_es']:<8.2f} {r['RMSE']:<8.2f}")

    print("\n" + "-" * 70)
    print("AGGREGATED (weighted by test size):")
    for h in HORIZONS:
        hr = [r for r in all_results if r['horizon'] == f'T+{h}']
        if hr:
            total = sum(r['n_test'] for r in hr)
            w = lambda key: sum(r[key]*r['n_test'] for r in hr) / total
            avg_alpha = sum(r['alpha']*r['n_test'] for r in hr) / total
            print(f"  T+{h:<3d}  α={avg_alpha:.2f}  "
                  f"RMSE={w('RMSE'):.2f}  MAE={w('MAE'):.2f}  "
                  f"R²={w('R2')*100:.2f}%  MAPE={w('MAPE'):.2f}%")
    print("=" * 70)


if __name__ == '__main__':
    run()
