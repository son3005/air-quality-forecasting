"""
S08 V2 Pipeline: XLinear V2 (Fair Benchmark)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Cai tien so voi ban S08 cu:
  - Chia Chronological Split giong het S09.
  - Test test data tu 2025-05-01.
  - Output Metrics chia theo Region: North vs South.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import custom XLinear V2 model
sys.path.append('models/S08_XLinear')
from xlinear_v2_model import XLinearV2

from graph import get_base_matrices

# ── CONFIG ────────────────────────────────────────────────────────────────
DATA_DIR    = 'data/normalized'
SCALER_DIR  = DATA_DIR
INFO_PATH   = 'data/info.csv'

SELECTED_STATIONS = [1, 3, 4, 7, 9, 12, 13, 15, 16, 17, 18, 24, 27, 29, 31, 32]
REGIONS = {
    'north': [1, 3, 4, 13, 15, 16, 17, 27, 29],
    'south': [7, 9, 12, 18, 24, 31, 32],
}

EXCLUDE_COLS = ['timestamp', 'split', 'station_id', 'province', 'district']
PM25_COL = 'pm25'

SEQ_LEN    = 48
HORIZONS   = [1, 3, 6, 12, 24]
BATCH_SIZE = 128
EPOCHS     = 20
LR         = 5e-4
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

USE_EMBEDDINGS = False
EMB_DIM_LIMIT = 16

print(f"[*] Device: {DEVICE}")
print(f"[*] USE_EMBEDDINGS: {USE_EMBEDDINGS}")


def build_knn_indices(k=5):
    distances, _ = get_base_matrices(info_path=INFO_PATH)
    N = distances.shape[0]
    knn = {}
    for i in range(N):
        dist_row = distances[i].copy()
        dist_row[i] = np.inf
        knn[i] = np.argsort(dist_row)[:k].tolist()
    return knn


class S08Dataset(Dataset):
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
        x = self.features[s_local][start : start + self.seq_len]
        y = self.targets[s_local][start + self.seq_len - 1 + self.horizon]
        return (torch.tensor(x, dtype=torch.float32), 
                torch.tensor([y], dtype=torch.float32), 
                self.sids[s_local])


def prepare_features():
    print("\n[1] Loading station data...")
    all_dfs = {}
    for sid in SELECTED_STATIONS:
        all_dfs[sid] = pd.read_csv(os.path.join(DATA_DIR, f'norm_station_{sid}.csv'), parse_dates=['timestamp']).sort_values('timestamp')
    
    pm25_arrays = {}
    for s_idx, sid in enumerate(SELECTED_STATIONS):
        pm25_arrays[s_idx] = all_dfs[sid][PM25_COL].values
    
    print("[2] Computing K=5 nearest neighbors...")
    knn = build_knn_indices(k=5)
    
    emb_cols = []
    df_emb = None
    if USE_EMBEDDINGS:
        print("[3] Loading spatial embeddings (reduced dim)...")
        df_emb = pd.read_csv('data/extracted_embeddings/spatial_embeddings.csv')
        all_emb_cols = [c for c in df_emb.columns if c.startswith('ST_Emb_')]
        emb_cols = all_emb_cols[:EMB_DIM_LIMIT]
        print(f"    Using {len(emb_cols)} embedding dims")
    else:
        print("[3] SKIPPING embeddings")
    
    sample_df = all_dfs[SELECTED_STATIONS[0]]
    raw_cols = [c for c in sample_df.columns if c not in EXCLUDE_COLS]
    
    result = {'train': {}, 'val': {}, 'test': {}}
    
    for s_idx, sid in enumerate(SELECTED_STATIONS):
        df = all_dfs[sid]
        T = len(df)
        
        # Chronological Split
        ts = df['timestamp']
        is_train = ts < '2025-01-01'
        is_val   = (ts >= '2025-01-01') & (ts < '2025-05-01')
        is_test  = ts >= '2025-05-01'
        splits = np.full(T, 'train', dtype='U5')
        splits[is_val] = 'val'
        splits[is_test] = 'test'
        
        raw_feats = df[raw_cols].values
        
        # Neighbor stats
        nbr_idxs = knn[s_idx]
        nbr_stack = np.column_stack([pm25_arrays[n] for n in nbr_idxs])
        nbr_mean = nbr_stack.mean(axis=1, keepdims=True)
        nbr_max  = nbr_stack.max(axis=1, keepdims=True)
        nbr_std  = nbr_stack.std(axis=1, keepdims=True)
        
        # Time
        h, dw, dy = ts.dt.hour.values, ts.dt.dayofweek.values, ts.dt.dayofyear.values
        time_feats = np.column_stack([
            np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24),
            np.sin(2*np.pi*dw/7), np.cos(2*np.pi*dw/7),
            np.sin(2*np.pi*dy/365), np.cos(2*np.pi*dy/365),
        ])
        
        # Station ID
        station_feat = np.full((T, 1), s_idx / len(SELECTED_STATIONS), dtype=np.float32)
        
        # PM2.5 to last position
        pm25_idx = raw_cols.index(PM25_COL) if PM25_COL in raw_cols else None
        if pm25_idx is not None:
            exo_feats = np.delete(raw_feats, pm25_idx, axis=1)
            pm25_arr = raw_feats[:, pm25_idx:pm25_idx+1]
        else:
            exo_feats = raw_feats
            pm25_arr = np.zeros((T, 1))
        
        parts = [exo_feats, nbr_mean, nbr_max, nbr_std, time_feats, station_feat]
        
        if USE_EMBEDDINGS and df_emb is not None:
            emb_array = np.zeros((T, len(emb_cols)), dtype=np.float32)
            # Embedding matching skipped for simpler chrono code
            parts.append(emb_array)
        
        parts.append(pm25_arr)
        
        features = np.concatenate(parts, axis=1).astype(np.float32)
        features = np.nan_to_num(features, nan=0.0)
        target = df[PM25_COL].values.astype(np.float32)
        
        for split_name in ['train', 'val', 'test']:
            mask = splits == split_name
            result[split_name][sid] = (features[mask], target[mask])
    
    num_features = features.shape[1]
    emb_info = f"Emb: {len(emb_cols)}" if USE_EMBEDDINGS else "Emb: OFF"
    print(f"    Total features: {num_features} (Exo: {exo_feats.shape[1]} + Nbr: 3 + Time: 6 + StID: 1 + {emb_info} + PM2.5: 1)")
    return result, num_features


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_v2_config(num_features, horizon=24):
    cfg = DotDict()
    cfg.seq_len = SEQ_LEN
    cfg.pred_len = 1
    cfg.d_model = 128
    cfg.enc_in = num_features
    cfg.t_ff = 256
    cfg.c_ff = 256
    cfg.usenorm = False
    cfg.embed_dropout = 0.1
    cfg.head_dropout = 0.1
    cfg.t_dropout = 0.1
    cfg.c_dropout = 0.1
    cfg.features = 'MS'
    cfg.horizon = horizon
    return cfg


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


def run_s08_v2():
    print("="*70)
    print("  S08 V2: XLinear V2 Baseline (Fair Benchmark S09)")
    print("="*70)
    
    data, num_features = prepare_features()
    all_results = []
    
    for r_name, r_sids in REGIONS.items():
        print(f"\n[{r_name.upper()}] Processing {len(r_sids)} stations...")
        
        for horizon_h in HORIZONS:
            
            # Chỉ lấy trạm thuộc Region này
            train_ds = S08Dataset(
                [data['train'][s][0] for s in r_sids],
                [data['train'][s][1] for s in r_sids],
                r_sids, SEQ_LEN, horizon_h)
            val_ds = S08Dataset(
                [data['val'][s][0] for s in r_sids],
                [data['val'][s][1] for s in r_sids],
                r_sids, SEQ_LEN, horizon_h)
            test_ds = S08Dataset(
                [data['test'][s][0] for s in r_sids],
                [data['test'][s][1] for s in r_sids],
                r_sids, SEQ_LEN, horizon_h)
            
            train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
            val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, pin_memory=True)
            test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, pin_memory=True)
            
            cfg = get_v2_config(num_features, horizon=horizon_h)
            model = XLinearV2(cfg).to(DEVICE)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
            criterion = nn.MSELoss()
            
            best_val = float('inf')
            patience_cnt = 0
            
            os.makedirs('data/models_saved', exist_ok=True)
            model_path = f'data/models_saved/s08v2_{r_name}_t{horizon_h}.pth'
            
            print(f"  [{r_name.upper()}] T+{horizon_h:<2d} | Đang huấn luyện XLinear V2... (Train: {len(train_ds):,})", end=" ", flush=True)
            t_total_0 = time.time()
            for epoch in range(EPOCHS):
                model.train()
                for bx, by, _ in train_loader:
                    bx = bx.to(DEVICE, non_blocking=True)
                    by = by.to(DEVICE, non_blocking=True)
                    optimizer.zero_grad()
                    out = model(bx)[:, -1, :]
                    loss = criterion(out, by)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                
                model.eval()
                vlosses = []
                with torch.no_grad():
                    for bx, by, _ in val_loader:
                        bx = bx.to(DEVICE, non_blocking=True)
                        by = by.to(DEVICE, non_blocking=True)
                        out = model(bx)[:, -1, :]
                        vlosses.append(criterion(out, by).item())
                
                vl = np.mean(vlosses)
                if vl < best_val:
                    best_val = vl
                    torch.save(model.state_dict(), model_path)
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= 5:
                        break
            
            dt = time.time() - t_total_0
            
            # Eval
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
            
            print(f"-> RMSE={rmse:.2f} | MAE={mae:.2f} | R2={r2*100:.2f}% | MAPE={mape:.2f}% ({dt:.1f}s)")
            all_results.append({'region': r_name, 'horizon': f'T+{horizon_h}', 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape, 'n_test': len(y_true)})

    print("\n" + "="*70)
    print("  FINAL BENCHMARK -- S08 V2 XLinear")
    print("="*70)
    print(f"{'Region':<10} {'Horizon':<8} {'RMSE':>8} {'MAE':>8} {'R2 %':>8} {'MAPE %':>10}")
    print("-"*55)
    for r in all_results:
        print(f"{r['region']:<10} {r['horizon']:<8} {r['RMSE']:>8.2f} {r['MAE']:>8.2f} {r['R2']*100:>7.2f}% {r['MAPE']:>9.2f}%")

    print("\n" + "-" * 55)
    print("AGGREGATED (weighted by test size):")
    for h in HORIZONS:
        hr = [r for r in all_results if r['horizon'] == f'T+{h}']
        if hr:
            total = sum(r['n_test'] for r in hr)
            agg = lambda key: sum(r[key]*r['n_test'] for r in hr) / total
            print(f"  T+{h:<3d}  RMSE={agg('RMSE'):.2f}  MAE={agg('MAE'):.2f}  "
                  f"R2={agg('R2')*100:.2f}%  MAPE={agg('MAPE'):.2f}%")
    print("="*70)

if __name__ == '__main__':
    run_s08_v2()
