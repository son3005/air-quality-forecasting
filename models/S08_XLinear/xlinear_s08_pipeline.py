"""
S08_XLinear: xlinear_s08_pipeline.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Kiến trúc giống hệt S07 nhưng thay XGBoost bằng XLinear.

S07 Flow:
  1) E-STGCN train → extract spatial embeddings
  2) Feature engineering (lags, rolling, neighbors, precursors)
  3) XGBoost predict PM2.5 at T+h

S08 Flow:
  1) Reuse spatial embeddings từ S07 (đã có sẵn)
  2) Raw features + spatial embeddings + neighbor stats (KHÔNG CẦN lags/rolling vì XLinear đọc sequence trực tiếp)
  3) XLinear predict PM2.5 at T+h

Lợi thế: XLinear nhìn chuỗi 48h liên tục → tự học temporal patterns.
         Không cần handcraft 150+ lag features → train nhanh gấp 3x.
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

# Import XLinear model
sys.path.append('tmp_xlinear')
from models.XLinear import Model as XLinearModel

# Import spatial utilities
sys.path.append('models/S08_XLinear')
from graph import get_base_matrices

# ── CẤU HÌNH ─────────────────────────────────────────────────────────────
DATA_DIR    = 'data/normalized'
EMBED_PATH  = 'data/extracted_embeddings/spatial_embeddings.csv'
SCALER_DIR  = DATA_DIR
INFO_PATH   = 'data/info.csv'

SELECTED_STATIONS = [1, 3, 4, 7, 9, 12, 13, 15, 16, 17, 18, 24, 27, 29, 31, 32]
EXCLUDE_COLS = ['timestamp', 'split', 'station_id', 'province', 'district']
PM25_COL = 'pm25'

SEQ_LEN    = 48     # 48h lookback
HORIZONS   = [1, 6, 12, 24]
BATCH_SIZE = 128    # Tăng batch size (ít features hơn = GPU chứa được nhiều hơn)
EPOCHS     = 15
LR         = 5e-4
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"[*] Using device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════
# 1. SPATIAL UTILITIES (reuse từ S07)
# ══════════════════════════════════════════════════════════════════════════

def build_knn_indices(k=5):
    distances, _ = get_base_matrices(info_path=INFO_PATH)
    N = distances.shape[0]
    knn = {}
    for i in range(N):
        dist_row = distances[i].copy()
        dist_row[i] = np.inf
        neighbor_idxs = np.argsort(dist_row)[:k]
        knn[i] = neighbor_idxs.tolist()
    return knn


# ══════════════════════════════════════════════════════════════════════════
# 2. DATASET: On-the-fly windowing (tiết kiệm RAM)
# ══════════════════════════════════════════════════════════════════════════

class S08Dataset(Dataset):
    """
    Mỗi station: features array shape (T, F).
    Sliding window tạo (seq_len, F) → predict PM2.5 at t + horizon.
    On-the-fly: không lưu toàn bộ sequences vào RAM.
    """
    def __init__(self, station_features, station_targets, station_ids, 
                 seq_len=48, horizon=1):
        """
        station_features: list of np.array (T_i, F) for each station
        station_targets: list of np.array (T_i,) — PM2.5 at t for each station
        station_ids: list of int — station IDs
        """
        self.seq_len = seq_len
        self.horizon = horizon
        
        # Build index: (station_local_idx, start_pos)
        self.index = []
        self.features = station_features
        self.targets = station_targets
        self.sids = station_ids
        
        for s_local, (feat, tgt) in enumerate(zip(station_features, station_targets)):
            max_start = len(feat) - seq_len - horizon
            for i in range(max(0, max_start + 1)):
                # Kiểm tra target có NaN không
                t_idx = i + seq_len - 1 + horizon
                if t_idx < len(tgt) and not np.isnan(tgt[t_idx]):
                    self.index.append((s_local, i))
                    
    def __len__(self):
        return len(self.index)
        
    def __getitem__(self, idx):
        s_local, start = self.index[idx]
        feat = self.features[s_local]
        tgt  = self.targets[s_local]
        sid  = self.sids[s_local]
        
        x = feat[start : start + self.seq_len]           # (seq_len, F)
        y = tgt[start + self.seq_len - 1 + self.horizon]  # scalar
        
        return (torch.tensor(x, dtype=torch.float32), 
                torch.tensor([y], dtype=torch.float32), 
                sid)


# ══════════════════════════════════════════════════════════════════════════
# 3. DATA LOADING & FEATURE CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════

def prepare_station_features():
    """
    Cho mỗi trạm, tạo feature matrix:
      - Raw normalized features (pm25, pm10, so2, no2, o3, co, temp, precip, ...)
      - Neighbor mean PM2.5 (K=5 nearest)
      - Spatial Embeddings (từ E-STGCN S07)
      - Time encodings (hour, dow, doy sin/cos)
    
    Trả về dict {split: {sid: (features_array, target_array)}}
    """
    print("\n[1] Loading all station data...")
    all_dfs = {}
    for sid in SELECTED_STATIONS:
        path = os.path.join(DATA_DIR, f'norm_station_{sid}.csv')
        all_dfs[sid] = pd.read_csv(path)
    
    # PM2.5 arrays cho neighbor computation
    pm25_arrays = {}
    for s_idx, sid in enumerate(SELECTED_STATIONS):
        pm25_arrays[s_idx] = all_dfs[sid][PM25_COL].values
    
    # K-nearest neighbors
    print("[2] Computing K=5 nearest neighbors...")
    knn = build_knn_indices(k=5)
    
    # Spatial embeddings
    print("[3] Loading spatial embeddings...")
    df_emb = pd.read_csv(EMBED_PATH)
    emb_cols = [c for c in df_emb.columns if c.startswith('ST_Emb_')]
    print(f"    Embedding dims: {len(emb_cols)}")
    
    # Valid feature columns (raw normalized)
    sample_df = all_dfs[SELECTED_STATIONS[0]]
    raw_cols = [c for c in sample_df.columns if c not in EXCLUDE_COLS]
    print(f"    Raw feature cols: {len(raw_cols)}")
    
    # Build per-station, per-split data
    result = {'train': {}, 'val': {}, 'test': {}}
    
    for s_idx, sid in enumerate(SELECTED_STATIONS):
        df = all_dfs[sid]
        T = len(df)
        
        # === Raw features ===
        raw_feats = df[raw_cols].values  # (T, ~35)
        
        # === Neighbor mean PM2.5 ===
        nbr_idxs = knn[s_idx]
        nbr_pm25_stack = np.column_stack([pm25_arrays[n] for n in nbr_idxs])  # (T, K)
        nbr_mean = nbr_pm25_stack.mean(axis=1, keepdims=True)   # (T, 1)
        nbr_max  = nbr_pm25_stack.max(axis=1, keepdims=True)    # (T, 1)
        nbr_std  = nbr_pm25_stack.std(axis=1, keepdims=True)    # (T, 1)
        
        # === Time encodings ===
        if 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'])
            hour = ts.dt.hour.values
            dow  = ts.dt.dayofweek.values
            doy  = ts.dt.dayofyear.values
            time_feats = np.column_stack([
                np.sin(2 * np.pi * hour / 24),
                np.cos(2 * np.pi * hour / 24),
                np.sin(2 * np.pi * dow / 7),
                np.cos(2 * np.pi * dow / 7),
                np.sin(2 * np.pi * doy / 365),
                np.cos(2 * np.pi * doy / 365),
            ])  # (T, 6)
        else:
            time_feats = np.zeros((T, 6))
        
        # === Spatial embeddings (per-split alignment) ===
        splits = df['split'].values
        emb_array = np.zeros((T, len(emb_cols)), dtype=np.float32)
        for split_name in ['train', 'val', 'test']:
            mask = splits == split_name
            emb_split = df_emb[(df_emb['split'] == split_name) & 
                              (df_emb['station_idx'] == s_idx)][emb_cols].values
            n_mask = mask.sum()
            if len(emb_split) >= n_mask:
                emb_array[mask] = emb_split[:n_mask]
            else:
                emb_array[mask][:len(emb_split)] = emb_split
        
        # === Station ID embedding (one-hot-like) ===
        station_feat = np.full((T, 1), s_idx / len(SELECTED_STATIONS), dtype=np.float32)
        
        # === Concatenate ALL features ===
        # PM2.5 (endogenous) MUST be the LAST column for XLinear
        # raw_feats already contains pm25 somewhere, remove it and append at end
        pm25_idx = raw_cols.index(PM25_COL) if PM25_COL in raw_cols else None
        if pm25_idx is not None:
            exo_feats = np.delete(raw_feats, pm25_idx, axis=1)  # Remove pm25 from middle
            pm25_col_arr = raw_feats[:, pm25_idx:pm25_idx+1]     # Save pm25 separately
        else:
            exo_feats = raw_feats
            pm25_col_arr = np.zeros((T, 1))
        
        features = np.concatenate([
            exo_feats,       # Exogenous raw features (weather, precursors) 
            nbr_mean,        # Neighbor mean PM2.5
            nbr_max,         # Neighbor max PM2.5
            nbr_std,         # Neighbor std PM2.5
            time_feats,      # Time encodings
            emb_array,       # Spatial embeddings
            station_feat,    # Station ID
            pm25_col_arr,    # PM2.5 (ENDOGENOUS - LAST!)
        ], axis=1).astype(np.float32)
        
        # Replace NaN with 0
        features = np.nan_to_num(features, nan=0.0)
        
        # Target = PM2.5 (normalized values) — horizons handled by Dataset class
        target = df[PM25_COL].values.astype(np.float32)
        
        # Split
        for split_name in ['train', 'val', 'test']:
            mask = splits == split_name
            result[split_name][sid] = (features[mask], target[mask])
    
    if s_idx == 0:
        pass
    num_features = features.shape[1]
    print(f"    Total features per timestep: {num_features}")
    print(f"    (Exo: {exo_feats.shape[1]} + Nbr: 3 + Time: 6 + Emb: {len(emb_cols)} + StID: 1 + PM2.5: 1)")
    
    return result, num_features


# ══════════════════════════════════════════════════════════════════════════
# 4. XLinear CONFIG & UTILITIES
# ══════════════════════════════════════════════════════════════════════════

class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_xlinear_config(num_features):
    cfg = DotDict()
    cfg.seq_len = SEQ_LEN
    cfg.pred_len = 1
    cfg.d_model = 128
    cfg.enc_in = num_features
    cfg.t_ff = 256
    cfg.c_ff = 256
    cfg.usenorm = False    # TẮT internal norm — data đã normalized sẵn bởi per-station scaler
    cfg.embed_dropout = 0.1
    cfg.head_dropout = 0.1
    cfg.t_dropout = 0.1
    cfg.c_dropout = 0.1
    cfg.features = 'MS'    # Multivariate → Single (predict last channel = PM2.5)
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

def get_metrics_real(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = compute_mape(y_true, y_pred)
    return rmse, mae, r2, mape


# ══════════════════════════════════════════════════════════════════════════
# 5. MAIN TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def run_s08():
    print("="*70)
    print("  S08 PIPELINE: E-STGCN Embeddings + XLinear (replaces XGBoost)")
    print("="*70)
    
    # Pre-compute features ONCE (shared across all horizons)
    data_splits, num_features = prepare_station_features()
    
    benchmark = []
    
    for horizon_h in HORIZONS:
        print(f"\n{'-'*60}")
        print(f"  Training XLinear for T+{horizon_h}h")
        print(f"{'-'*60}")
        
        # Build datasets from pre-computed features
        train_feats = [data_splits['train'][sid][0] for sid in SELECTED_STATIONS]
        train_tgts  = [data_splits['train'][sid][1] for sid in SELECTED_STATIONS]
        val_feats   = [data_splits['val'][sid][0]   for sid in SELECTED_STATIONS]
        val_tgts    = [data_splits['val'][sid][1]    for sid in SELECTED_STATIONS]
        test_feats  = [data_splits['test'][sid][0]  for sid in SELECTED_STATIONS]
        test_tgts   = [data_splits['test'][sid][1]   for sid in SELECTED_STATIONS]
        
        train_ds = S08Dataset(train_feats, train_tgts, SELECTED_STATIONS, SEQ_LEN, horizon_h)
        val_ds   = S08Dataset(val_feats,   val_tgts,   SELECTED_STATIONS, SEQ_LEN, horizon_h)
        test_ds  = S08Dataset(test_feats,  test_tgts,  SELECTED_STATIONS, SEQ_LEN, horizon_h)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  
                                  drop_last=True, num_workers=0, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, 
                                  num_workers=0, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, 
                                  num_workers=0, pin_memory=True)
        
        print(f"    Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
        print(f"    Features: {num_features}")
        
        # Init model
        cfg = get_xlinear_config(num_features)
        model = XLinearModel(cfg).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(EPOCHS):
            t0 = time.time()
            
            model.train()
            train_losses = []
            for batch_x, batch_y, _ in train_loader:
                batch_x = batch_x.to(DEVICE, non_blocking=True)
                batch_y = batch_y.to(DEVICE, non_blocking=True)
                
                optimizer.zero_grad()
                out = model(batch_x)[:, -1, :]  # (B, 1)
                loss = criterion(out, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())
            
            scheduler.step()
            
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, batch_y, _ in val_loader:
                    batch_x = batch_x.to(DEVICE, non_blocking=True)
                    batch_y = batch_y.to(DEVICE, non_blocking=True)
                    out = model(batch_x)[:, -1, :]
                    loss = criterion(out, batch_y)
                    val_losses.append(loss.item())
            
            tl = np.mean(train_losses)
            vl = np.mean(val_losses)
            dt = time.time() - t0
            print(f"    Epoch {epoch+1:02d}/{EPOCHS} | Train: {tl:.4f} | Val: {vl:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | {dt:.1f}s")
            
            if vl < best_val_loss:
                best_val_loss = vl
                os.makedirs('models_saved', exist_ok=True)
                torch.save(model.state_dict(), f'models_saved/s08_t{horizon_h}.pth')
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    Early stop at epoch {epoch+1}!")
                    break
        
        # ── Evaluation ────────────────────────────────────────────────────
        print(f"\n  [*] Evaluating S08 T+{horizon_h}...")
        model.load_state_dict(torch.load(f'models_saved/s08_t{horizon_h}.pth', weights_only=True))
        model.eval()
        
        all_preds, all_trues = [], []
        with torch.no_grad():
            for batch_x, batch_y, sids in test_loader:
                batch_x = batch_x.to(DEVICE, non_blocking=True)
                preds = model(batch_x)[:, -1, :].cpu().numpy()  # (B, 1)
                trues = batch_y.numpy()                          # (B, 1)
                
                for i in range(len(sids)):
                    sid = sids[i].item()
                    p = inverse_pm25(preds[i], sid)
                    t = inverse_pm25(trues[i], sid)
                    all_preds.append(p[0])
                    all_trues.append(t[0])
        
        y_true = np.array(all_trues)
        y_pred = np.array(all_preds)
        
        rmse, mae, r2, mape = get_metrics_real(y_true, y_pred)
        print(f"  T+{horizon_h:2d} | RMSE={rmse:.2f} | MAE={mae:.2f} | R²={r2*100:.2f}% | MAPE={mape:.2f}%")
        
        benchmark.append({
            'horizon': f'T+{horizon_h}', 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape
        })
    
    # ── Final Summary ─────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  FINAL BENCHMARK — S08 (E-STGCN Embeddings + XLinear)")
    print("="*70)
    print(f"{'Horizon':<10} {'RMSE':>8} {'MAE':>8} {'R² %':>8} {'MAPE %':>10}")
    print("-"*50)
    for r in benchmark:
        print(f"{r['horizon']:<10} {r['RMSE']:>8.2f} {r['MAE']:>8.2f} {r['R2']*100:>7.2f}% {r['MAPE']:>9.2f}%")
    print("="*70)

if __name__ == '__main__':
    run_s08()
