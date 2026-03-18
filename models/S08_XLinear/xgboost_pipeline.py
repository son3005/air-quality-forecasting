"""
S07 XGBoost Pipeline (Fix #4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Load spatial embeddings từ DynamicESTGCN, sau đó kết hợp toàn bộ feature
engineering và train XGBoost riêng cho T+1, T+6, T+12, T+24.
"""
import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Path for graph utils
sys.path.append('models/S07_Full_SOTA')
from graph import get_base_matrices

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR    = 'data/normalized'
EMBED_PATH  = 'data/extracted_embeddings/spatial_embeddings.csv'
SCALER_DIR  = DATA_DIR
INFO_PATH   = 'data/info.csv'

SELECTED_STATIONS = [1, 3, 4, 7, 9, 12, 13, 15, 16, 17, 18, 24, 27, 29, 31, 32]
HORIZONS = [1, 6, 12, 24]

# ── Feature Columns ───────────────────────────────────────────────────────
PRECURSOR_COLS = ['pm10', 'so2', 'no2', 'o3', 'co']
FUTURE_WEATHER_COLS = ['temp', 'precip', 'wind_spd', 'rh', 'atm']  # Available in dataset

EXCLUDE_COLS = ['timestamp', 'split', 'station_id', 'province', 'district']
PM25_COL = 'pm25'

LAGS    = [1, 2, 3, 6, 12, 24, 36, 48, 60, 72]
ROLLS   = [3, 6, 12, 24, 48]


# ══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def inverse_pm25(y_norm, station_id, scaler_dir=SCALER_DIR):
    scaler_path = os.path.join(scaler_dir, f'scalers_{station_id}.pkl')
    if not os.path.exists(scaler_path):
        return y_norm
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    method_tuple = scalers.get('pm25')
    if method_tuple is None:
        return y_norm
    method, sc = method_tuple
    y_inv = sc.inverse_transform(y_norm.reshape(-1, 1)).flatten()
    if 'log1p' in method:
        y_inv = np.expm1(y_inv)
    return y_inv

def compute_mape(y_true, y_pred):
    mask = y_true > 1.0
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def get_metrics_real(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = compute_mape(y_true, y_pred)
    return rmse, mae, r2, mape


# ══════════════════════════════════════════════════════════════════════════
# SPATIAL NEIGHBOR UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def build_knn_indices(k=5):
    """
    Tính K-nearest neighbors cho mỗi trong 16 trạm dựa theo khoảng cách địa lý.
    Trả về: knn dict và distances array (N, N)
    """
    distances, _ = get_base_matrices(info_path='data/info.csv')
    N = distances.shape[0]
    knn = {}
    for i in range(N):
        dist_row = distances[i].copy()
        dist_row[i] = np.inf  # loại self
        neighbor_idxs = np.argsort(dist_row)[:k]
        knn[i] = neighbor_idxs.tolist()
    return knn, distances


def compute_neighbor_features(s_idx, all_station_pm25, knn_indices, distances_matrix, lags=(1, 3, 6, 12, 24)):
    """
    Tả về DataFrame của Aggregated Neighbor PM2.5 Stats. 
    Bao gồm cả simple agg (mean/max/std) và distance-weighted mean.
    all_station_pm25: dict {station_idx -> pd.Series index aligned với df_raw}
    """
    neighbor_idxs = knn_indices[s_idx]
    neighbor_pm25 = [all_station_pm25[n] for n in neighbor_idxs if n in all_station_pm25]
    neighbor_dists = [distances_matrix[s_idx, n] for n in neighbor_idxs if n in all_station_pm25]
    
    if not neighbor_pm25:
        return pd.DataFrame()
    
    # Stack neighbors: each Series already indexed by row position
    stacked = pd.concat(neighbor_pm25, axis=1)
    stacked.columns = [f'n{i}' for i in range(len(neighbor_pm25))]
    
    # Compute inverse-distance weights: closer station -> higher weight
    inv_dists = np.array([1.0 / (d + 1e-3) for d in neighbor_dists])  # avoid div/0
    inv_dists_norm = inv_dists / inv_dists.sum()  # normalize to sum=1
    
    result = pd.DataFrame(index=stacked.index)
    
    # Aggregate: mean/max/std AND distance-weighted mean at each lag
    for lag in lags:
        lagged = stacked.shift(lag)
        result[f'nbr_mean_lag{lag}'] = lagged.mean(axis=1)
        result[f'nbr_max_lag{lag}']  = lagged.max(axis=1)
        result[f'nbr_std_lag{lag}']  = lagged.std(axis=1).fillna(0)
        # Distance-weighted mean: each neighbor weighted by inverse distance
        result[f'nbr_wmean_lag{lag}'] = (lagged * inv_dists_norm).sum(axis=1)
    
    # Nearest neighbor's raw value (most relevant single station)
    result['nbr_nearest_lag1']  = stacked.iloc[:, 0].shift(1)   # closest neighbor at t-1
    result['nbr_nearest_lag6']  = stacked.iloc[:, 0].shift(6)   # closest neighbor at t-6
    result['nbr_nearest_lag24'] = stacked.iloc[:, 0].shift(24)  # closest neighbor at t-24
    
    return result


# ══════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING PER STATION
# ══════════════════════════════════════════════════════════════════════════

def build_features_per_station(df_raw, horizon_h, embed_df_station, neighbor_feats_df=None, s_idx=0):
    """
    Tạo feature matrix cho từng trạm với:
      - PM2.5 lag features
      - Precursor lag features (PM10, SO2, NO2, O3, CO)  
      - Rolling stats of PM2.5
      - Future weather at T+h (shift by -h)
      - Spatial embeddings
      - Neighbor aggregated stats (mean/max/std)
      - Time encodings
    """
    df = df_raw.copy()
    feats = {}
    
    # 1. PM2.5 Lag features
    for lag in LAGS:
        feats[f'pm25_lag_{lag}'] = df[PM25_COL].shift(lag)
    
    # 2. Precursor lags
    for col in PRECURSOR_COLS:
        if col in df.columns:
            for lag in [1, 3, 6, 12, 24]:
                feats[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # 3. Rolling stats of PM2.5
    for w in ROLLS:
        feats[f'pm25_roll_mean_{w}'] = df[PM25_COL].rolling(w, min_periods=1).mean().shift(1)
        feats[f'pm25_roll_std_{w}']  = df[PM25_COL].rolling(w, min_periods=1).std().shift(1).fillna(0)
        feats[f'pm25_roll_max_{w}']  = df[PM25_COL].rolling(w, min_periods=1).max().shift(1)
    
    # 4. Future weather at T+h
    for col in FUTURE_WEATHER_COLS:
        if col in df.columns:
            feats[f'{col}_fut_h{horizon_h}'] = df[col].shift(-horizon_h)
    
    # 5. Time encodings
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'])
        hour = ts.dt.hour
        feats['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        feats['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        dow = ts.dt.dayofweek
        feats['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        feats['dow_cos'] = np.cos(2 * np.pi * dow / 7)
        doy = ts.dt.dayofyear
        feats['doy_sin'] = np.sin(2 * np.pi * doy / 365)
        feats['doy_cos'] = np.cos(2 * np.pi * doy / 365)
    
    # 6. Current normalized features as-is
    valid_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c != PM25_COL]
    for col in valid_cols:
        feats[f'cur_{col}'] = df[col].values
    
    feat_df = pd.DataFrame(feats, index=df.index)
    
    # 7. Merge spatial embeddings (align by position)
    if embed_df_station is not None and not embed_df_station.empty:
        embed_cols = [c for c in embed_df_station.columns if c.startswith('ST_Emb_')]
        for col in embed_cols:
            feat_df[col] = embed_df_station[col].values[:len(feat_df)] if len(embed_df_station) >= len(feat_df) \
                           else np.nan
    
    # 8. Station ID (categorical) - allows XGBoost to learn per-station biases
    feat_df['station_id'] = s_idx
    
    # 9. Neighbor Aggregated Stats (spatial lags)
    if neighbor_feats_df is not None:
        nb_cols = [c for c in neighbor_feats_df.columns]
        for col in nb_cols:
            feat_df[col] = neighbor_feats_df[col].values
        
        # 10. Interaction Features (pm25 × neighbor context)
        pm25_lag1 = feat_df.get('pm25_lag_1', pd.Series(0.0, index=feat_df.index))
        pm25_lag24 = feat_df.get('pm25_lag_24', pd.Series(0.0, index=feat_df.index))
        
        # A. PM2.5 lag1 × neighbor mean lag1 (cross-station pollution flow)
        if 'nbr_mean_lag1' in feat_df.columns:
            feat_df['interact_pm25_x_nbrmean'] = pm25_lag1 * feat_df['nbr_mean_lag1']
        
        # B. PM2.5 trend ratio (current / 24h ago) - captures rising/falling trends
        feat_df['pm25_trend_ratio'] = pm25_lag1 / (pm25_lag24.replace(0, np.nan)).fillna(1.0)
        feat_df['pm25_trend_ratio'] = feat_df['pm25_trend_ratio'].clip(-5, 5)  # bound extremes
        
        # C. Hour of day × PM2.5 lag1 (daily pollution pattern interaction)
        if 'hour_sin' in feat_df.columns:
            feat_df['interact_hour_x_pm25'] = feat_df['hour_sin'] * pm25_lag1
        
        # D. Neighbor pm25 rate of change (lag1 - lag6) ÷ 5
        if 'nbr_mean_lag1' in feat_df.columns and 'nbr_mean_lag6' in feat_df.columns:
            feat_df['nbr_delta_1_6'] = (feat_df['nbr_mean_lag1'] - feat_df['nbr_mean_lag6']) / 5.0
    
    # 11. Target: PM2.5 at T+h
    target = df[PM25_COL].shift(-horizon_h)
    
    return feat_df, target


# ══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def run_xgboost_pipeline():
    print("="*70)
    print("  S07 XGBoost Pipeline   (Dynamic STGCN Embeddings + Precursors)")
    print("="*70)
    
    # Load spatial embeddings
    print(f"\n[1] Loading spatial embeddings from {EMBED_PATH}...")
    df_emb = pd.read_csv(EMBED_PATH)
    emb_cols = [c for c in df_emb.columns if c.startswith('ST_Emb_')]
    print(f"    Embedding columns: {len(emb_cols)} dims | Rows: {len(df_emb)}")
    
    # [NEW] Pre-load toàn bộ PM2.5 data của 16 trạm để tính Neighbor Features
    print(f"\n[2] Pre-loading all station PM2.5 data for Neighbor Features...")
    all_station_pm25_full = {}  # {station_idx: pd.Series aligned with df_raw}
    all_station_df = {}
    for s_idx, sid in enumerate(SELECTED_STATIONS):
        path = os.path.join(DATA_DIR, f'norm_station_{sid}.csv')
        df_s = pd.read_csv(path)
        all_station_pm25_full[s_idx] = df_s[PM25_COL].reset_index(drop=True)
        all_station_df[s_idx] = df_s
    
    # [NEW] Tính K-nearest neighbor indices (K=5, dựa theo khoảng cách địa lý)
    print(f"[3] Computing K=5 nearest neighbor indices...")
    knn_indices, distances_matrix = build_knn_indices(k=5)
    
    results_all = []
    
    for horizon_h in HORIZONS:
        print(f"\n{'-'*60}")
        print(f"  Training XGBoost for T+{horizon_h}h")
        print(f"{'-'*60}")
        
        X_train_list, y_train_list = [], []
        X_val_list,   y_val_list   = [], []
        X_test_list,  y_test_list  = [], []
        
        for s_idx, sid in enumerate(SELECTED_STATIONS):
            df_raw = all_station_df[s_idx].copy()
            
            # Get embeddings for this station
            emb_station_train = df_emb[(df_emb['split'] == 'train') & (df_emb['station_idx'] == s_idx)].reset_index(drop=True)
            emb_station_val   = df_emb[(df_emb['split'] == 'val')   & (df_emb['station_idx'] == s_idx)].reset_index(drop=True)
            emb_station_test  = df_emb[(df_emb['split'] == 'test')  & (df_emb['station_idx'] == s_idx)].reset_index(drop=True)
            
            # [NEW] Compute Neighbor Aggregated Stats for this station
            nbr_feats = compute_neighbor_features(
                s_idx=s_idx,
                all_station_pm25=all_station_pm25_full,
                knn_indices=knn_indices,
                distances_matrix=distances_matrix,
                lags=(1, 3, 6, 12, 24)
            )
            
            # Build features for full timeline, then split (inject s_idx for station_id feature)
            feat_df, target = build_features_per_station(
                df_raw, horizon_h, None,
                neighbor_feats_df=nbr_feats if not nbr_feats.empty else None,
                s_idx=s_idx
            )
            
            train_mask = df_raw['split'] == 'train'
            val_mask   = df_raw['split'] == 'val'
            test_mask  = df_raw['split'] == 'test'
            
            def align_embs(emb_df, mask_df, feat_subset):
                """Append embedding as extra columns."""
                n = len(feat_subset)
                emb_arr = emb_df[emb_cols].values
                # Pad/trim
                if len(emb_arr) >= n:
                    emb_arr = emb_arr[:n]
                else:
                    pad = np.zeros((n - len(emb_arr), len(emb_cols)))
                    emb_arr = np.vstack([emb_arr, pad])
                emb_part = pd.DataFrame(emb_arr, columns=emb_cols, index=feat_subset.index)
                return pd.concat([feat_subset.reset_index(drop=True), emb_part.reset_index(drop=True)], axis=1)
            
            X_tr = align_embs(emb_station_train, train_mask, feat_df[train_mask])
            y_tr = target[train_mask].values
            
            X_vl = align_embs(emb_station_val, val_mask, feat_df[val_mask])
            y_vl = target[val_mask].values
            
            X_te = align_embs(emb_station_test, test_mask, feat_df[test_mask])
            y_te = target[test_mask].values
            
            X_train_list.append(X_tr); y_train_list.append(y_tr)
            X_val_list.append(X_vl);   y_val_list.append(y_vl)
            X_test_list.append(X_te);  y_test_list.append(y_te)

        
        X_train = pd.concat(X_train_list, ignore_index=True)
        y_train = np.concatenate(y_train_list)
        X_val   = pd.concat(X_val_list, ignore_index=True)
        y_val   = np.concatenate(y_val_list)
        X_test  = pd.concat(X_test_list, ignore_index=True)
        y_test  = np.concatenate(y_test_list)
        
        # Drop rows with NaN target or NaN features
        valid_train = ~(np.isnan(y_train) | X_train.isnull().any(axis=1).values)
        valid_val   = ~(np.isnan(y_val)   | X_val.isnull().any(axis=1).values)
        valid_test  = ~(np.isnan(y_test)  | X_test.isnull().any(axis=1).values)
        
        X_train, y_train = X_train[valid_train].fillna(0), y_train[valid_train]
        X_val,   y_val   = X_val[valid_val].fillna(0),     y_val[valid_val]
        X_test,  y_test  = X_test[valid_test].fillna(0),   y_test[valid_test]
        
        print(f"    Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
        print(f"    Features: {X_train.shape[1]}")
        
        # Train XGBoost
        model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method='hist',   # XGBoost v2+: use 'hist' + device='cuda'
            device='cuda',
            n_jobs=-1,
            random_state=42,
            early_stopping_rounds=30,
            eval_metric='rmse',
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )
        
        # Predict on test
        y_pred_norm = model.predict(X_test)
        
        # Inverse transform to real scale (per station)
        # Re-attach station info for inverse transform
        n_per_station = len(y_test) // len(SELECTED_STATIONS)
        y_test_inv  = np.zeros_like(y_test)
        y_pred_inv  = np.zeros_like(y_pred_norm)
        
        for s_idx, sid in enumerate(SELECTED_STATIONS):
            sl = slice(s_idx * n_per_station, (s_idx + 1) * n_per_station)
            try:
                y_test_inv[sl] = inverse_pm25(y_test[sl], sid)
                y_pred_inv[sl] = inverse_pm25(y_pred_norm[sl], sid)
            except Exception:
                y_test_inv[sl] = y_test[sl]
                y_pred_inv[sl] = y_pred_norm[sl]
        
        rmse, mae, r2, mape = get_metrics_real(y_test_inv, y_pred_inv)
        
        print(f"\n  T+{horizon_h:2d} | RMSE={rmse:.2f} | MAE={mae:.2f} | R²={r2*100:.2f}% | MAPE={mape:.2f}%")
        results_all.append({
            'horizon': f'T+{horizon_h}',
            'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape
        })
        
        # Save model
        os.makedirs('models_saved', exist_ok=True)
        model.save_model(f'models_saved/xgb_s07_t{horizon_h}.json')
        print(f"  [*] Saved: models_saved/xgb_s07_t{horizon_h}.json")
    
    # ── Summary Table ──────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  FINAL BENCHMARK RESULTS S07 (DynamicSTGCN + XGBoost)")
    print("="*70)
    print(f"{'Horizon':<10} {'RMSE':>8} {'MAE':>8} {'R² %':>8} {'MAPE %':>10}")
    print("-"*50)
    for r in results_all:
        print(f"{r['horizon']:<10} {r['RMSE']:>8.2f} {r['MAE']:>8.2f} {r['R2']*100:>7.2f}% {r['MAPE']:>9.2f}%")
    print("="*70)
    return results_all


if __name__ == '__main__':
    # Run from repo root: e:\University\Year 3 -2\DA2\CODE
    run_xgboost_pipeline()
