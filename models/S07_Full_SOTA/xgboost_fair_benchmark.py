import warnings; warnings.filterwarnings('ignore')
import os, sys, time
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.append('models/S07_Full_SOTA')
from graph import get_base_matrices

DATA_DIR   = 'data/normalized'
SCALER_DIR = DATA_DIR
INFO_PATH  = 'data/info.csv'

SELECTED_STATIONS = [1, 3, 4, 7, 9, 12, 13, 15, 16, 17, 18, 24, 27, 29, 31, 32]
REGIONS = {
    'north': [1, 3, 4, 13, 15, 16, 17, 27, 29],
    'south': [7, 9, 12, 18, 24, 31, 32],
}
HORIZONS = [1, 3, 6, 12, 24]
PM25_COL = 'pm25'

PRECURSOR_COLS = ['pm10', 'so2', 'no2', 'o3', 'co']
FUTURE_WEATHER_COLS = ['temp', 'precip', 'wind_spd', 'rh', 'atm']
EXCLUDE_COLS = ['timestamp', 'split', 'station_id', 'province', 'district']
LAGS = [1, 2, 3, 6, 12, 24, 36, 48, 60, 72]
ROLLS = [3, 6, 12, 24, 48]

def inverse_pm25(y_norm, station_id):
    scaler_path = os.path.join(SCALER_DIR, f'scalers_{station_id}.pkl')
    if not os.path.exists(scaler_path): return y_norm
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    method_tuple = scalers.get('pm25')
    if not method_tuple: return y_norm
    method, sc = method_tuple
    y_inv = sc.inverse_transform(y_norm.reshape(-1, 1)).flatten()
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

def build_knn_indices(k=5):
    distances, _ = get_base_matrices(info_path=INFO_PATH)
    N = distances.shape[0]
    knn = {}
    for i in range(N):
        dist_row = distances[i].copy()
        dist_row[i] = np.inf
        knn[i] = np.argsort(dist_row)[:k].tolist()
    return knn, distances

def compute_neighbor_features(s_idx, all_pm25, knn, distances, lags=(1, 3, 6, 12, 24)):
    neighbor_idxs = knn[s_idx]
    neighbor_pm25 = [all_pm25[n] for n in neighbor_idxs if n in all_pm25]
    neighbor_dists = [distances[s_idx, n] for n in neighbor_idxs if n in all_pm25]
    if not neighbor_pm25: return pd.DataFrame()
    stacked = pd.concat(neighbor_pm25, axis=1)
    inv_dists = np.array([1.0 / (d + 1e-3) for d in neighbor_dists])
    inv_dists_norm = inv_dists / inv_dists.sum()
    result = pd.DataFrame(index=stacked.index)
    for lag in lags:
        lagged = stacked.shift(lag)
        result[f'nbr_mean_lag{lag}'] = lagged.mean(axis=1)
        result[f'nbr_max_lag{lag}'] = lagged.max(axis=1)
        result[f'nbr_std_lag{lag}'] = lagged.std(axis=1).fillna(0)
        result[f'nbr_wmean_lag{lag}'] = (lagged * inv_dists_norm).sum(axis=1)
    result['nbr_nearest_lag1'] = stacked.iloc[:, 0].shift(1)
    result['nbr_nearest_lag6'] = stacked.iloc[:, 0].shift(6)
    result['nbr_nearest_lag24'] = stacked.iloc[:, 0].shift(24)
    return result

def build_features(df_raw, horizon_h, nbr_df=None, s_idx=0):
    df = df_raw.copy()
    feats = {}
    for lag in LAGS:
        feats[f'pm25_lag_{lag}'] = df[PM25_COL].shift(lag)
    for col in PRECURSOR_COLS:
        if col in df.columns:
            for lag in [1, 3, 6, 12, 24]:
                feats[f'{col}_lag_{lag}'] = df[col].shift(lag)
    for w in ROLLS:
        feats[f'pm25_roll_mean_{w}'] = df[PM25_COL].rolling(w, min_periods=1).mean().shift(1)
        feats[f'pm25_roll_std_{w}'] = df[PM25_COL].rolling(w, min_periods=1).std().shift(1).fillna(0)
        feats[f'pm25_roll_max_{w}'] = df[PM25_COL].rolling(w, min_periods=1).max().shift(1)
    for col in FUTURE_WEATHER_COLS:
        if col in df.columns:
            feats[f'{col}_fut_h{horizon_h}'] = df[col].shift(-horizon_h)
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'])
        hour = ts.dt.hour
        feats['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        feats['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        feats['dow_sin'] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
        feats['dow_cos'] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
        feats['doy_sin'] = np.sin(2 * np.pi * ts.dt.dayofyear / 365)
        feats['doy_cos'] = np.cos(2 * np.pi * ts.dt.dayofyear / 365)
    valid_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c != PM25_COL]
    for col in valid_cols:
        feats[f'cur_{col}'] = df[col].values
    feat_df = pd.DataFrame(feats, index=df.index)
    if nbr_df is not None:
        for col in nbr_df.columns:
            feat_df[col] = nbr_df[col].values
            
    # OHE cho Trạm (Thay vì mã hóa numeric index)
    for sid in SELECTED_STATIONS:
        feat_df[f'station_is_{sid}'] = int(sid == SELECTED_STATIONS[s_idx])
        
    target = df[PM25_COL].shift(-horizon_h)
    return feat_df, target

def run_benchmark():
    print("=" * 70)
    print("  S07 XGBoost SOTA — Fair Benchmark (Chronological Split)")
    print("  Train: 2023-01→2024-12 | Val: 2025-01→04 | Test: 2025-05→12")
    print("=" * 70)

    # Nạp toàn bộ data trạm
    all_dfs = {}
    all_pm25 = {}
    for s_idx, sid in enumerate(SELECTED_STATIONS):
        df = pd.read_csv(os.path.join(DATA_DIR, f'norm_station_{sid}.csv'))
        all_dfs[s_idx] = df
        all_pm25[s_idx] = df[PM25_COL].reset_index(drop=True)

    knn, distances = build_knn_indices(k=5)
    all_results = []
    
    device_setup = 'cuda' if torch.cuda.is_available() else 'cpu'

    for r_name, r_sids in REGIONS.items():
        print(f"\n[{r_name.upper()}] Processing {len(r_sids)} stations...")
        
        for horizon_h in HORIZONS:
            X_train_list, y_train_list = [], []
            X_val_list, y_val_list = [], []
            X_test_list, y_test_list = [], []
            sid_test_list = []

            for sid in r_sids:
                s_idx = SELECTED_STATIONS.index(sid)
                df = all_dfs[s_idx].copy()
                ts = pd.to_datetime(df['timestamp'])

                # CHRONOLOGICAL SPLIT
                train_mask = ts < '2025-01-01'
                val_mask = (ts >= '2025-01-01') & (ts < '2025-05-01')
                test_mask = ts >= '2025-05-01'

                nbr_feats = compute_neighbor_features(s_idx, all_pm25, knn, distances)
                feat_df, target = build_features(df, horizon_h, nbr_feats, s_idx)

                X_train_list.append(feat_df[train_mask])
                y_train_list.append(target[train_mask])
                X_val_list.append(feat_df[val_mask])
                y_val_list.append(target[val_mask])
                X_te = feat_df[test_mask]
                X_test_list.append(X_te)
                y_test_list.append(target[test_mask])
                sid_test_list.extend([sid] * len(X_te))

            X_train = pd.concat(X_train_list, ignore_index=True)
            y_train = pd.concat(y_train_list, ignore_index=True)
            X_val = pd.concat(X_val_list, ignore_index=True)
            y_val = pd.concat(y_val_list, ignore_index=True)
            X_test = pd.concat(X_test_list, ignore_index=True)
            y_test_arr = pd.concat(y_test_list, ignore_index=True).values

            # Lọc bỏ dòng NaN
            valid_tr = ~(pd.isna(y_train) | X_train.isnull().any(axis=1))
            valid_vl = ~(pd.isna(y_val) | X_val.isnull().any(axis=1))
            valid_te = ~(pd.isna(y_test_arr) | X_test.isnull().any(axis=1))

            # Loại bỏ features có tên không hợp lệ đối với XGBoost (chứa kí tự [])
            X_train.columns = X_train.columns.str.replace('[', '').str.replace(']', '').str.replace('<', '')
            X_val.columns = X_val.columns.str.replace('[', '').str.replace(']', '').str.replace('<', '')
            X_test.columns = X_test.columns.str.replace('[', '').str.replace(']', '').str.replace('<', '')

            # Ráp vào Array để Train
            X_train, y_train = X_train[valid_tr].fillna(0).values, y_train[valid_tr].values
            X_val, y_val = X_val[valid_vl].fillna(0).values, y_val[valid_vl].values
            X_test, y_test_arr = X_test[valid_te].fillna(0).values, y_test_arr[valid_te]
            sid_test_arr = np.array(sid_test_list)[valid_te.values]

            print(f"  [{r_name.upper()}] T+{horizon_h:<2d} | Đang huấn luyện XGBoost... (Train: {len(X_train):,})", end=" ", flush=True)

            t0 = time.time()
            model = xgb.XGBRegressor(
                n_estimators=300, learning_rate=0.08, max_depth=7,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                reg_alpha=0.1, reg_lambda=1.0,
                tree_method='hist', device=device_setup,
                n_jobs=-1, random_state=42,
                early_stopping_rounds=30, eval_metric='rmse',
            )
            
            # Câm họng cái early stopping để đỡ trôi tin nhắn console
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
            dt = time.time() - t0

            y_pred_norm = model.predict(X_test)

            # Inverse metrics per station
            y_true_inv = np.zeros_like(y_test_arr)
            y_pred_inv = np.zeros_like(y_pred_norm)
            for sid in r_sids:
                mask = sid_test_arr == sid
                if mask.sum() > 0:
                    y_true_inv[mask] = inverse_pm25(y_test_arr[mask], sid)
                    y_pred_inv[mask] = inverse_pm25(y_pred_norm[mask], sid)

            rmse, mae, r2, mape = get_metrics(y_true_inv, y_pred_inv)
            print(f"-> RMSE={rmse:.2f} | MAE={mae:.2f} | R2={r2*100:.2f}% | MAPE={mape:.2f}% ({dt:.1f}s)")
            
            all_results.append({
                'region': r_name, 'horizon': f'T+{horizon_h}',
                'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape,
                'n_test': len(y_test_arr)
            })

    print("\n" + "=" * 70)
    print("  FINAL BENCHMARK — S07 XGBoost SOTA")
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
            agg = lambda key: sum(r[key]*r['n_test'] for r in hr) / total
            print(f"  T+{h:<3d}  RMSE={agg('RMSE'):.2f}  MAE={agg('MAE'):.2f}  "
                  f"R2={agg('R2')*100:.2f}%  MAPE={agg('MAPE'):.2f}%")
    print("=" * 70)

if __name__ == '__main__':
    run_benchmark()
