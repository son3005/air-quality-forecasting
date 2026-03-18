import os
import pandas as pd
import numpy as np
import pickle
import warnings
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════
# CONFIG & HORIZONS
# ══════════════════════════════════════════════════════════════════════════
HORIZONS = [1, 3, 6, 12, 24]

REGIONS = {
    'north': [1, 3, 4, 13, 15, 16, 17, 27, 29],
    'south': [7, 9, 12, 18, 24, 31, 32],
}

def inverse_pm25(scaled_vals, scaler_tuple):
    if not scaler_tuple: return scaled_vals
    method, sc = scaler_tuple
    unscaled = sc.inverse_transform(np.array(scaled_vals).reshape(-1, 1)).flatten()
    if 'log1p' in method:
        return np.expm1(unscaled)
    return unscaled

def compute_mape(y_true, y_pred):
    mask = y_true > 1.0
    if mask.sum() == 0: return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def get_metrics(y_true, y_pred):
    return (
        np.sqrt(mean_squared_error(y_true, y_pred)),
        mean_absolute_error(y_true, y_pred),
        r2_score(y_true, y_pred),
        compute_mape(y_true, y_pred)
    )

def create_xgboost_dataset(region_sids, horizon=1, seq_len=24):
    all_X_train, all_y_train = [], []
    all_X_test, all_y_test = [], []
    
    scalers_dict = {}
    target_col_idx = None
    
    for sid in region_sids:
        scaler_path = f'data/normalized/scalers_{sid}.pkl'
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scalers_dict[sid] = pickle.load(f).get('pm25')
        else:
            scalers_dict[sid] = None
            
        df = pd.read_csv(f'data/normalized/norm_station_{sid}.csv', parse_dates=['timestamp']).sort_values('timestamp')
        
        # Chronological Split (Train: 2023-24, Val: T1-T4 2025, Test: Từ T5 2025)
        ts = df['timestamp']
        split_flags = np.full(len(df), 'test', dtype='U5')
        split_flags[ts < '2025-01-01'] = 'train'
        split_flags[(ts >= '2025-01-01') & (ts < '2025-05-01')] = 'val'
        
        drop_cols = ['timestamp', 'split', 'province', 'district', 'station_id']
        feature_cols = [c for c in df.columns if c not in drop_cols]
        
        if target_col_idx is None:
            target_col_idx = feature_cols.index('pm25')
            
        data_matrix = df[feature_cols].values
        
        # One-Hot Encoding Tọa độ Trạm vào XGBoost
        station_one_hot = np.zeros(33)
        station_one_hot[sid] = 1.0
        
        total_length = len(data_matrix)
        for i in range(total_length - seq_len - horizon):
            # Input X: Lấy t-24h của TẤT CẢ các cột và Băm phẳng 1D Array
            window = data_matrix[i : i + seq_len]
            x_flat = window.flatten() 
            
            x_feature = np.concatenate([x_flat, station_one_hot])
            y_target = data_matrix[i + seq_len + horizon - 1, target_col_idx]
            current_split = split_flags[i + seq_len + horizon - 1]
            
            if current_split in ['train', 'val']:
                all_X_train.append(x_feature)
                all_y_train.append(y_target)
            else:
                all_X_test.append(x_feature)
                all_y_test.append((y_target, sid))
                
    return np.array(all_X_train), np.array(all_y_train), np.array(all_X_test), all_y_test, scalers_dict

def run_xgboost_region(region_name, sids):
    results = []
    
    for h in HORIZONS:
        print(f"  [{region_name.upper()}] T+{h:<2d} | Đang tạo dataset và huấn luyện XGBoost...")
        X_train, y_train, X_test, test_tuples, scalers_dict = create_xgboost_dataset(sids, horizon=h, seq_len=24)
        
        # XGBoost Tree_method Hist cho tốc độ xé gió
        xgb_model = xgb.XGBRegressor(
            n_estimators=100, 
            max_depth=6, 
            learning_rate=0.08,
            objective='reg:squarederror',
            tree_method='hist',
            n_jobs=-1,
            random_state=42
        )
        
        xgb_model.fit(X_train, y_train)
        y_pred_scaled = xgb_model.predict(X_test)
        
        y_true_real = []
        y_pred_real = []
        
        for i, (scaled_true, sid) in enumerate(test_tuples):
            sc_tuple = scalers_dict[sid]
            real_t = inverse_pm25([scaled_true], sc_tuple)[0]
            real_p = inverse_pm25([y_pred_scaled[i]], sc_tuple)[0]
            y_true_real.append(real_t)
            y_pred_real.append(real_p)
            
        rmse, mae, r2, mape = get_metrics(np.array(y_true_real), np.array(y_pred_real))
        print(f"    -> RMSE={rmse:.2f} | MAE={mae:.2f} | R2={r2*100:.2f}% | MAPE={mape:.2f}%")
        
        results.append({
            'region': region_name, 'horizon': f'T+{h}',
            'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape,
            'n_test': len(y_true_real)
        })
        
    return results

def run_all():
    print("=" * 70)
    print("  S02: ML Tree (XGBoost) Baseline (Fair Benchmark S09)")
    print("  Horizons: T+1, T+3, T+6, T+12, T+24")
    print("  Chronological Split: Test từ 2025-05-01")
    print("=" * 70)
    
    all_results = []
    
    for r_name, r_sids in REGIONS.items():
        print(f"\n[{r_name.upper()}] Processing {len(r_sids)} stations...")
        res = run_xgboost_region(r_name, r_sids)
        all_results.extend(res)
        
    # Summary
    print("\n" + "=" * 70)
    print("  FINAL BENCHMARK — S02 ML Tree (XGBoost)")
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
    run_all()
