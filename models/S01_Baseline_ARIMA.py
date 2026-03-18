import os
import pandas as pd
import numpy as np
import pickle
import warnings
from statsmodels.tsa.arima.model import ARIMA
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

def run_arima_baseline(station_id, order=(2, 0, 2)):
    # 1. Đọc Dataset
    df = pd.read_csv(f'data/normalized/norm_station_{station_id}.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Chronological Split (Giống S09, S10)
    ts = df['timestamp']
    is_train = ts < '2025-01-01'
    is_val   = (ts >= '2025-01-01') & (ts < '2025-05-01')
    is_test  = ts >= '2025-05-01'
    
    scaler_path = f'data/normalized/scalers_{station_id}.pkl'
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
        sc_tuple = scalers.get('pm25')
    else:
        sc_tuple = None

    pm25_all = df['pm25'].values
    train_val_data = pm25_all[is_train | is_val]
    test_data = pm25_all[is_test]
    
    # Lấy 60 ngày cuối của train+val làm history gốc để fit (24*60 = 1440 giờ)
    history = list(train_val_data[-1440:])
    
    # 2. Khởi tạo & Fit ARIMA Core
    print(f"    [*] Trạm {station_id:02d} | Đang fit ARIMA{order} và Rolling Forecast (N={len(test_data)})...")
    try:
        model = ARIMA(history, order=order)
        fitted = model.fit()
    except Exception as e:
        print(f"    [!] Trạm {station_id:02d} Fit thất bại: {e}")
        return None
    
    # 3. Rolling Forecast
    preds = {h: [] for h in HORIZONS}
    actuals = {h: [] for h in HORIZONS}
    
    test_eval_len = len(test_data) - max(HORIZONS)
    for t in range(test_eval_len):
        forecast = fitted.forecast(steps=max(HORIZONS))
        
        for h in HORIZONS:
            preds[h].append(forecast[h-1])
            actuals[h].append(test_data[t + h - 1])
            
        # Nạp observation thật để dịch step tới, KHÔNG fit lại
        fitted = fitted.append([test_data[t]], refit=False)
        
    # 4. Inverse & Lọc array
    res_metrics = {}
    for h in HORIZONS:
        p_real = inverse_pm25(preds[h], sc_tuple)
        a_real = inverse_pm25(actuals[h], sc_tuple)
        res_metrics[h] = (a_real, p_real)
        
    return res_metrics

# ══════════════════════════════════════════════════════════════════════════
# ĐÁNH GIÁ THEO REGION
# ══════════════════════════════════════════════════════════════════════════
def run_all():
    print("=" * 70)
    print("  S01: ARIMA Baseline (Fair Chronological Benchmark)")
    print("  Horizons: T+1, T+3, T+6, T+12, T+24")
    print("  Note: Rolling forecast without refit is used for speed.")
    print("=" * 70)
    
    all_results = []
    
    for region_name, sids in REGIONS.items():
        print(f"\n[{region_name.upper()}] Processing {len(sids)} stations...")
        
        region_preds = {h: [] for h in HORIZONS}
        region_actuals = {h: [] for h in HORIZONS}
        
        for sid in sids:
            res = run_arima_baseline(sid, order=(2, 0, 2))
            if res is not None:
                for h in HORIZONS:
                    region_actuals[h].extend(res[h][0])
                    region_preds[h].extend(res[h][1])
                    
        for h in HORIZONS:
            a_arr = np.array(region_actuals[h])
            p_arr = np.array(region_preds[h])
            rmse, mae, r2, mape = get_metrics(a_arr, p_arr)
            
            print(f"  [{region_name.upper()}] T+{h:<2d} | RMSE={rmse:.2f} | MAE={mae:.2f} | R2={r2*100:.2f}% | MAPE={mape:.2f}%")
            all_results.append({
                'region': region_name, 'horizon': f'T+{h}',
                'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape,
                'n_test': len(a_arr)
            })

    # Summary
    print("\n" + "=" * 70)
    print("  FINAL BENCHMARK — S01 ARIMA")
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
