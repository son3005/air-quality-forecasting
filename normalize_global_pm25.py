"""
normalize_global_pm25.py
~~~~~~~~~~~~~~~~~~~~~~~~
Tạo bộ normalized data mới với PM2.5 dùng GLOBAL scaler (fit trên tất cả 16 trạm).
Các features khác vẫn normalize per-station như bình thường.

Input:  data/clean/clean_station_{id}.csv
Output: data/normalized_global/norm_station_{id}.csv
        data/normalized_global/pm25_global_scaler.pkl
        data/normalized_global/scalers_{id}.pkl (non-pm25 per-station scalers)

Lý do: Neighbor PM2.5 features cần cùng scale để XGBoost học được spatial signal đúng.
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

warnings.filterwarnings('ignore')

SELECTED_STATIONS = [1, 3, 4, 7, 9, 12, 13, 15, 16, 17, 18, 24, 27, 29, 31, 32]
CLEAN_DIR   = 'data/clean'
INPUT_DIR   = 'data/normalized'    # dùng để lấy per-station scalers các features khác
OUTPUT_DIR  = 'data/normalized_global'

TRAIN_END = '2024-12-31'
VAL_END   = '2025-04-30'

print("="*65)
print("  GLOBAL PM2.5 NORMALIZATION PIPELINE")
print("="*65)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Bước 1: Gộp PM2.5 log1p từ tất cả trạm để fit global scaler ──────────────
print("\n[1] Fitting Global PM2.5 RobustScaler on all 16 stations (train only)...")
all_train_pm25 = []

for sid in SELECTED_STATIONS:
    path = os.path.join(CLEAN_DIR, f'clean_station_{sid}.csv')
    df = pd.read_csv(path, parse_dates=['timestamp']).set_index('timestamp').sort_index()
    train_mask = df.index <= TRAIN_END
    pm25_train = df.loc[train_mask, 'pm25'].clip(lower=0).values
    pm25_log   = np.log1p(pm25_train)
    all_train_pm25.append(pm25_log)
    print(f"  Station {sid:2d}: {len(pm25_log):,} train rows | "
          f"PM2.5 mean={pm25_train.mean():.2f} max={pm25_train.max():.1f} ug/m3")

# Stack tất cả train PM2.5 → fit 1 scaler chung
all_train_pool = np.concatenate(all_train_pm25).reshape(-1, 1)
global_scaler = RobustScaler()
global_scaler.fit(all_train_pool)

print(f"\n[Global Scaler] center_={global_scaler.center_[0]:.4f} | scale_={global_scaler.scale_[0]:.4f}")
print(f"  (This means log1p(PM2.5) will be normalized with shared median+IQR)")

# Lưu global scaler
global_scaler_path = os.path.join(OUTPUT_DIR, 'pm25_global_scaler.pkl')
with open(global_scaler_path, 'wb') as f:
    pickle.dump(global_scaler, f)
print(f"[+] Saved global scaler to {global_scaler_path}")

# ─── Bước 2: Tạo normalized files với global PM2.5 scale ────────────────────
print("\n[2] Re-normalizing all stations with global PM2.5 scaler...")

results = []
for sid in SELECTED_STATIONS:
    # Load clean data
    clean_path = os.path.join(CLEAN_DIR, f'clean_station_{sid}.csv')
    df = pd.read_csv(clean_path, parse_dates=['timestamp']).set_index('timestamp').sort_index()
    
    # Load per-station scalers từ normalized hiện tại (cho các features khác)
    per_station_scaler_path = os.path.join(INPUT_DIR, f'scalers_{sid}.pkl')
    with open(per_station_scaler_path, 'rb') as f:
        per_station_scalers = pickle.load(f)
    
    # Load normalized CSV (đã có tất cả features scaled except PM2.5)
    norm_path = os.path.join(INPUT_DIR, f'norm_station_{sid}.csv')
    df_norm = pd.read_csv(norm_path, parse_dates=['timestamp']).set_index('timestamp').sort_index()
    
    # Thay thế PM2.5 với global-normalized version
    pm25_log = np.log1p(df['pm25'].clip(lower=0).values.reshape(-1, 1))
    pm25_global_norm = global_scaler.transform(pm25_log).flatten()
    
    df_out = df_norm.copy()
    df_out['pm25'] = pm25_global_norm
    
    # Verify stats
    train_mask = df_out['split'] == 'train'
    pm25_train_norm = df_out.loc[train_mask, 'pm25']
    print(f"  Station {sid:2d}: pm25_norm mean={pm25_train_norm.mean():.3f} "
          f"std={pm25_train_norm.std():.3f} "
          f"[min={pm25_train_norm.min():.2f}, max={pm25_train_norm.max():.2f}]")
    
    # Lưu file
    out_path = os.path.join(OUTPUT_DIR, f'norm_station_{sid}.csv')
    df_out.reset_index().to_csv(out_path, index=False)
    
    # Lưu scalers (kết hợp per-station + global pm25 scaler ref)
    out_scalers = per_station_scalers.copy()
    out_scalers['pm25'] = ('log1p+global_robust', global_scaler)  # override với global scaler
    out_pkl = os.path.join(OUTPUT_DIR, f'scalers_{sid}.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump(out_scalers, f)
    
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    results.append({'sid': sid, 'rows': len(df_out), 'mb': size_mb})

print("\n" + "="*65)
print("  TONG KET GLOBAL NORMALIZATION")
print("="*65)
for r in results:
    print(f"  OK  Tram {r['sid']:2d} | {r['rows']:,} rows | {r['mb']:.2f} MB")

print(f"\n[+] All {len(results)} stations normalized with GLOBAL PM2.5 scale.")
print(f"[+] Output dir: {os.path.abspath(OUTPUT_DIR)}")
print("\nGlobal Scaler Summary:")
print(f"  log1p(PM2.5) center (median): {global_scaler.center_[0]:.4f}")
print(f"  log1p(PM2.5) scale (IQR):     {global_scaler.scale_[0]:.4f}")
