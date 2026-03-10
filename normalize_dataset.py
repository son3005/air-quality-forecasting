"""
normalize_dataset.py

Chuẩn hoá dataset theo chiến lược per-feature-group, per-station.
Pipeline: Clip → log1p → Scale (RobustScaler / StandardScaler / MinMaxScaler)

Input:  data/clean/clean_station_{id}.csv
Output: data/normalized/norm_station_{id}.csv
        data/normalized/scalers_{id}.pkl  (scalers để inverse-transform sau)
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

warnings.filterwarnings('ignore')

# ─── Cấu hình ─────────────────────────────────────────────────────────────────
SELECTED_STATIONS = [1, 3, 4, 7, 9, 12, 13, 15, 16, 17, 18, 24, 27, 29, 31, 32]
CLEAN_DIR  = 'data/clean'
OUTPUT_DIR = 'data/normalized'

# Train/Val/Test split (chronological)
# Dieu chinh de dat gan 70/14/16%:
#   TRAIN_END='2024-12-31' -> 24 thang / 35 = 68.6%
#   VAL_END  ='2025-04-30' -> them 4 thang  -> val = 11.4%
#   TEST     : con lai     -> 7 thang       -> test = 20%
TRAIN_END = '2024-12-31'   # ~68.6% (tu 2024-09-30)
VAL_END   = '2025-04-30'   # ~11.4%  (tu 2025-02-28)
# Test: phan con lai ~20%

# ─── Chiến lược per-feature ───────────────────────────────────────────────────

# Nhóm 1: log1p → StandardScaler
LOG_STANDARD = ['co', 'oxidation_potential', 'pollution_load']

# Nhóm 2: log1p → RobustScaler
LOG_ROBUST = ['so2', 'humid_sulfate_risk']

# Nhóm 3: Clip 99th pct → log1p → MinMaxScaler [0,1]
CLIP_LOG_MINMAX = ['precip', 'dust_source_potential']

# Nhóm 4: Clip p99.5 → log1p → RobustScaler  [MO RONG]
# pm25, pm10, no2: phan phoi heavy-tailed (skew=7-8), can log1p de nen tail
# Clip p99.5 truoc de loai sensor error con sot (no2 max=310 la bat thuong)
CLIP_LOG_ROBUST = ['pm25', 'pm10', 'no2']

# Nhóm 5: RobustScaler (khong co log)
# Xoa pm25, pm10, no2 (da chuyen sang CLIP_LOG_ROBUST)
ROBUST_ONLY = ['o3', 'wind_spd', 'wind_gusts']

# Nhóm 6: StandardScaler
STANDARD_ONLY = ['temp', 'dewpt', 'thermal_stability', 'soil_temp_0_7',
                 'no2_so2_log_diff']
# Ghi chu: 'dpd' da bi xoa khoi ENGINEERED_COLS (r=-0.993 voi rh)

# Nhóm 7: MinMaxScaler [0,1]
# Ghi chu: 'stagnation_index' da bi xoa khoi pipeline (r=-0.932 vs wind_spd)
MINMAX_ONLY = ['rh', 'clouds', 'soil_moist_0_7']

# Nhóm 8: Cyclic encoding (sin/cos) — không cần scaler
CYCLIC = ['wind_dir']

# Không scale: flags, metadata, timestamp
NO_SCALE = ['is_frozen', 'is_outlier', 'is_pm25_sensor_error',
            'station_id', 'province', 'district', 'timestamp']


# ─── Hàm chuẩn hoá 1 trạm ─────────────────────────────────────────────────────
def normalize_station(station_id: int) -> pd.DataFrame:
    path = f'{CLEAN_DIR}/clean_station_{station_id}.csv'
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df = df.set_index('timestamp').sort_index()

    province = df['province'].iloc[0]
    district = df['district'].iloc[0]
    print(f'\nTram {station_id:2d} | {province} - {district} | {len(df):,} rows')

    # Train mask (fit scaler chỉ trên train)
    train_mask = df.index <= TRAIN_END

    df_out = df.copy()
    scalers = {}

    # ── Bước 0: Cyclic encoding wind_dir ──────────────────────────────────────
    if 'wind_dir' in df.columns:
        rad = np.deg2rad(df['wind_dir'])
        df_out['wind_sin'] = np.sin(rad)
        df_out['wind_cos'] = np.cos(rad)
        df_out.drop(columns=['wind_dir'], inplace=True)

    def _fit_transform(col, scaler, df_in):
        """Fit trên train, transform tất cả."""
        vals = df_in[[col]].values
        train_vals = df_in.loc[train_mask, [col]].values
        if train_vals.shape[0] == 0:
            return vals.flatten()
        scaler.fit(train_vals)
        return scaler.transform(vals).flatten()

    # ── Nhóm 1: log1p → StandardScaler ────────────────────────────────────────
    for col in LOG_STANDARD:
        if col not in df_out.columns:
            continue
        df_out[col] = np.log1p(df_out[col].clip(lower=0))
        sc = StandardScaler()
        df_out[col] = _fit_transform(col, sc, df_out)
        scalers[col] = ('log1p+standard', sc)

    # ── Nhóm 2: log1p → RobustScaler ──────────────────────────────────────────
    for col in LOG_ROBUST:
        if col not in df_out.columns:
            continue
        df_out[col] = np.log1p(df_out[col].clip(lower=0))
        sc = RobustScaler()
        df_out[col] = _fit_transform(col, sc, df_out)
        scalers[col] = ('log1p+robust', sc)

    # ── Nhóm 3: Clip 99th → log1p → MinMaxScaler ──────────────────────────────
    for col in CLIP_LOG_MINMAX:
        if col not in df_out.columns:
            continue
        q99 = df_out.loc[train_mask, col].quantile(0.99) if train_mask.sum() > 0 \
              else df_out[col].quantile(0.99)
        df_out[col] = np.log1p(df_out[col].clip(lower=0, upper=q99))
        sc = MinMaxScaler()
        df_out[col] = _fit_transform(col, sc, df_out)
        scalers[col] = ('clip_log1p+minmax', sc, q99)

    # ── Nhóm 4: Clip p99.5 → log1p → RobustScaler  [CLIP_LOG_ROBUST] ─────────
    # Danh cho: pm25, pm10, no2 — phan phoi heavy-tailed (skew=7-8 truoc fix)
    # Buoc 1: clip p99.5 (tinh tren train) de loai extreme outlier / sensor error
    # Buoc 2: log1p de nen tail (giam skew tu 7-8 xuong ~1-2)
    # Buoc 3: RobustScaler de chuan hoa cuoi cung
    for col in CLIP_LOG_ROBUST:
        if col not in df_out.columns:
            continue
        # Clip threshold tinh tren tap train de tranh future leakage
        q995 = df_out.loc[train_mask, col].quantile(0.995) if train_mask.sum() > 0 \
               else df_out[col].quantile(0.995)
        df_out[col] = np.log1p(df_out[col].clip(lower=0, upper=q995))
        sc = RobustScaler()
        df_out[col] = _fit_transform(col, sc, df_out)
        scalers[col] = ('clip_p995_log1p+robust', sc, q995)

    # ── Nhóm 5: RobustScaler ──────────────────────────────────────────────────
    for col in ROBUST_ONLY:
        if col not in df_out.columns:
            continue
        sc = RobustScaler()
        df_out[col] = _fit_transform(col, sc, df_out)
        scalers[col] = ('robust', sc)

    # ── Nhóm 6: StandardScaler ────────────────────────────────────────────────
    for col in STANDARD_ONLY:
        if col not in df_out.columns:
            continue
        sc = StandardScaler()
        df_out[col] = _fit_transform(col, sc, df_out)
        scalers[col] = ('standard', sc)

    # ── Nhóm 7: MinMaxScaler ──────────────────────────────────────────────────
    for col in MINMAX_ONLY:
        if col not in df_out.columns:
            continue
        sc = MinMaxScaler()
        df_out[col] = _fit_transform(col, sc, df_out)
        scalers[col] = ('minmax', sc)

    # Thêm split label
    df_out['split'] = 'test'
    df_out.loc[df_out.index <= TRAIN_END, 'split'] = 'train'
    df_out.loc[(df_out.index > TRAIN_END) & (df_out.index <= VAL_END), 'split'] = 'val'

    # Report
    n_train = (df_out['split'] == 'train').sum()
    n_val   = (df_out['split'] == 'val').sum()
    n_test  = (df_out['split'] == 'test').sum()
    print(f'  Split — train:{n_train:,} val:{n_val:,} test:{n_test:,}')
    print(f'  Scaled {len(scalers)} features | Cols: {len(df_out.columns)}')

    return df_out, scalers


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('=' * 65)
    print('  NORMALIZATION PIPELINE — 16 TRAM')
    print(f'  Train: <= {TRAIN_END}')
    print(f'  Val:   {TRAIN_END} < x <= {VAL_END}')
    print(f'  Test:  > {VAL_END}')
    print('=' * 65)

    results = []
    errors  = []

    for sid in SELECTED_STATIONS:
        try:
            df_norm, scalers = normalize_station(sid)

            # Lưu normalized CSV
            out_csv = f'{OUTPUT_DIR}/norm_station_{sid}.csv'
            df_norm.reset_index().to_csv(out_csv, index=False)

            # Lưu scalers (để inverse-transform khi dự đoán)
            out_pkl = f'{OUTPUT_DIR}/scalers_{sid}.pkl'
            with open(out_pkl, 'wb') as f:
                pickle.dump(scalers, f)

            size_mb = os.path.getsize(out_csv) / (1024 * 1024)
            results.append({'sid': sid, 'rows': len(df_norm),
                            'cols': len(df_norm.columns), 'mb': size_mb})
            print(f'  Saved: {out_csv} ({size_mb:.2f} MB)')

        except Exception as e:
            errors.append((sid, str(e)))
            print(f'  ERR Tram {sid}: {e}')
            import traceback; traceback.print_exc()

    print('\n' + '=' * 65)
    print('  TONG KET')
    for r in results:
        print(f'  OK  Tram {r["sid"]:2d} | {r["rows"]:,} rows | '
              f'{r["cols"]} cols | {r["mb"]:.2f} MB')
    if errors:
        for sid, msg in errors:
            print(f'  ERR Tram {sid}: {msg}')
    print(f'\n  {len(results)}/{len(SELECTED_STATIONS)} tram thanh cong.')
    print(f'  Output: {os.path.abspath(OUTPUT_DIR)}')


if __name__ == '__main__':
    main()
