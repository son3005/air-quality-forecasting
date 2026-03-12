"""
run_pipeline_16stations.py

Chạy preprocessing pipeline cho 16 trạm đã chọn.
Output: chỉ giữ Raw inputs + 8 engineered features mới.
Lưu mỗi trạm vào data/clean/clean_station_{id}.csv
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from preprocessing import (
    WEATHER_COLUMN_MAP,
    create_wind_components,
    validate_time_range,
    detect_frozen_data,
    detect_outliers,
    detect_pm25_sensor_error,
    remove_duplicates,
    impute_missing_values,
    create_weather_features,    # 6 engineered weather/pollution features
    create_time_features,       # V3: is_weekend_holiday + cyclic time
)

# ─── Cấu hình ─────────────────────────────────────────────────────────────────
SELECTED_STATIONS = [1, 3, 4, 7, 9, 12, 13, 15, 16, 17, 18, 24, 27, 29, 31, 32]

AIR_DIR     = 'data/raw/air'
WEATHER_DIR = 'data/raw/weather'
INFO_PATH   = 'data/info.csv'
OUTPUT_DIR  = 'data/clean'

# ─── Cột đầu ra cuối cùng ──────────────────────────────────────────────────────
# Raw inputs (AQI bị loại bỏ — là hàm của PM2.5, gây data leakage khi training)
RAW_INPUT_COLS = [
    # Chất lượng không khí — inputs
    'pm25', 'pm10', 'co', 'o3', 'no2', 'so2',
    # Khí tượng cơ bản
    'temp', 'rh', 'dewpt', 'precip', 'clouds',
    'wind_spd', 'wind_dir', 'wind_gusts',
    # Đất (dùng để tính features)
    'soil_temp_0_7', 'soil_moist_0_7',
]

# 8 Engineered features (giu lai 6 sau khi xoa dpd va stagnation_index)
ENGINEERED_COLS = [
    'oxidation_potential',   # O3 x (SO2 + NO2)
    'pollution_load',        # CO + SO2 + NO2
    'no2_so2_log_diff',      # log(NO2+1) - log(SO2+1)
    'humid_sulfate_risk',    # RH x SO2
    # 'dpd' da xoa: r=-0.993 voi rh, redundant hoan toan
    'thermal_stability',     # Temp - Soil Temp (0-7cm)
    # 'stagnation_index' da xoa: r=-0.932 voi wind_spd, r=-0.966 voi dust_source_potential
    'dust_source_potential', # Wind Speed / (Soil Moisture + 1)
]

# Quality flags + V3 binary flags (giữ để traceable)
FLAG_COLS = ['is_frozen', 'is_outlier', 'is_pm25_sensor_error',
             'is_weekend_holiday',      # V3: ngay le VN chinh thuc
             'is_extreme_pm25_1h_ago']  # V3: spike PM2.5 cuc doan

# Metadata
META_COLS = ['station_id', 'province', 'district']

FINAL_COLS = RAW_INPUT_COLS + ENGINEERED_COLS + FLAG_COLS + META_COLS


# ─── Load station info ─────────────────────────────────────────────────────────
info_df = pd.read_csv(INFO_PATH)
info_df = info_df[info_df['station'].isin(SELECTED_STATIONS)].set_index('station')


# ─── Hàm xử lý 1 trạm ────────────────────────────────────────────────────────
def process_station(station_id: int) -> pd.DataFrame:
    province = info_df.loc[station_id, 'province']
    district = info_df.loc[station_id, 'district']
    location = f"{province}_{district}"

    print(f"\n{'='*60}")
    print(f"Tram {station_id:2d} | {province} - {district}")
    print(f"{'='*60}")

    # Load air quality
    df_air = pd.read_csv(f'{AIR_DIR}/air_{station_id}.csv')
    df_air['timestamp_local'] = pd.to_datetime(df_air['timestamp_local'])
    df_air = df_air.sort_values('timestamp_local').reset_index(drop=True)

    aq_cols = ['timestamp_local', 'aqi', 'co', 'no2', 'o3', 'pm10', 'pm25', 'so2']
    df_air = df_air[[c for c in aq_cols if c in df_air.columns]]

    # Load weather
    df_wth = pd.read_csv(f'{WEATHER_DIR}/weather_{station_id}.csv')
    df_wth = df_wth.rename(columns=WEATHER_COLUMN_MAP)
    df_wth['timestamp_local'] = pd.to_datetime(df_wth['timestamp_local'])
    df_wth = df_wth.sort_values('timestamp_local').reset_index(drop=True)

    # Merge
    df = pd.merge(df_air, df_wth, on='timestamp_local', how='outer')
    df = df.drop_duplicates(subset=['timestamp_local'], keep='first')

    # Pipeline
    df = create_wind_components(df)
    df = validate_time_range(df, location)
    df = detect_frozen_data(df)
    df = detect_outliers(df)
    df = detect_pm25_sensor_error(df)
    df = remove_duplicates(df)
    df = impute_missing_values(df, location)

    # Chỉ tạo 8 engineered weather/pollution features
    df = create_weather_features(df)

    # V3: Thêm is_weekend_holiday từ create_time_features()
    # Gọi sau impute để index đã là DatetimeIndex được validate
    df = create_time_features(df)

    # V3: is_extreme_pm25_1h_ago — giữ signal spike cực đoan
    # Ngưỡng 75 µg/m³ = WHO Unhealthy level (hourly)
    if 'pm25' in df.columns:
        df['is_extreme_pm25_1h_ago'] = (
            df['pm25'].shift(1).bfill() > 75.0
        ).astype(int)

    # Thêm metadata
    df['station_id'] = station_id
    df['province']   = province
    df['district']   = district

    # Chỉ giữ các cột đã định nghĩa
    keep_cols = [c for c in FINAL_COLS if c in df.columns]
    missing   = [c for c in FINAL_COLS if c not in df.columns]
    if missing:
        print(f"   [WARN] Thieu cot: {missing}")
    df = df[keep_cols]

    # Reset index về cột timestamp
    df = df.reset_index(names='timestamp')

    print(f"   Shape cuoi: {df.shape}")
    print(f"   Cot: {list(df.columns)}")
    n_err = df['is_pm25_sensor_error'].sum() if 'is_pm25_sensor_error' in df.columns else 0
    print(f"   Sensor errors: {n_err}")

    return df


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("  PREPROCESSING PIPELINE — 16 TRAM DA CHON")
    print("="*60)
    print(f"  Stations: {SELECTED_STATIONS}")
    print(f"  Output: {OUTPUT_DIR}/clean_station_{{id}}.csv")
    print(f"  Output columns: {len(FINAL_COLS)} cols")
    print("="*60)

    results = {}
    errors  = []

    for sid in SELECTED_STATIONS:
        try:
            df = process_station(sid)
            out_path = f'{OUTPUT_DIR}/clean_station_{sid}.csv'
            df.to_csv(out_path, index=False)
            size_mb = os.path.getsize(out_path) / (1024*1024)
            results[sid] = {'rows': len(df), 'cols': len(df.columns), 'size_mb': size_mb}
            print(f"   Luu: {out_path}  ({size_mb:.2f} MB)")
        except Exception as e:
            errors.append((sid, str(e)))
            print(f"   LOI tram {sid}: {e}")

    # Summary
    print("\n" + "="*60)
    print("  TONG KET")
    print("="*60)
    for sid, info in results.items():
        prov = info_df.loc[sid, 'province']
        dist = info_df.loc[sid, 'district']
        print(f"  OK  Tram {sid:2d} | {prov:12s} - {dist:15s} | {info['rows']:,} rows | {info['size_mb']:.2f} MB")
    if errors:
        for sid, msg in errors:
            print(f"  ERR Tram {sid}: {msg}")
    print(f"\n  {len(results)}/{len(SELECTED_STATIONS)} tram thanh cong.")
    print(f"  Output dir: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == '__main__':
    main()
