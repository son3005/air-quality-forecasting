"""
build_dataset.py

Unified Data Pipeline
Gộp 3 bước: Preprocessing (Raw -> Clean), Normalization (Clean -> Norm), và Splitting (Norm -> Split).
Toàn bộ parameter điều khiển nằm ở block CONFIGURATION bên dưới để dễ dàng quản lý.
"""

import os
import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

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
    create_weather_features,
    create_time_features,
)

# =============================================================================
# 1. GLOBAL CONFIGURATION
# =============================================================================

# --- A. General Settings ---
SELECTED_STATIONS = [1, 4, 5, 16, 17, 27, 7, 18, 24, 30, 31, 32]
AIR_DIR     = 'data/raw/air'
WEATHER_DIR = 'data/raw/weather'
INFO_PATH   = 'data/info.csv'
CLEAN_DIR   = 'data/clean'
NORM_DIR    = 'data/normalized'
SPLIT_DIR   = 'data/split'

# --- B. Normalization Data Split Boundaries ---
# Chỉ fit scaler trên tập Train, sau đó transform cho toàn bộ
TRAIN_END = '2024-12-31'
VAL_END   = '2025-04-30'

# --- C. Final Dataset Splitting Configs ---
# Chiến lược chia block days (chu kỳ Lặp lại)
SPLIT_CONFIGS = {
    'block5': {'total': 5, 'train': 3, 'val': 1, 'test': 1},
    'block7': {'total': 7, 'train': 5, 'val': 1, 'test': 1},
    'block30': {'total': 30, 'train': 20, 'val': 5, 'test': 5}
}

# --- D. Preprocessing Feature Columns ---
RAW_INPUT_COLS = [
    'pm25', 'pm10', 'co', 'o3', 'no2', 'so2',
    'temp', 'rh', 'dewpt', 'precip', 'clouds',
    'wind_spd', 'wind_dir', 'wind_gusts',
    'soil_temp_0_7', 'soil_moist_0_7',
]

ENGINEERED_COLS = [
    'oxidation_potential', 'pollution_load', 'no2_so2_log_diff',
    'humid_sulfate_risk', 'thermal_stability', 'dust_source_potential',
]

FLAG_COLS = [
    'is_frozen', 'is_outlier', 'is_pm25_sensor_error',
    'is_weekend_holiday', 'is_extreme_pm25_1h_ago'
]

LAG_COLS = [
    'pm25_lag_1', 'pm25_lag_3', 'pm25_lag_6', 'pm25_lag_12', 'pm25_lag_24',
    'pm25_roll_mean_6', 'pm25_roll_mean_12', 'pm25_roll_std_6',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
]
META_COLS = ['station_id', 'province', 'district']
FINAL_COLS = RAW_INPUT_COLS + ENGINEERED_COLS + FLAG_COLS + META_COLS + LAG_COLS

# --- E. Normalization Feature Groups ---
LOG_STANDARD            = ['co', 'oxidation_potential', 'pollution_load']
LOG_ROBUST              = ['so2', 'humid_sulfate_risk']
CLIP_LOG_MINMAX         = ['precip', 'dust_source_potential']
LOG_ROBUST_ONLY_NO_CLIP = ['pm25'] # Target variable không clip (nhưng vẫn có is_extreme_pm25 flag đính kèm)
CLIP_LOG_ROBUST         = ['pm10', 'no2']
ROBUST_ONLY             = ['o3', 'wind_spd', 'wind_gusts']
STANDARD_ONLY           = ['temp', 'dewpt', 'thermal_stability', 'soil_temp_0_7', 'no2_so2_log_diff']
MINMAX_ONLY             = ['rh', 'clouds', 'soil_moist_0_7']
PM25_LAG_COLS           = ['pm25_lag_1', 'pm25_lag_3', 'pm25_lag_6', 'pm25_lag_12', 'pm25_lag_24']
PM25_ROLL_MEAN          = ['pm25_roll_mean_6', 'pm25_roll_mean_12']
PM25_ROLL_STD           = ['pm25_roll_std_6']
# Features không scale, giữ nguyên giá trị
NO_SCALE_COLS           = FLAG_COLS + META_COLS + ['timestamp', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

info_df = pd.read_csv(INFO_PATH).set_index('station')

def create_pm25_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pm25 = df['pm25']
    for k in [1, 3, 6, 12, 24]:
        df[f'pm25_lag_{k}'] = pm25.shift(k).bfill()
    
    df['pm25_roll_mean_6']  = pm25.rolling(window=6,  min_periods=1).mean()
    df['pm25_roll_mean_12'] = pm25.rolling(window=12, min_periods=1).mean()
    df['pm25_roll_std_6']   = pm25.rolling(window=6,  min_periods=2).std().fillna(0)
    
    if isinstance(df.index, pd.DatetimeIndex):
        hour, month = df.index.hour, df.index.month
    else:
        ts = pd.to_datetime(df.index)
        hour, month = ts.hour, ts.month
        
    df['hour_sin']  = np.sin(2 * np.pi * hour  / 24)
    df['hour_cos']  = np.cos(2 * np.pi * hour  / 24)
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    return df

# =============================================================================
# 3. PIPELINE STEPS
# =============================================================================

def step1_preprocess(station_id: int):
    province = info_df.loc[station_id, 'province']
    district = info_df.loc[station_id, 'district']
    location = f"{province}_{district}"
    
    df_air = pd.read_csv(f'{AIR_DIR}/air_{station_id}.csv')
    df_air['timestamp_local'] = pd.to_datetime(df_air['timestamp_local'])
    df_air = df_air.sort_values('timestamp_local').reset_index(drop=True)
    aq_cols = ['timestamp_local', 'aqi', 'co', 'no2', 'o3', 'pm10', 'pm25', 'so2']
    df_air = df_air[[c for c in aq_cols if c in df_air.columns]]

    df_wth = pd.read_csv(f'{WEATHER_DIR}/weather_{station_id}.csv')
    df_wth = df_wth.rename(columns=WEATHER_COLUMN_MAP)
    df_wth['timestamp_local'] = pd.to_datetime(df_wth['timestamp_local'])
    df_wth = df_wth.sort_values('timestamp_local').reset_index(drop=True)

    df = pd.merge(df_air, df_wth, on='timestamp_local', how='outer')
    df = df.drop_duplicates(subset=['timestamp_local'], keep='first')

    df = create_wind_components(df)
    df = validate_time_range(df, location)
    df = detect_frozen_data(df)
    df = detect_outliers(df)
    df = detect_pm25_sensor_error(df)
    df = remove_duplicates(df)
    df = impute_missing_values(df, location)
    df = create_weather_features(df)
    df = create_time_features(df)

    if 'pm25' in df.columns:
        df['is_extreme_pm25_1h_ago'] = (df['pm25'].shift(1).bfill() > 75.0).astype(int)

    df = create_pm25_lag_features(df)
    
    df['station_id'] = station_id
    df['province']   = province
    df['district']   = district
    
    keep_cols = [c for c in FINAL_COLS if c in df.columns]
    df = df[keep_cols].reset_index(names='timestamp')
    
    out_path = f'{CLEAN_DIR}/clean_station_{station_id}.csv'
    df.to_csv(out_path, index=False)
    return len(df), out_path


def step2_normalize(station_id: int):
    path = f'{CLEAN_DIR}/clean_station_{station_id}.csv'
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df = df.set_index('timestamp').sort_index()

    train_mask = df.index <= TRAIN_END
    df_out = df.copy()
    scalers = {}

    if 'wind_dir' in df.columns:
        rad = np.deg2rad(df['wind_dir'])
        df_out['wind_sin'] = np.sin(rad)
        df_out['wind_cos'] = np.cos(rad)
        df_out.drop(columns=['wind_dir'], inplace=True)

    def _fit_transform(col, scaler, df_in):
        vals = df_in[[col]].values
        train_vals = df_in.loc[train_mask, [col]].values
        if train_vals.shape[0] == 0: return vals.flatten()
        scaler.fit(train_vals)
        return scaler.transform(vals).flatten()

    for col in LOG_STANDARD:
        if col in df_out.columns:
            df_out[col] = np.log1p(df_out[col].clip(lower=0))
            sc = StandardScaler()
            df_out[col] = _fit_transform(col, sc, df_out)
            scalers[col] = ('log1p+standard', sc)

    for col in LOG_ROBUST + LOG_ROBUST_ONLY_NO_CLIP + PM25_LAG_COLS + PM25_ROLL_MEAN:
        if col in df_out.columns:
            df_out[col] = np.log1p(df_out[col].clip(lower=0))
            sc = RobustScaler()
            df_out[col] = _fit_transform(col, sc, df_out)
            scalers[col] = ('log1p+robust', sc)

    for col in CLIP_LOG_MINMAX:
        if col in df_out.columns:
            q99 = df_out.loc[train_mask, col].quantile(0.99) if train_mask.sum() > 0 else df_out[col].quantile(0.99)
            df_out[col] = np.log1p(df_out[col].clip(lower=0, upper=q99))
            sc = MinMaxScaler()
            df_out[col] = _fit_transform(col, sc, df_out)
            scalers[col] = ('clip_log1p+minmax', sc, q99)

    for col in CLIP_LOG_ROBUST:
        if col in df_out.columns:
            q995 = df_out.loc[train_mask, col].quantile(0.995) if train_mask.sum() > 0 else df_out[col].quantile(0.995)
            df_out[col] = np.log1p(df_out[col].clip(lower=0, upper=q995))
            sc = RobustScaler()
            df_out[col] = _fit_transform(col, sc, df_out)
            scalers[col] = ('clip_p995_log1p+robust', sc, q995)

    for col in ROBUST_ONLY + PM25_ROLL_STD:
        if col in df_out.columns:
            sc = RobustScaler()
            df_out[col] = _fit_transform(col, sc, df_out)
            scalers[col] = ('robust', sc)

    for col in STANDARD_ONLY:
        if col in df_out.columns:
            sc = StandardScaler()
            df_out[col] = _fit_transform(col, sc, df_out)
            scalers[col] = ('standard', sc)

    for col in MINMAX_ONLY:
        if col in df_out.columns:
            sc = MinMaxScaler()
            df_out[col] = _fit_transform(col, sc, df_out)
            scalers[col] = ('minmax', sc)

    df_out['split'] = 'test'
    df_out.loc[df_out.index <= TRAIN_END, 'split'] = 'train'
    df_out.loc[(df_out.index > TRAIN_END) & (df_out.index <= VAL_END), 'split'] = 'val'
    
    out_csv = f'{NORM_DIR}/norm_station_{station_id}.csv'
    df_out.reset_index().to_csv(out_csv, index=False)
    
    with open(f'{NORM_DIR}/scalers_{station_id}.pkl', 'wb') as f:
        pickle.dump(scalers, f)
        
    return len(df_out), out_csv


def step3_split():
    for cfg_name, cfg in SPLIT_CONFIGS.items():
        os.makedirs(os.path.join(SPLIT_DIR, cfg_name), exist_ok=True)

    min_time = None
    dfs = {}

    for sid in SELECTED_STATIONS:
        fpath = os.path.join(NORM_DIR, f'norm_station_{sid}.csv')
        df = pd.read_csv(fpath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        station_min = df['timestamp'].min()
        if min_time is None or station_min < min_time:
            min_time = station_min
            
        dfs[sid] = df
        
    min_date = min_time.normalize()
    print(f"\n[Step 3] Global Base Date for block splits: {min_date}")

    for cfg_name, cfg in SPLIT_CONFIGS.items():
        D = cfg['total']
        T_end = cfg['train']
        V_end = T_end + cfg['val']
        
        print(f"  -> Generating {cfg_name}...")
        for sid, df in dfs.items():
            df_out = df.copy()
            delta_days = (df_out['timestamp'] - min_date).dt.floor('D')
            pos = delta_days.dt.days % D
            
            if 'split' in df_out.columns:
                df_out = df_out.drop(columns=['split'])
                
            df_out.loc[pos < T_end, 'split'] = 'train'
            df_out.loc[(pos >= T_end) & (pos < V_end), 'split'] = 'val'
            df_out.loc[pos >= V_end, 'split'] = 'test'
            
            df_out.to_csv(os.path.join(SPLIT_DIR, cfg_name, f'station_{sid}.csv'), index=False)

# =============================================================================
# 4. MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    for d in [CLEAN_DIR, NORM_DIR, SPLIT_DIR]:
        os.makedirs(d, exist_ok=True)
        
    print("="*60)
    print(" UNIFIED DATA PIPELINE: Preprocess -> Normalize -> Split")
    print("="*60)
    print(f" Stations: {SELECTED_STATIONS}")
    print(f" Split Modes: {list(SPLIT_CONFIGS.keys())}")
    print("="*60)
    
    for sid in SELECTED_STATIONS:
        print(f"\nProcessing Station {sid}...")
        try:
            sz1, p1 = step1_preprocess(sid)
            print(f"  [Step 1] Cleaned: {sz1} rows -> {p1}")
            sz2, p2 = step2_normalize(sid)
            print(f"  [Step 2] Normalized: {sz2} rows -> {p2}")
        except Exception as e:
            print(f"  [ERROR] Station {sid} Failed: {str(e)}")
            import traceback; traceback.print_exc()
            
    try:
        step3_split()
        print("\n[SUCCESS] Pipeline Finished Successfully!")
    except Exception as e:
        print(f"\n[ERROR] Step 3 Splitting Failed: {str(e)}")
        import traceback; traceback.print_exc()

