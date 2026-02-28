"""preprocessing_clean.py
--------------------------------
Script: đọc dữ liệu từng trạm (air + weather) và thực hiện Feature Engineering nâng cao
đầu ra: data/clean/station_{i}.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Cấu hình
# ---------------------------------------------------------------------------
DATA_DIR = "data"
RAW_AIR_DIR = os.path.join(DATA_DIR, "raw", "air")
RAW_WX_DIR  = os.path.join(DATA_DIR, "raw", "weather")
OUT_DIR     = os.path.join(DATA_DIR, "clean")
INFO_FILE   = os.path.join(DATA_DIR, "info.csv")

START_DATE = pd.Timestamp("2023-01-01 00:00:00")
END_DATE   = pd.Timestamp("2025-12-01 00:00:00")

# Mapping cột weather (Open‑Meteo → tên nội bộ)
WEATHER_MAP = {
    "time": "timestamp_local",
    "temperature_2m (°C)": "temp",
    "relative_humidity_2m (%)": "rh",
    "dew_point_2m (°C)": "dewpt",
    "precipitation (mm)": "precip",
    "cloud_cover (%)": "clouds",
    "wind_speed_10m (m/s)": "wind_spd",
    "wind_direction_10m (°)": "wind_dir",
}

# ---------------------------------------------------------------------------
# Curated feature set (24 features, excluding 'time') — ordered for the model
# Mỗi station sẽ chỉ giữ những cột này theo đúng thứ tự.
# ---------------------------------------------------------------------------
# 0  aqi          — AQI tổng hợp
# 1  pm25         — PM2.5 raw
# 2  pm10         — PM10
# 3  co           — CO
# 4  no2          — NO2 (mới)
# 5  o3           — O3 (mới)
# 6  temp         — Nhiệt độ
# 7  rh           — Độ ẩm tương đối
# 8  dewpt        — Điểm sương
# 9  precip       — Lượng mưa
# 10 wind_spd     — Tốc độ gió
# 11 wind_sin     — Hướng gió (sin)
# 12 wind_cos     — Hướng gió (cos)
# 13 ah           — Độ ẩm tuyệt đối (vật lý)
# 14 dpd          — Dew Point Depression (dự báo nghịch nhiệt)
# 15 is_stagnant  — Không khí đọng (gió < 1.5 & rh > 80)
# 16 rush_hour    — Giờ cao điểm giao thông
# 17 hour_sin     — Giờ trong ngày (sin)
# 18 hour_cos     — Giờ trong ngày (cos)
# 19 month_sin    — Tháng (sin)
# 20 month_cos    — Tháng (cos)
# 21 ma_pm25_24   — Target: moving average PM2.5 24h (index 21)
# 22 delta_pm25   — Tốc độ thay đổi PM2.5
# 23 pm25_lag_1   — PM2.5 lag 1h
# 24 pm25_lag_24  — PM2.5 cùng giờ hôm qua
# 25 rain_sum_6   — Tổng mưa 6h (rửa trôi ô nhiễm)
OUTPUT_SELECTED_COLS = [
    "time",
    "aqi", "pm25", "pm10", "co", "no2", "o3",
    "temp", "rh", "dewpt", "precip", "wind_spd",
    "wind_sin", "wind_cos",
    "ah", "dpd", "is_stagnant",
    "rush_hour",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "ma_pm25_24",
    "delta_pm25", "pm25_lag_1", "pm25_lag_24", "rain_sum_6",
]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_air(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp_local"] = pd.to_datetime(df["timestamp_local"], errors="coerce")
    df = df.sort_values("timestamp_local").reset_index(drop=True)
    # Lấy cả các cột khí mới (no2, o3, so2) nếu có
    keep = ["timestamp_local", "aqi", "pm25", "pm10", "co", "no2", "o3", "so2"]
    return df[[c for c in keep if c in df.columns]]


def load_weather(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=WEATHER_MAP)
    df["timestamp_local"] = pd.to_datetime(df["timestamp_local"], errors="coerce")
    df = df.sort_values("timestamp_local").reset_index(drop=True)
    keep = ["timestamp_local", "temp", "rh", "dewpt", "precip", "clouds", "wind_spd", "wind_dir"]
    return df[[c for c in keep if c in df.columns]]


def full_hourly_index(df: pd.DataFrame) -> pd.DataFrame:
    max_ts = df["timestamp_local"].max().ceil("h")
    end = min(END_DATE, max_ts)
    full_idx = pd.date_range(start=START_DATE, end=end, freq="h")
    df = df.drop_duplicates(subset=["timestamp_local"]).set_index("timestamp_local")
    return df.reindex(full_idx)


def impute(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns
    # Phase 1: linear interpolation (max 6h gap)
    df[num_cols] = df[num_cols].interpolate(method="time", limit=6)
    # Phase 2: KNN for remaining gaps
    if df[num_cols].isnull().any().any():
        means = df[num_cols].mean()
        stds = df[num_cols].std().replace(0, 1)
        scaled = (df[num_cols] - means) / stds
        imputer = KNNImputer(n_neighbors=12, weights="distance")
        imputed = imputer.fit_transform(scaled)
        df[num_cols] = imputed * stds.values + means.values
    # Clip non-negative columns
    for col in ["aqi", "pm25", "pm10", "co", "no2", "o3", "so2", "rh", "precip", "wind_spd", "clouds"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
    return df


def generate_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo TẤT CẢ các features nâng cao theo yêu cầu."""
    
    # 1. Thời gian cơ bản
    df["hour"] = df.index.hour
    df["day"] = df.index.day
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["day_of_week"] = df.index.dayofweek
    
    # 2. Cyclic Features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    season_val = (df["month"] % 12 + 3) // 3  # 1: Spring, 2: Summer, 3: Autumn, 4: Winter
    df["season_sin"] = np.sin(2 * np.pi * season_val / 4)
    
    # 3. Wind Features
    if "wind_dir" in df.columns and "wind_spd" in df.columns:
        rad = np.deg2rad(df["wind_dir"])
        df["wind_sin"] = np.sin(rad)
        df["wind_cos"] = np.cos(rad)
        df["spd_wind_sin"] = df["wind_spd"] * df["wind_sin"]
        df["spd_wind_cos"] = df["wind_spd"] * df["wind_cos"]
    else:
        for c in ["wind_sin", "wind_cos", "spd_wind_sin", "spd_wind_cos"]:
            df[c] = 0.0

    # 4. Domain / Khí tượng
    if "temp" in df.columns and "rh" in df.columns:
        # Absolute Humidity (vật lý)
        # Bão hòa hơi nước: ew = 6.112 * exp((17.67 * t) / (t + 243.5))
        # RH = e/ew * 100 -> e = RH * ew / 100
        # AH = (e * 2.1674) / (t + 273.15) * 1000  (gam/m3) -> đơn giản xấp xỉ
        try:
            ew = 6.112 * np.exp((17.67 * df["temp"]) / (df["temp"] + 243.5))
            e = (df["rh"] / 100.0) * ew
            df["ah"] = (e * 216.7) / (df["temp"] + 273.15)
        except:
            df["ah"] = 0.0
            
    if "temp" in df.columns and "dewpt" in df.columns:
        df["dpd"] = df["temp"] - df["dewpt"] # Dew Point Depression
        df["dtr"] = df["temp"].rolling(24, min_periods=1).max() - df["temp"].rolling(24, min_periods=1).min() # Diurnal Temp Range

    if "wind_spd" in df.columns and "rh" in df.columns:
        df["is_stagnant"] = ((df["wind_spd"] < 1.5) & (df["rh"] > 80)).astype(float)

    # 5. Domain / Air Quality
    if "pm25" in df.columns and "pm10" in df.columns:
        df["ratio_pm"] = df["pm25"] / (df["pm10"] + 1e-5)
    
    if "pm25" in df.columns and "wind_spd" in df.columns:
        df["w_pm25"] = df["pm25"] / (df["wind_spd"] + 1.0) # Trọng số pm25 theo gió (mức độ thông thoáng)
        
    df["rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(float)

    # 6. PM2.5 Lags & Rolling
    if "pm25" in df.columns:
        df["ma_pm25_4"] = df["pm25"].rolling(4, min_periods=1).mean()
        df["ma_pm25_24"] = df["pm25"].rolling(24, min_periods=1).mean()
        df["delta_pm25"] = df["pm25"].diff(1).fillna(0)
        df["pm25_lag_1"] = df["pm25"].shift(1).bfill()
        df["pm25_lag_3"] = df["pm25"].shift(3).bfill()
        df["pm25_lag_6"] = df["pm25"].shift(6).bfill()
        df["pm25_lag_24"] = df["pm25"].shift(24).bfill()
        
    if "precip" in df.columns:
        df["rain_sum_6"] = df["precip"].rolling(6, min_periods=1).sum()

    return df


def detect_abnormal_and_spike(df: pd.DataFrame) -> pd.DataFrame:
    """Cập nhật để trả về cờ is_frozen và is_outlier thay vì xóa hẳn,
    cho phép model có thêm thông tin về chất lượng cảm biến.
    """
    df = df.copy()
    df["is_frozen"] = 0.0
    df["is_outlier"] = 0.0
    
    cols_to_check = ["aqi", "pm25", "pm10", "co", "no2", "o3", "so2"]
    
    for col in cols_to_check:
        if col not in df.columns: continue
        series = df[col].copy()
        
        # 1. Frozen
        diff = series.diff().fillna(1)
        is_frozen = (diff == 0)
        if is_frozen.any():
            runs = (is_frozen != is_frozen.shift()).cumsum()
            for _, grp in is_frozen.groupby(runs):
                if grp.all() and len(grp) >= 24:
                    df.loc[grp.index, "is_frozen"] = 1.0
                    df.loc[grp.index, col] = np.nan
        
        # 2. Spike (Z-Score dưa trên MAD)
        rolling_med = series.rolling(window=48, center=True, min_periods=6).median()
        rolling_mad = (series - rolling_med).abs().rolling(window=48, center=True, min_periods=6).median().replace(0, np.nan)
        modified_z = (series - rolling_med).abs() / (1.4826 * rolling_mad)
        spike_mask = modified_z > 4.0
        if spike_mask.any():
            df.loc[spike_mask, "is_outlier"] = 1.0
            df.loc[spike_mask, col] = np.nan
            
    return df


# ---------------------------------------------------------------------------
# Main per-station processing
# ---------------------------------------------------------------------------

def process_station(station_id: int) -> pd.DataFrame | None:
    aq_path = os.path.join(RAW_AIR_DIR, f"air_{station_id}.csv")
    wx_path = os.path.join(RAW_WX_DIR, f"weather_{station_id}.csv")
    if not os.path.exists(aq_path) or not os.path.exists(wx_path):
        return None

    df_air = load_air(aq_path)
    df_wx  = load_weather(wx_path)
    
    df = pd.merge(df_air, df_wx, on="timestamp_local", how="outer")
    df = full_hourly_index(df)

    # Bước 1: Phát hiện nhiễu -> NaN nhưng giữ cờ.
    df = detect_abnormal_and_spike(df)

    # Bước 2: Điền thiếu (rất quan trọng phải chạy trước khi fill các cột feature phức tạp)
    df = impute(df)
    
    # Smooth nhẹ sau impute
    for col in ["aqi", "pm25", "pm10", "co", "no2", "o3", "so2"]:
        if col in df.columns:
            df[col] = df[col].rolling(3, center=True, min_periods=1).mean()

    # Bước 3: Generate toàn bộ features phức tạp
    df = generate_advanced_features(df)

    # Rename index
    df = df.rename_axis("time").reset_index()

    # === Bước 4: Chỉ giữ lại các cột được chọn lọc (curated feature set) ===
    # Đảm bảo các cột thiếu (e.g., no2, o3 nếu station không có) được fill = 0
    for col in OUTPUT_SELECTED_COLS:
        if col != "time" and col not in df.columns:
            df[col] = 0.0
    df = df[OUTPUT_SELECTED_COLS]

    # Đảm bảo không còn NaN ở đầu cuối do rolling
    df = df.bfill().ffill()

    return df

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(INFO_FILE):
        ids = sorted(pd.read_csv(INFO_FILE)["station"].dropna().astype(int).tolist())
    else:
        ids = sorted([int(f.replace("air_", "").replace(".csv", "")) for f in os.listdir(RAW_AIR_DIR) if f.startswith("air_")])

    for sid in ids:
        print(f"\n🔧 Processing station {sid}")
        df = process_station(sid)
        if df is None:
            print(f"⚠️  Missing files for station {sid}")
            continue
        out_path = os.path.join(OUT_DIR, f"station_{sid}.csv")
        df.to_csv(out_path, index=False)
        print(f"💾 Saved {out_path} ({len(df)} rows, {len(df.columns)} features)")

if __name__ == "__main__":
    run()
