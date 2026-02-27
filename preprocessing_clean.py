"""preprocessing_clean.py
--------------------------------
Script: đọc dữ liệu từng trạm (air + weather) và lưu file riêng
đầu ra: data/clean/station_{i}.csv
các cột yêu cầu:
    time (timestamp_local renamed), aqi, pm25, pm10, co,
    ma_pm25_24, rh, dewpt, temp, precip,
    wind_spd, wind_sin, wind_cos,
    hour_sin, hour_cos, month_sin, month_cos
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
    "wind_speed_10m (m/s)": "wind_spd",
    "wind_direction_10m (°)": "wind_dir",
}

# Các cột cuối cùng (thứ tự cố định)
OUTPUT_COLS = [
    "time",
    "aqi", "pm25", "pm10", "co",
    "ma_pm25_24",
    "rh", "dewpt", "temp", "precip",
    "wind_spd", "wind_sin", "wind_cos",
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
]

# Các cột air quality cần xử lý bất thường
AIR_QUALITY_COLS = ["aqi", "pm25", "pm10", "co"]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_air(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp_local"] = pd.to_datetime(df["timestamp_local"], errors="coerce")
    df = df.sort_values("timestamp_local").reset_index(drop=True)
    keep = ["timestamp_local", "aqi", "pm25", "pm10", "co"]
    return df[[c for c in keep if c in df.columns]]


def load_weather(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=WEATHER_MAP)
    df["timestamp_local"] = pd.to_datetime(df["timestamp_local"], errors="coerce")
    df = df.sort_values("timestamp_local").reset_index(drop=True)
    keep = ["timestamp_local", "temp", "rh", "dewpt", "precip", "wind_spd", "wind_dir"]
    return df[[c for c in keep if c in df.columns]]


def full_hourly_index(df: pd.DataFrame) -> pd.DataFrame:
    max_ts = df["timestamp_local"].max().ceil("h")
    end = min(END_DATE, max_ts)
    full_idx = pd.date_range(start=START_DATE, end=end, freq="h")
    df = df.drop_duplicates(subset=["timestamp_local"]).set_index("timestamp_local")
    return df.reindex(full_idx)


def wind_cyclic(df: pd.DataFrame) -> pd.DataFrame:
    if "wind_dir" in df.columns:
        rad = np.deg2rad(df["wind_dir"])
        df["wind_sin"] = np.sin(rad)
        df["wind_cos"] = np.cos(rad)
    else:
        df["wind_sin"] = 0.0
        df["wind_cos"] = 1.0
    return df


def time_cyclic(df: pd.DataFrame) -> pd.DataFrame:
    hour = df.index.hour
    month = df.index.month
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    return df


def rolling_pm25(df: pd.DataFrame) -> pd.DataFrame:
    if "pm25" in df.columns:
        df["ma_pm25_24"] = df["pm25"].rolling(window=24, min_periods=1).mean()
    else:
        df["ma_pm25_24"] = np.nan
    return df


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
    # Clip non‑negative columns
    for col in ["aqi", "pm25", "pm10", "co", "rh", "precip", "wind_spd"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
    return df


def detect_abnormal(df: pd.DataFrame, min_hours: int = 24) -> pd.DataFrame:
    """Phát hiện dữ liệu bất thường kéo dài cho feature air quality.
    - Frozen sensor: giá trị không đổi >= min_hours giờ liên tiếp → NaN
    - Prolonged extreme spikes: > Q3 + 3*IQR kéo dài >= min_hours giờ → NaN
    """
    df = df.copy()
    for col in AIR_QUALITY_COLS:
        if col not in df.columns:
            continue
        series = df[col]

        # 1. Frozen detection
        diff = series.diff().fillna(1)
        is_frozen = (diff == 0)
        if is_frozen.any():
            runs = (is_frozen != is_frozen.shift()).cumsum()
            for _, grp in is_frozen.groupby(runs):
                if grp.all() and len(grp) >= min_hours:
                    df.loc[grp.index, col] = np.nan

        # 2. Prolonged extreme spikes
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 3 * iqr
        if upper_bound > 0:
            is_extreme = (series > upper_bound)
            if is_extreme.any():
                runs = (is_extreme != is_extreme.shift()).cumsum()
                for _, grp in is_extreme.groupby(runs):
                    if grp.all() and len(grp) >= min_hours:
                        df.loc[grp.index, col] = np.nan
    return df


def detect_point_spikes(df: pd.DataFrame, window: int = 48,
                        z_threshold: float = 4.0) -> pd.DataFrame:
    """Phát hiện và loại bỏ các spike đột ngột (point anomalies) dựa trên
    rolling z-score: so sánh mỗi điểm với median và MAD của cửa sổ xung quanh.

    Nếu |x - rolling_median| / rolling_MAD > z_threshold → coi là spike → set NaN.
    Các NaN này sẽ được impute lại sau.
    """
    df = df.copy()
    for col in AIR_QUALITY_COLS:
        if col not in df.columns:
            continue
        series = df[col]
        rolling_med = series.rolling(window=window, center=True, min_periods=6).median()
        # MAD = Median Absolute Deviation (robust hơn std)
        rolling_mad = (series - rolling_med).abs().rolling(
            window=window, center=True, min_periods=6
        ).median()
        # Tránh chia cho 0
        rolling_mad = rolling_mad.replace(0, np.nan)
        modified_z = (series - rolling_med).abs() / (1.4826 * rolling_mad)
        # 1.4826 là hệ số chuẩn hóa MAD để tương đương std cho phân phối chuẩn
        spike_mask = modified_z > z_threshold
        n_spikes = spike_mask.sum()
        if n_spikes > 0:
            print(f"    ⚡ {col}: removed {n_spikes} point spikes")
            df.loc[spike_mask, col] = np.nan
    return df


def smooth_after_impute(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Áp dụng rolling median nhẹ (window=3) sau imputation để
    giảm nhiễu và các giá trị imputed bất thường.
    Chỉ áp dụng cho các cột air quality.
    """
    for col in AIR_QUALITY_COLS:
        if col in df.columns:
            df[col] = df[col].rolling(window=window, center=True, min_periods=1).median()
    return df


# ---------------------------------------------------------------------------
# Main per‑station processing
# ---------------------------------------------------------------------------

def process_station(station_id: int) -> pd.DataFrame | None:
    """Process a single station and return cleaned DataFrame."""
    aq_path = os.path.join(RAW_AIR_DIR, f"air_{station_id}.csv")
    wx_path = os.path.join(RAW_WX_DIR, f"weather_{station_id}.csv")
    if not os.path.exists(aq_path) or not os.path.exists(wx_path):
        print(f"⚠️  Missing files for station {station_id}")
        return None

    df_air = load_air(aq_path)
    df_wx  = load_weather(wx_path)
    df = pd.merge(df_air, df_wx, on="timestamp_local", how="outer")
    df = full_hourly_index(df)

    # Bước 1: Phát hiện dữ liệu frozen/extreme kéo dài → NaN
    df = detect_abnormal(df, min_hours=24)

    # Bước 2: Phát hiện spike đột ngột (point anomalies) → NaN
    df = detect_point_spikes(df, window=48, z_threshold=4.0)

    # Bước 3: Xử lý wind direction
    df = wind_cyclic(df)

    # Bước 4: Impute (linear interpolation + KNN)
    df = impute(df)

    # Bước 5: Smoothing nhẹ sau impute để giảm nhiễu
    df = smooth_after_impute(df, window=3)

    # Bước 6: Rolling features
    df = rolling_pm25(df)

    # Bước 7: Time cyclic features
    df = time_cyclic(df)

    # Rename index → time, giữ lại tất cả cột
    df = df.rename_axis("time").reset_index()
    for col in OUTPUT_COLS:
        if col not in df.columns:
            df[col] = np.nan
    extra_cols = [c for c in df.columns if c not in OUTPUT_COLS]
    df = df[OUTPUT_COLS + extra_cols]
    return df

# ---------------------------------------------------------------------------
# Run for all stations
# ---------------------------------------------------------------------------

def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(INFO_FILE):
        info = pd.read_csv(INFO_FILE)
        ids = sorted(info["station"].astype(int).tolist())
    else:
        ids = []
        for f in os.listdir(RAW_AIR_DIR):
            if f.startswith("air_") and f.endswith(".csv"):
                ids.append(int(f.replace("air_", "").replace(".csv", "")))
        ids = sorted(ids)

    for sid in ids:
        print(f"\n🔧 Processing station {sid}")
        df = process_station(sid)
        if df is None:
            continue
        out_path = os.path.join(OUT_DIR, f"station_{sid}.csv")
        df.to_csv(out_path, index=False)
        print(f"💾 Saved {out_path} ({len(df)} rows)")

if __name__ == "__main__":
    run()
