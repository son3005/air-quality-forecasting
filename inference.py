"""
inference.py
============
Module thực hiện inference XGBoost cho ứng dụng Streamlit.

Nhiệm vụ:
  - Xây dựng feature vector từ dữ liệu lịch sử (clean_station_*.csv)
  - Load và cache các XGBoost model đã train (block7)
  - Inverse-transform dự báo từ normalized space về μg/m³
  - Hỗ trợ cluster North / South theo định nghĩa dự án

Sử dụng:
  from inference import predict_pm25
  result = predict_pm25(station_id=1, df_clean=df, horizon=1)
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Cấu hình cluster và model (khớp với pipeline.py training)
# ─────────────────────────────────────────────────────────────────────────────

# Tất cả các trạm được dùng khi training (thứ tự quan trọng cho station_is_* onehot)
SELECTED_STATIONS = [1, 4, 5, 16, 17, 27, 7, 18, 24, 30, 31, 32]

# Phân cluster theo vùng địa lý
CLUSTER_NORTH = [1, 4, 5, 16, 17, 27]
CLUSTER_SOUTH = [7, 18, 24, 30, 31, 32]

# Các cột đặc trưng dùng trong training
PRECURSOR_COLS      = ['pm10', 'so2', 'no2', 'o3', 'co']
FUTURE_WEATHER_COLS = ['temp', 'precip', 'wind_spd', 'rh']
# Các cột loại trừ khi tạo cur_ features
EXCLUDE_COLS        = {'timestamp', 'split', 'station_id', 'province', 'district', 'pm25'}

PM25_COL = 'pm25'
LAGS     = [1, 2, 3, 6, 12, 24, 36, 48, 60, 72]   # lags PM2.5
ROLLS    = [3, 6, 12, 24, 48]                       # rolling windows

# Các cột cur_ đúng theo thứ tự training (từ split/block7 data)
# wind_dir → đã encode thành wind_sin + wind_cos khi training
_TRAINING_VALID_COLS = [
    'pm10', 'co', 'o3', 'no2', 'so2', 'temp', 'rh', 'dewpt', 'precip',
    'clouds', 'wind_spd', 'wind_gusts', 'soil_temp_0_7', 'soil_moist_0_7',
    'oxidation_potential', 'pollution_load', 'no2_so2_log_diff',
    'humid_sulfate_risk', 'thermal_stability', 'dust_source_potential',
    'is_frozen', 'is_outlier', 'is_pm25_sensor_error', 'is_weekend_holiday',
    'is_extreme_pm25_1h_ago', 'pm25_lag_1', 'pm25_lag_3', 'pm25_lag_6',
    'pm25_lag_12', 'pm25_lag_24', 'pm25_roll_mean_6', 'pm25_roll_mean_12',
    'pm25_roll_std_6', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'wind_sin', 'wind_cos',
]

# Số lượng neighbor features (từ compute_neighbor_features trong training)
# Vì app không có dữ liệu realtime của các trạm lân cận, ta fill = 0
# (model đã học được từ station_is_* onehot và PM2.5 history, ảnh hưởng nhỏ)
_NEIGHBOR_LAG_VALS = [1, 3, 6, 12, 24]
_NEIGHBOR_FEAT_COUNT = len(_NEIGHBOR_LAG_VALS) * 4 + 3   # 23 neighbor features

BASE_DIR    = os.path.dirname(__file__)
MODELS_DIR  = os.path.join(BASE_DIR, "models_saved")
SCALERS_DIR = os.path.join(BASE_DIR, "data", "normalized")
NORM_DIR    = os.path.join(BASE_DIR, "data", "normalized")   # chứa norm_station_*.csv
CLEAN_DIR   = os.path.join(BASE_DIR, "data", "clean")        # chứa clean_station_*.csv (chỉ dùng hiển thị)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: xác định cluster của trạm
# ─────────────────────────────────────────────────────────────────────────────
def get_region(station_id: int) -> str:
    """Trả về 'north' hoặc 'south' dựa theo station_id."""
    if station_id in CLUSTER_NORTH:
        return "north"
    elif station_id in CLUSTER_SOUTH:
        return "south"
    else:
        raise ValueError(f"Trạm {station_id} không thuộc cluster nào (north/south).")


@st.cache_data(ttl=600)
def load_norm_data(station_id: int) -> pd.DataFrame:
    """
    Nhiệm vụ: Load và cache dữ liệu đã normalized của một trạm.
    Dữ liệu này được dùng để build feature vector cho XGBoost inference.
    (Khác với clean_station_*.csv dùng để hiển thị, norm_station_*.csv có
     các giá trị đã được scale đún vửi không gian mà model được train.)
    Trả về: DataFrame được sắp xếp theo timestamp tăng dần
    """
    fpath = os.path.join(NORM_DIR, f"norm_station_{station_id}.csv")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Không tìm thấy norm data: {fpath}")
    df = pd.read_csv(fpath, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load model từ file pkl (cached để không load lại mỗi lần)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_xgb_model(region: str, horizon: int):
    """
    Nhiệm vụ: Load và cache XGBoost model đã train từ file .pkl.
    Tham số:
        region  – 'north' hoặc 'south'
        horizon – số giờ dự báo: 1, 3, 6, 12, hoặc 24
    Trả về: XGBRegressor object đã fit
    """
    fname = f"xgboost_{region}_T{horizon}_block7.pkl"
    fpath = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Model không tìm thấy: {fpath}")
    with open(fpath, "rb") as f:
        model = pickle.load(f)
        # Ép model chạy trên CPU để tránh lỗi crash do CUDA trong môi trường đa luồng (Streamlit)
        model.set_params(device="cpu")
        return model


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load scaler để inverse-transform (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_scalers(station_id: int):
    """
    Nhiệm vụ: Load dict scaler từ scalers_{id}.pkl.
    Trả về: dict {'pm25': (method_str, sklearn_scaler), ...}
    """
    fpath = os.path.join(SCALERS_DIR, f"scalers_{station_id}.pkl")
    if not os.path.exists(fpath):
        return None
    with open(fpath, "rb") as f:
        return pickle.load(f)


def inverse_pm25(y_norm: np.ndarray, station_id: int) -> np.ndarray:
    """
    Nhiệm vụ: Chuyển giá trị PM2.5 từ normalized space về μg/m³.
    Thực hiện:
        1. Load scaler tương ứng với trạm.
        2. Inverse_transform bằng sklearn scaler.
        3. Nếu method có 'log1p' thì áp dụng expm1 thêm.
    Trả về: mảng numpy giá trị μg/m³
    """
    scalers = load_scalers(station_id)
    if scalers is None:
        return y_norm
    method_tuple = scalers.get("pm25")
    if method_tuple is None:
        return y_norm
    method, sc = method_tuple[:2]
    y_inv = sc.inverse_transform(np.array(y_norm).reshape(-1, 1)).flatten()
    if "log1p" in method:
        y_inv = np.expm1(y_inv)
    return y_inv


# ─────────────────────────────────────────────────────────────────────────────
# Hàm chính: xây dựng feature vector từ DataFrame clean
# ─────────────────────────────────────────────────────────────────────────────
def build_inference_features(df_norm: pd.DataFrame, station_id: int, horizon: int) -> np.ndarray:
    """
    Nhiệm vụ: Xây dựng 1 feature vector (hàng cuối) từ dữ liệu đã normalized.
    Tham số:
        df_norm    – DataFrame từ data/normalized/norm_station_*.csv (đã normalized)
        station_id – ID của trạm đang xem
        horizon    – mốc dự báo (giờ): 1, 3, 6, 12, 24
    QUÁ TRÌNH:
        Dữ liệu norm_station_*.csv đã có sẵn các cột cần thiết – cấu trúc giống
        split/block7/ mà model được train. Chú yếu là các lag/roll đã precomputed
        trên normalized space, và wind_sin/wind_cos đã encode. Ta chỉ cần:
        1. Tính tại các lag dài hơn (chưa có trong CSV) trên normalized PM2.5
        2. Lấy các cur_ feature từ các cột đã có sẵn
        3. Thêm station onehot, neighbor placeholder
    Trả về: numpy array shape (1, 134)
    """
    df = df_norm.copy().reset_index(drop=True)
    n = len(df)
    feats = {}

    # ── 1. PM2.5 lags ──────────────────────────────────────────────
    for lag in LAGS:
        feats[f"pm25_lag_{lag}"] = df[PM25_COL].shift(lag)

    # ── 2. Precursor pollutant lags ─────────────────────────────────
    for col in PRECURSOR_COLS:
        if col in df.columns:
            for lag in [1, 3, 6, 12, 24]:
                feats[f"{col}_lag_{lag}"] = df[col].shift(lag)

    # ── 3. Rolling statistics PM2.5 ─────────────────────────────────
    for w in ROLLS:
        feats[f"pm25_roll_mean_{w}"] = df[PM25_COL].rolling(w, min_periods=1).mean().shift(1)
        feats[f"pm25_roll_std_{w}"]  = df[PM25_COL].rolling(w, min_periods=1).std().shift(1).fillna(0)
        feats[f"pm25_roll_max_{w}"]  = df[PM25_COL].rolling(w, min_periods=1).max().shift(1)

    # ── 4. Future weather (shift -horizon) ──────────────────────────
    # Tại hàng cuối sẽ không có giá trị tương lai → NaN → fill 0 sau
    for col in FUTURE_WEATHER_COLS:
        if col in df.columns:
            feats[f"{col}_fut_h{horizon}"] = df[col].shift(-horizon)

    # ── 5. Time features (cyclical encoding) ─────────────────────────
    ts = pd.to_datetime(df["timestamp"])
    hour = ts.dt.hour
    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feats["dow_sin"]  = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
    feats["dow_cos"]  = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
    feats["doy_sin"]  = np.sin(2 * np.pi * ts.dt.dayofyear / 365)
    feats["doy_cos"]  = np.cos(2 * np.pi * ts.dt.dayofyear / 365)

    # ── 6. cur_ features (khớp với thứ tự training) ─────────────────
    # Các cột luôn có trong clean data
    always_cols = [
        'pm10', 'co', 'o3', 'no2', 'so2', 'temp', 'rh', 'dewpt', 'precip',
        'clouds', 'wind_spd', 'wind_gusts', 'soil_temp_0_7', 'soil_moist_0_7',
        'oxidation_potential', 'pollution_load', 'no2_so2_log_diff',
        'humid_sulfate_risk', 'thermal_stability', 'dust_source_potential',
        'is_frozen', 'is_outlier', 'is_pm25_sensor_error', 'is_weekend_holiday',
        'is_extreme_pm25_1h_ago',
    ]
    for col in always_cols:
        feats[f"cur_{col}"] = df[col].values if col in df.columns else np.zeros(n)

    # Pre-computed lag/roll cols đã có sẵn trong clean data
    precomputed = [
        'pm25_lag_1', 'pm25_lag_3', 'pm25_lag_6', 'pm25_lag_12', 'pm25_lag_24',
        'pm25_roll_mean_6', 'pm25_roll_mean_12', 'pm25_roll_std_6',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    ]
    for col in precomputed:
        feats[f"cur_{col}"] = df[col].values if col in df.columns else np.zeros(n)

    # wind_sin, wind_cos đã có sẵn trong norm data (khác với clean data có wind_dir thô)
    if "wind_sin" in df.columns:
        feats["cur_wind_sin"] = df["wind_sin"].values
        feats["cur_wind_cos"] = df["wind_cos"].values
    elif "wind_dir" in df.columns:
        # fallback: nếu có wind_dir thô thì encode
        wd_rad = np.deg2rad(df["wind_dir"].fillna(0).values)
        feats["cur_wind_sin"] = np.sin(wd_rad)
        feats["cur_wind_cos"] = np.cos(wd_rad)
    else:
        feats["cur_wind_sin"] = np.zeros(n)
        feats["cur_wind_cos"] = np.zeros(n)

    # ── 7. Station one-hot encoding ─────────────────────────────────
    s_idx = SELECTED_STATIONS.index(station_id) if station_id in SELECTED_STATIONS else -1
    for si, sid in enumerate(SELECTED_STATIONS):
        feats[f"station_is_{sid}"] = int(si == s_idx)

    # ── 8. Neighbor features placeholder = 0 ────────────────────────
    # Lúc training dùng compute_neighbor_features() nhưng inference
    # không có dữ liệu realtime của trạm lân cận → fill 0
    for lag in _NEIGHBOR_LAG_VALS:
        feats[f"nbr_mean_lag{lag}"]  = 0.0
        feats[f"nbr_max_lag{lag}"]   = 0.0
        feats[f"nbr_std_lag{lag}"]   = 0.0
        feats[f"nbr_wmean_lag{lag}"] = 0.0
    feats["nbr_nearest_lag1"]  = 0.0
    feats["nbr_nearest_lag6"]  = 0.0
    feats["nbr_nearest_lag24"] = 0.0

    # ── 9. Gộp thành DataFrame, lấy dòng cuối, fill NaN ────────────
    feat_df = pd.DataFrame(feats, index=df.index)
    # Sanitize tên cột (khớp với X_train.columns.str.replace trong training)
    feat_df.columns = (feat_df.columns
                       .str.replace("[", "", regex=False)
                       .str.replace("]", "", regex=False)
                       .str.replace("<", "", regex=False))
    last_row = feat_df.iloc[[-1]].fillna(0)
    return last_row.values  # shape (1, n_features)


# ─────────────────────────────────────────────────────────────────────────────
# API chính: dự báo PM2.5 cho một trạm tại một mốc thời gian
# ─────────────────────────────────────────────────────────────────────────────
def predict_pm25(station_id: int, df_clean: pd.DataFrame, horizon: int) -> float:
    """
    Nhiệm vụ: Trả về giá trị PM2.5 dự báo (μg/m³) cho trạm và mốc thời gian chỉ định.
    QUAN TRỌNG: dùng df_norm (đã normalized) để build features, không dùng df_clean (raw).
    Tham số:
        station_id – ID của trạm đo (phải thuộc CLUSTER_NORTH hoặc CLUSTER_SOUTH)
        df_clean   – không dùng trong inference, chỉ giữ tham số để tương thích API
        horizon    – số giờ dự báo: 1, 3, 6, 12, hoặc 24
    Thực hiện:
        1. Load df_norm từ data/normalized/ (cached)
        2. Xác định region (north/south)
        3. Load model tương ứng (từ cache nếu đã gọi trước đó)
        4. Build feature vector từ norm data
        5. Predict (trả về giá trị normalized)
        6. Inverse-transform về μg/m³
    Trả về: float – giá trị PM2.5 dự báo (μg/m³), đã làm tròn 1 chữ số
    """
    df_norm = load_norm_data(station_id)       # đã normalized, dùng cho feature engineering
    region  = get_region(station_id)
    model   = load_xgb_model(region, horizon)
    X       = build_inference_features(df_norm, station_id, horizon)
    y_norm  = model.predict(X)                 # normalized space
    y_inv   = inverse_pm25(y_norm, station_id) # μg/m³
    return float(max(0.0, round(float(y_inv[0]), 1)))


def predict_all_horizons(station_id: int, df_clean: pd.DataFrame,
                         horizons: tuple = (1, 3, 6, 12, 24)) -> dict:
    """
    Nhiệm vụ: Dự báo PM2.5 cho nhiều mốc thời gian cùng lúc.
    Tham số:
        station_id – ID trạm đo
        df_clean   – DataFrame dữ liệu trạm
        horizons   – tuple các mốc dự báo (giờ), mặc định (1, 3, 6, 12, 24)
    Trả về: dict {horizon_giờ: giá_trị_PM2.5_μg/m³}
    Lỗi xử lý: nếu một mốc lỗi thì fallback = giá trị PM2.5 hiện tại
    """
    current_pm25 = float(df_clean["pm25"].iloc[-1])
    result = {}
    for h in horizons:
        try:
            result[h] = predict_pm25(station_id, df_clean, h)
        except Exception as e:
            # Fallback về giá trị hiện tại nếu model không chạy được
            result[h] = round(current_pm25, 1)
    return result
