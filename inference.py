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
  from inference import predict_pollutant
  result = predict_pollutant(station_id=1, df_clean=df, horizon=1, target_pollutant="pm25")
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import streamlit as st
import torch

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


def inverse_prediction(y_norm: np.ndarray, station_id: int, target_pollutant: str) -> np.ndarray:
    """
    Nhiệm vụ: Chuyển giá trị từ normalized space về đơn vị gốc (μg/m³).
    Thực hiện:
        1. Load scaler tương ứng với trạm.
        2. Lấy scaler của target_pollutant.
        3. Inverse_transform bằng sklearn scaler.
        4. Nếu method có 'log1p' thì áp dụng expm1 thêm.
    Trả về: mảng numpy giá trị gốc
    """
    scalers = load_scalers(station_id)
    if scalers is None:
        return y_norm
    method_tuple = scalers.get(target_pollutant)
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
# Cấu hình siêu tham số cho các mô hình DL
# ─────────────────────────────────────────────────────────────────────────────
MODEL_CONFIG_ITRANSFORMER = {
    1:  {'north': (64,  128, 4, 2), 'south': (64,  128, 4, 2)},
    3:  {'north': (64,  128, 4, 2), 'south': (64,  128, 4, 2)},
    6:  {'north': (64,  128, 4, 2), 'south': (128, 256, 4, 2)},
    12: {'north': (128, 256, 4, 2), 'south': (128, 256, 4, 2)},
    24: {'north': (128, 256, 4, 2), 'south': (128, 256, 4, 3)},
}

MODEL_CONFIG_MAMBA = {
    1:  {'north': (64, 16, 4, 2), 'south': (64, 16, 4, 2)},
    3:  {'north': (64, 16, 4, 2), 'south': (64, 16, 4, 2)},
    6:  {'north': (64, 16, 4, 2), 'south': (64, 16, 4, 2)},
    12: {'north': (64, 16, 4, 2), 'south': (64, 16, 4, 2)},
    24: {'north': (64, 16, 4, 2), 'south': (64, 16, 4, 2)},
}

MODEL_CONFIG_TFT = {
    1:  {'north': (64, 4), 'south': (64, 4)},
    3:  {'north': (64, 4), 'south': (64, 4)},
    6:  {'north': (64, 4), 'south': (64, 4)},
    12: {'north': (64, 4), 'south': (64, 4)},
    24: {'north': (64, 4), 'south': (64, 4)},
}

MODEL_CONFIG_PATCHTST = {
    1:  {'north': (16, 8, 64, 4, 2, 128), 'south': (16, 8, 64, 4, 2, 128)},
    3:  {'north': (16, 8, 64, 4, 2, 128), 'south': (16, 8, 64, 4, 2, 128)},
    6:  {'north': (16, 8, 64, 4, 2, 128), 'south': (16, 8, 64, 4, 2, 128)},
    12: {'north': (16, 8, 64, 4, 2, 128), 'south': (16, 8, 64, 4, 2, 128)},
    24: {'north': (16, 8, 64, 4, 2, 128), 'south': (16, 8, 64, 4, 2, 128)},
}

def get_dl_model_config(model_type: str, horizon: int, region: str):
    if model_type == "iTransformer":
        return MODEL_CONFIG_ITRANSFORMER[horizon][region]
    elif model_type == "Mamba":
        return MODEL_CONFIG_MAMBA[horizon][region]
    elif model_type == "TFT":
        return MODEL_CONFIG_TFT[horizon][region]
    elif model_type == "PatchTST":
        return MODEL_CONFIG_PATCHTST[horizon][region]
    elif model_type == "Toto-313M":
        return MODEL_CONFIG_TFT[horizon][region] # Reusing TFT size or default config
    raise ValueError(f"Unknown model config for {model_type}")

@st.cache_resource
def load_dl_model(model_type: str, region: str, horizon: int, enc_in: int):
    """
    Load và cache các mô hình Deep Learning (PyTorch).
    """
    save_dir = os.path.join(BASE_DIR, "models_saved", "block7", model_type)
    save_path = os.path.join(save_dir, f"{region}_t{horizon}.pth")
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint mô hình: {save_path}")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_type == "iTransformer":
        from models.iTransformer.model import iTransformer
        d_model, d_ff, n_heads, e_layers = get_dl_model_config(model_type, horizon, region)
        model = iTransformer(
            seq_len=48, pred_len=1, enc_in=enc_in,
            d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_ff=d_ff,
            dropout=0.15, use_norm=True
        )
    elif model_type == "Mamba":
        from models.Mamba.model import HFMambaModel
        d_model, d_state, d_conv, e_layers = get_dl_model_config(model_type, horizon, region)
        model = HFMambaModel(
            seq_len=48, pred_len=1, enc_in=enc_in,
            d_model=d_model, d_state=d_state, d_conv=d_conv, e_layers=e_layers,
            use_norm=True
        )
    elif model_type == "TFT":
        from models.TFT.model import TFTModel
        d_model, n_heads = get_dl_model_config(model_type, horizon, region)
        model = TFTModel(
            seq_len=48, pred_len=1, enc_in=enc_in,
            d_model=d_model, n_heads=n_heads,
            dropout=0.1, use_norm=True
        )
    elif model_type == "PatchTST":
        from models.PatchTST.model import HFPatchTSTModel
        patch_len, stride, d_model, n_heads, e_layers, d_ff = get_dl_model_config(model_type, horizon, region)
        model = HFPatchTSTModel(
            seq_len=48, pred_len=1, enc_in=enc_in,
            patch_len=patch_len, stride=stride, d_model=d_model, n_heads=n_heads,
            e_layers=e_layers, d_ff=d_ff,
            dropout=0.1, use_norm=True
        )
    elif model_type == "Toto-313M":
        from models.Toto.model import HFTotoModel
        d_model, n_heads = get_dl_model_config(model_type, horizon, region)
        model = HFTotoModel(
            seq_len=48, pred_len=1, enc_in=enc_in,
            model_name="DataDog/toto-313m", d_model=d_model,
            use_norm=True
        )
    else:
        raise ValueError(f"Unknown deep learning model: {model_type}")
        
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

# ─────────────────────────────────────────────────────────────────────────────
# API chính: dự báo chất ô nhiễm cho một trạm tại một mốc thời gian
# ─────────────────────────────────────────────────────────────────────────────
def predict_pollutant(station_id: int, df_clean: pd.DataFrame, horizon: int, model_type: str = "XGBoost", target_pollutant: str = "pm25") -> float:
    """
    Nhiệm vụ: Trả về giá trị dự báo (μg/m³) cho trạm và mốc thời gian chỉ định.
    QUAN TRỌNG: hỗ trợ cả XGBoost (truyền thống) và các model Deep Learning (PyTorch).
    """
    df_norm = load_norm_data(station_id)       # đã normalized, dùng cho feature engineering
    region  = get_region(station_id)
    
    if model_type == "XGBoost":
        if target_pollutant != "pm25":
            raise ValueError("XGBoost currently only supports PM2.5 forecasting")
        model   = load_xgb_model(region, horizon)
        X       = build_inference_features(df_norm, station_id, horizon)
        y_norm  = model.predict(X)                 # normalized space
        y_inv   = inverse_prediction(y_norm, station_id, target_pollutant) # μg/m³
        return float(max(0.0, round(float(y_inv[0]), 1)))
        
    # Xử lý các mô hình Deep Learning (PyTorch)
    # Xác định các trạm thuộc cùng vùng (region)
    from models.shared.dataset import EXCLUDE_COLS, PM25_COL
    POLLUTANTS = ['pm25', 'pm10', 'co', 'o3', 'no2', 'so2']
    if target_pollutant not in POLLUTANTS:
        raise ValueError(f"Unknown target pollutant: {target_pollutant}")
        
    if region == "north":
        sids = CLUSTER_NORTH
    else:
        sids = CLUSTER_SOUTH
        
    node_idx = sids.index(station_id)
    num_nodes = len(sids)
    pol_idx = POLLUTANTS.index(target_pollutant)
    num_targets = num_nodes * len(POLLUTANTS)
    
    # Load 48 giờ gần nhất cho tất cả trạm trong region
    dfs = [load_norm_data(sid) for sid in sids]
    
    features_list = []
    pollutants_list = []
    for df_node in dfs:
        df_s = df_node.tail(48).reset_index(drop=True)
        feat_cols = [c for c in df_s.columns if c not in EXCLUDE_COLS and c not in POLLUTANTS
                     and df_s[c].dtype in ['float64', 'float32', 'int64', 'int32']]
        features_list.append(df_s[feat_cols].fillna(0).values.astype(np.float32))
        
        # Lấy 6 chất ô nhiễm
        pol_vals = df_s[POLLUTANTS].fillna(0).values.astype(np.float32) # (48, 6)
        pollutants_list.append(pol_vals)
        
    # Nối mảng theo thứ tự: node0(p0..p5), node1(p0..p5), ...
    # Nghĩa là cột là [sid1_pm25, sid1_pm10, ..., sid2_pm25, ...]
    pollutants_matrix = np.concatenate(pollutants_list, axis=1)  # (48, N*6)
    shared = features_list[0]  # (48, F)
    x_data = np.concatenate([shared, pollutants_matrix], axis=1)  # (48, F + N*6)
    num_variates = x_data.shape[1]
    
    # Load model DL
    model = load_dl_model(model_type, region, horizon, num_variates)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_tensor = torch.tensor(x_data, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 48, F + N*6)
    
    with torch.no_grad():
        out = model(x_tensor)  # (1, 1, F + N*6)
        preds = out[0, 0, -num_targets:].cpu().numpy()  # (N*6,)
        preds_r = preds.reshape(num_nodes, len(POLLUTANTS)) # (N, 6)
        y_norm = preds_r[node_idx, pol_idx]
        y_inv = inverse_prediction(y_norm, station_id, target_pollutant)  # gốc
        return float(max(0.0, round(float(y_inv), 1)))


def predict_all_horizons(station_id: int, df_clean: pd.DataFrame,
                         horizons: tuple = (1, 3, 6, 12, 24), model_type: str = "XGBoost", target_pollutant: str = "pm25") -> dict:
    """
    Nhiệm vụ: Dự báo chất ô nhiễm cho nhiều mốc thời gian cùng lúc bằng model được chọn.
    """
    current_val = float(df_clean[target_pollutant].iloc[-1])
    result = {}
    for h in horizons:
        try:
            result[h] = predict_pollutant(station_id, df_clean, h, model_type, target_pollutant)
        except Exception as e:
            # Fallback về giá trị hiện tại nếu model không chạy được hoặc báo lỗi
            print(f"[Warning] Prediction failed for {target_pollutant} at horizon {h} with {model_type}: {e}")
            result[h] = round(current_val, 1)
    return result
