"""
=============================================================================
Module: AQI Graph Dataset v3 — Data Pipeline: CSV → Tensor 4D
=============================================================================
Nâng cấp:
  - Mở rộng từ 22 lên 52 features (bao gồm lag, rolling, tương tác khí)
  - Dùng RobustScaler thay StandardScaler (kháng outlier tốt hơn)
  - Log-transform cho target trước scaling (làm mượt lưỡi cưa)
  - Hỗ trợ custom feature groups linh hoạt
=============================================================================
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler


# ──────────────────────────────────────────────────────────────────────
# FEATURE GROUPS (v3: 52 features thay vì 22)
# ──────────────────────────────────────────────────────────────────────

# 1. AIR QUALITY: các chất ô nhiễm chính
AQ_CORE = ['pm25', 'aqi', 'pm10', 'co', 'no2', 'o3', 'so2']

# 2. WEATHER: điều kiện khí tượng ảnh hưởng trực tiếp
WEATHER_CORE = [
    'temp', 'rh', 'dewpt', 'apparent_temp',          # Nhiệt + ẩm
    'precip', 'clouds',                                # Mưa + mây
    'wind_spd', 'wind_dir', 'wind_gusts',             # Gió (tốc độ + hướng + giật)
    'wind_sin', 'wind_cos',                            # Gió (encoded sin/cos)
    'soil_temp_0_7', 'soil_moist_0_7',                # Đất (ảnh hưởng bốc hơi)
]

# 3. TEMPORAL ENCODING: chu kỳ thời gian
TEMPORAL = [
    'hour_sin', 'hour_cos',                            # Chu kỳ ngày
    'dow_sin', 'dow_cos',                              # Chu kỳ tuần
    'month_sin', 'month_cos',                          # Chu kỳ năm
    'rush_hour', 'is_weekend',                         # Binary flags hữu ích
]

# 4. DERIVED METEOROLOGICAL: điều kiện khí tượng phái sinh
METEO_DERIVED = [
    'ah',                                              # Absolute Humidity
    'dpd',                                             # Dew Point Depression
    'temp_inversion',                                  # Phân tầng nghịch nhiệt
    'temp_change_6h',                                  # Thay đổi nhiệt 6h
    'is_stagnant',                                     # Trì trệ gió
    'calm_humid',                                      # Lặng gió + ẩm
    'spd_wind_sin', 'spd_wind_cos',                    # Gió vector theo tốc độ
]

# 5. AIR QUALITY RATIOS: tỷ lệ khí chéo (cross-pollutant)
AQ_RATIOS = [
    'ratio_pm',                                        # PM2.5/PM10
    'o3_co',                                           # O3/CO
    'no2_o3',                                          # NO2/O3
]

# 6. PM2.5 LAG + ROLLING: xu hướng bụi quá khứ
PM25_HISTORY = [
    'pm25_lag_1', 'pm25_lag_3', 'pm25_lag_6', 'pm25_lag_24',   # Lag
    'delta_pm25',                                               # Biến thiên
    'ma_pm25_4', 'ma_pm25_12', 'ma_pm25_24',                  # Moving Average
    'std_pm25_24',                                              # Volatility
    'pm25_trend_12',                                            # Xu hướng 12h
]

# 7. AQI LAG: xu hướng AQI quá khứ
AQI_HISTORY = [
    'aqi_lag_1', 'aqi_lag_3', 'aqi_lag_6', 'aqi_lag_24',
    'delta_aqi',
    'ma_aqi_24',
]

# 8. OTHER POLLUTANT LAGS
OTHER_LAGS = [
    'o3_lag_1', 'o3_lag_3',
    'no2_lag_1', 'no2_lag_3',
    'temp_lag_1', 'temp_lag_24',
    'wind_lag_1', 'wind_lag_3',
    'rain_sum_6', 'rain_sum_24',
]

# ── TỔNG HỢP: 52 features ──
ALL_FEATURES = (
    AQ_CORE + WEATHER_CORE + TEMPORAL + METEO_DERIVED +
    AQ_RATIOS + PM25_HISTORY + AQI_HISTORY + OTHER_LAGS
)

# ── Target: dự báo PM2.5 và AQI (multi-task) ──
TARGET_COLS = ['pm25', 'aqi']


class AQIGraphDataset(Dataset):
    """
    Dataset v3 — 52 features + RobustScaler + Log-transform targets.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        node_list: list,
        feature_cols: list = None,
        target_cols: list = None,
        seq_len: int = 72,
        pred_len: int = 12,
        scaler=None,
        fit_scaler: bool = False,
        province_col: str = 'province',
        time_col: str = 'timestamp_local'
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.nodes = node_list
        self.N = len(node_list)

        if feature_cols is None:
            feature_cols = ALL_FEATURES
        if target_cols is None:
            target_cols = TARGET_COLS

        # Chỉ lấy features có trong data thực tế
        available_cols = [c for c in feature_cols if c in df.columns]
        missing_cols = [c for c in feature_cols if c not in df.columns]
        if missing_cols and fit_scaler:
            print(f"  ⚠️ Features thiếu trong CSV (bỏ qua): {missing_cols}")
        feature_cols = available_cols

        self.feature_cols = feature_cols
        self.target_cols = target_cols
        C = len(feature_cols)

        # ── Index của target cols trong feature_cols ──
        self.target_idx = [feature_cols.index(t) for t in target_cols if t in feature_cols]

        # ══════════════════════════════════════════════════
        # BƯỚC 1: Pivot — flat CSV → [T_total, N, C]
        # ══════════════════════════════════════════════════
        cols_to_keep = [time_col, province_col] + [c for c in feature_cols if c in df.columns]
        df_sub = df[cols_to_keep].copy()
        pivot = df_sub.pivot_table(
            index=time_col,
            columns=province_col,
            values=[c for c in feature_cols if c in df_sub.columns]
        ).sort_index()

        # ══════════════════════════════════════════════════
        # BƯỚC 2: Xử lý giá trị thiếu
        # ══════════════════════════════════════════════════
        pivot = pivot.interpolate(method='linear', limit=3)
        pivot = pivot.ffill().bfill()
        pivot = pivot.fillna(0)

        # ══════════════════════════════════════════════════
        # BƯỚC 3: Reshape → numpy array [T_total, N, C]
        # ══════════════════════════════════════════════════
        T_total = len(pivot)
        data = np.zeros((T_total, self.N, C), dtype=np.float32)
        for ni, prov in enumerate(node_list):
            for ci, feat in enumerate(feature_cols):
                try:
                    data[:, ni, ci] = pivot[(feat, prov)].values
                except KeyError:
                    pass

        # ══════════════════════════════════════════════════
        # BƯỚC 3.5: Log-transform cho targets (PM2.5, AQI)
        # ══════════════════════════════════════════════════
        for f_idx in self.target_idx:
            data[:, :, f_idx] = np.log1p(np.maximum(data[:, :, f_idx], 0))

        # Log-transform cho các lag/rolling features liên quan
        log_cols = [
            'pm25_lag_1', 'pm25_lag_3', 'pm25_lag_6', 'pm25_lag_24',
            'ma_pm25_4', 'ma_pm25_12', 'ma_pm25_24',
            'max_pm25_24', 'min_pm25_24',
            'aqi_lag_1', 'aqi_lag_3', 'aqi_lag_6', 'aqi_lag_24',
            'ma_aqi_24',
            'pm10', 'co',  # pollutants có đuôi dài
        ]
        for col_name in log_cols:
            if col_name in feature_cols:
                ci = feature_cols.index(col_name)
                data[:, :, ci] = np.log1p(np.maximum(data[:, :, ci], 0))

        # ══════════════════════════════════════════════════
        # BƯỚC 4: Chuẩn hóa (RobustScaler)
        # ══════════════════════════════════════════════════
        # RobustScaler: dùng median & IQR thay mean & std
        # → Kháng outlier tốt hơn StandardScaler cho dữ liệu AQI

        if fit_scaler:
            self.scaler = RobustScaler()
            flat = data.reshape(-1, C)
            self.scaler.fit(flat)
            if fit_scaler:
                print(f"  ✅ RobustScaler fitted trên {T_total * self.N:,} samples × {C} features")
        else:
            self.scaler = scaler

        if self.scaler is not None:
            flat = data.reshape(-1, C)
            data = self.scaler.transform(flat).reshape(T_total, self.N, C)

        # Clip giá trị cực đoan sau khi scale (an toàn)
        data = np.clip(data, -10, 10)

        # ── Chuyển sang Tensor ──
        self.data = torch.FloatTensor(data)

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx: int):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]
        y = y[:, :, self.target_idx]
        return x.permute(1, 0, 2), y.permute(1, 0, 2)


# ──────────────────────────────────────────────────────────────────────
# Build DataLoaders
# ──────────────────────────────────────────────────────────────────────

def build_dataloaders(
    csv_path: str,
    node_list: list,
    feature_cols: list = None,
    target_cols: list = None,
    seq_len: int = 72,
    pred_len: int = 12,
    batch_size: int = 32,
    train_end: str = '2023-09-30',
    val_end: str = '2023-11-30',
    province_col: str = 'province',
    time_col: str = 'timestamp_local',
    num_workers: int = 0
) -> tuple:
    """Tạo DataLoaders cho train, val, test với chronological split."""

    df = pd.read_csv(csv_path, parse_dates=[time_col])

    # ── Chronological split ──
    train_df = df[df[time_col] <= train_end].copy()
    val_df = df[(df[time_col] > train_end) & (df[time_col] <= val_end)].copy()
    test_df = df[df[time_col] > val_end].copy()

    print(f"[Data Split]")
    print(f"  Train: {len(train_df):>8,} rows  (→ {train_end})")
    print(f"  Val:   {len(val_df):>8,} rows  ({train_end} → {val_end})")
    print(f"  Test:  {len(test_df):>8,} rows  ({val_end} →)")

    # ── Tạo Datasets ──
    train_set = AQIGraphDataset(
        train_df, node_list, feature_cols, target_cols,
        seq_len, pred_len,
        fit_scaler=True,
        province_col=province_col, time_col=time_col
    )

    val_set = AQIGraphDataset(
        val_df, node_list, feature_cols, target_cols,
        seq_len, pred_len,
        scaler=train_set.scaler,
        province_col=province_col, time_col=time_col
    )

    test_set = AQIGraphDataset(
        test_df, node_list, feature_cols, target_cols,
        seq_len, pred_len,
        scaler=train_set.scaler,
        province_col=province_col, time_col=time_col
    )

    print(f"  Features used:   {len(train_set.feature_cols)}")
    print(f"  Train samples: {len(train_set):>6,}")
    print(f"  Val samples:   {len(val_set):>6,}")
    print(f"  Test samples:  {len(test_set):>6,}")

    # ── Tạo DataLoaders ──
    train_loader = DataLoader(
        train_set, batch_size=batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, train_set.scaler
