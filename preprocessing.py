import pandas as pd
import numpy as np
import warnings
from typing import Dict
import os
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================================
# HẰNG SỐ & CẤU HÌNH
# ============================================================================

# Khoảng thời gian xử lý
START_DATE = pd.Timestamp('2023-01-01 00:00:00')
END_DATE   = pd.Timestamp('2025-12-01 00:00:00')

# Mapping tên cột từ Open-Meteo sang tên chuẩn nội bộ
WEATHER_COLUMN_MAP = {
    'time':                                  'timestamp_local',
    'temperature_2m (°C)':                   'temp',
    'relative_humidity_2m (%)':              'rh',
    'dew_point_2m (°C)':                     'dewpt',
    'apparent_temperature (°C)':             'apparent_temp',
    'precipitation (mm)':                    'precip',
    'cloud_cover (%)':                       'clouds',
    'wind_speed_10m (m/s)':                  'wind_spd',
    'wind_direction_10m (°)':               'wind_dir',
    'wind_gusts_10m (m/s)':                  'wind_gusts',
    'soil_temperature_0_to_7cm (°C)':        'soil_temp_0_7',
    'soil_temperature_7_to_28cm (°C)':       'soil_temp_7_28',
    'soil_temperature_28_to_100cm (°C)':     'soil_temp_28_100',
    'soil_temperature_100_to_255cm (°C)':    'soil_temp_100_255',
    'soil_moisture_0_to_7cm (m³/m³)':        'soil_moist_0_7',
    'soil_moisture_7_to_28cm (m³/m³)':       'soil_moist_7_28',
    'soil_moisture_28_to_100cm (m³/m³)':     'soil_moist_28_100',
    'soil_moisture_100_to_255cm (m³/m³)':    'soil_moist_100_255',
}

# Ngưỡng vật lý để phát hiện ngoại lai — dùng tên cột sau khi rename
THRESHOLDS = {
    'aqi':            (0, 500),
    'pm25':           (0, 600),
    'pm10':           (0, 800),
    'co':             (0, 30000),
    'no2':            (0, 1000),
    'so2':            (0, 1000),
    'o3':             (0, 800),
    'temp':           (5, 45),
    'rh':             (0, 100),
    'dewpt':          (-10, 40),
    'apparent_temp':  (0, 55),
    'wind_spd':       (0, 50),
    'wind_gusts':     (0, 70),
    'clouds':         (0, 100),
    'precip':         (0, 200),
}

# Các cột cần kiểm tra dữ liệu đóng băng — dùng tên cột sau khi rename
FROZEN_CHECK_COLS = [
    'aqi', 'pm25', 'pm10', 'co', 'no2', 'so2', 'o3',
    'temp', 'rh', 'dewpt', 'wind_spd',
]

# Các cột đầu vào giữ lại sau khi merge
INPUT_COLUMNS = [
    'timestamp_local',
    # Chất lượng không khí
    'aqi', 'co', 'no2', 'o3', 'pm10', 'pm25', 'so2',
    # Thời tiết cơ bản
    'temp', 'rh', 'dewpt', 'apparent_temp',
    'precip', 'clouds',
    # Gió
    'wind_spd', 'wind_dir', 'wind_gusts',
    # Đất (chỉ lấy tầng trên cùng)
    'soil_temp_0_7', 'soil_moist_0_7',
]


# ============================================================================
# STEP 0: WIND COMPONENTS CALCULATION
# ============================================================================

def create_wind_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo wind_sin và wind_cos từ wind_dir.

    Args:
        df: DataFrame với cột wind_dir (độ)

    Returns:
        DataFrame có thêm cột wind_sin và wind_cos
    """
    if 'wind_dir' not in df.columns:
        print("⚠️  Cảnh báo: không tìm thấy cột wind_dir! Bỏ qua wind components.")
        return df

    df_copy = df.copy()
    wind_dir_rad = np.deg2rad(df_copy['wind_dir'])
    df_copy['wind_sin'] = np.sin(wind_dir_rad)
    df_copy['wind_cos'] = np.cos(wind_dir_rad)
    print("✅ Đã tạo wind_sin và wind_cos từ wind_dir")
    return df_copy


# ============================================================================
# STEP 1: TIME RANGE VALIDATION
# ============================================================================

def validate_time_range(df: pd.DataFrame, city_name: str) -> pd.DataFrame:
    """
    Validate time range và tạo full hourly index.

    Args:
        df: DataFrame với cột timestamp_local
        city_name: Tên thành phố để log

    Returns:
        DataFrame với full hourly index từ START_DATE đến END_DATE
        (hoặc đến max của dữ liệu nếu dữ liệu không đủ tới END_DATE)
    """
    print(f"\n📅 [{city_name}] Kiểm Tra Khoảng Thời Gian:")

    df['timestamp_local'] = pd.to_datetime(df['timestamp_local'])

    actual_min = df['timestamp_local'].min()
    actual_max = df['timestamp_local'].max()

    # Clip end date về max data nếu data không tới END_DATE
    effective_end = min(END_DATE, actual_max.ceil('h'))

    full_index = pd.date_range(start=START_DATE, end=effective_end, freq='h')

    print(f"   Kỳ vọng: {START_DATE} → {effective_end}")
    print(f"   Thực tế: {actual_min} → {actual_max}")
    print(f"   Số giờ kỳ vọng: {len(full_index):,}")
    print(f"   Số giờ thực tế: {len(df):,}")

    df.set_index('timestamp_local', inplace=True)
    df_reindexed = df.reindex(full_index)

    missing_count = df_reindexed.isnull().any(axis=1).sum()
    print(f"   Giờ thiếu:  {missing_count:,} ({missing_count / len(full_index) * 100:.2f}%)")

    return df_reindexed


# ============================================================================
# STEP 2: FROZEN DATA DETECTION + FLAGGING
# ============================================================================

def detect_frozen_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phát hiện frozen sensor data và tạo flag is_frozen.

    Frozen = std ≈ 0 trong rolling window 12h liên tục.
    Dùng rolling window thay vì daily group để phát hiện frozen ngắn hoặc qua đêm.

    Returns:
        DataFrame có thêm cột 'is_frozen' (0 = bình thường, 1 = frozen)
    """
    print(f"\n❄️  Phát Hiện Dữ Liệu Đóng Băng (rolling 12h):")

    df_clean = df.copy()
    df_clean['is_frozen'] = 0

    frozen_details = []

    for col in FROZEN_CHECK_COLS:
        if col not in df_clean.columns:
            continue

        # Rolling window 12h — phát hiện cả frozen ngắn và qua đêm
        rolling_std = df_clean[col].rolling(window=12, min_periods=6, center=True).std()
        frozen_mask = (rolling_std < 1e-6) & rolling_std.notna()
        frozen_count = frozen_mask.sum()

        if frozen_count > 0:
            df_clean.loc[frozen_mask, 'is_frozen'] = 1
            frozen_details.append(f"{col}: {frozen_count}")

    total_frozen = df_clean['is_frozen'].sum()
    print(f"   Tổng giờ đóng băng: {total_frozen:,}")
    if frozen_details:
        print(f"   Chi tiết: {', '.join(frozen_details[:5])}")

    return df_clean


# ============================================================================
# STEP 3: OUTLIER DETECTION + FLAGGING
# ============================================================================

def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phát hiện outliers và tạo flag is_outlier.

    Returns:
        DataFrame có thêm cột 'is_outlier' (0 = bình thường, 1 = outlier)
    """
    print(f"\n🚨 Phát Hiện Ngoại Lai:")

    df_clean = df.copy()
    df_clean['is_outlier'] = 0

    outlier_details = []

    for col, (min_val, max_val) in THRESHOLDS.items():
        if col not in df_clean.columns:
            continue

        mask = (df_clean[col] < min_val) | (df_clean[col] > max_val)
        count = mask.sum()

        if count > 0:
            df_clean.loc[mask, 'is_outlier'] = 1
            outlier_details.append(f"{col}: {count}")

    phys_outliers = df_clean['is_outlier'].sum()
    print(f"   Ngoại lai (vật lý): {phys_outliers:,}")
    if outlier_details:
        print(f"   Chi tiết: {', '.join(outlier_details[:5])}")

    # --- IQR-based outlier detection cho các cột AQ chính ---
    # Dùng 3×IQR (thay vì 1.5×) vì dữ liệu AQ có phân phối lệch phải tự nhiên
    # (ví dụ: spike PM2.5 mùa đông Hà Nội là hiện tượng thật)
    AQ_COLS_FOR_IQR = ['pm25', 'pm10', 'co', 'no2', 'so2', 'o3']
    iqr_details = []

    for col in AQ_COLS_FOR_IQR:
        if col not in df_clean.columns:
            continue

        valid_data = df_clean[col].dropna()
        if len(valid_data) == 0:
            continue

        Q1 = valid_data.quantile(0.25)
        Q3 = valid_data.quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            continue

        iqr_lower = Q1 - 3 * IQR
        iqr_upper = Q3 + 3 * IQR
        iqr_mask = (df_clean[col] < iqr_lower) | (df_clean[col] > iqr_upper)
        iqr_count = iqr_mask.sum()

        if iqr_count > 0:
            df_clean.loc[iqr_mask, 'is_outlier'] = 1
            iqr_details.append(f"{col}: {iqr_count} (>{iqr_upper:.0f})")

    if iqr_details:
        print(f"   Ngoại lai (IQR 3×): {', '.join(iqr_details[:5])}")

    total_outliers = df_clean['is_outlier'].sum()
    print(f"   Tổng ngoại lai: {total_outliers:,}")

    return df_clean


# ============================================================================
# STEP 4: DUPLICATE REMOVAL
# ============================================================================

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Xóa timestamp trùng lặp, giữ lại lần xuất hiện đầu tiên."""
    before = len(df)
    df_clean = df[~df.index.duplicated(keep='first')]
    removed = before - len(df_clean)

    if removed > 0:
        print(f"\n🔄 Xóa Trùng Lặp: loại bỏ {removed} timestamp trùng")

    return df_clean


# ============================================================================
# STEP 5: MISSING VALUES IMPUTATION
# ============================================================================

def impute_missing_values(df: pd.DataFrame, location: str) -> pd.DataFrame:
    """
    Điền giá trị thiếu bằng chiến lược hai giai đoạn.

    1. Nội suy tuyến tính theo thời gian cho khoảng trống nhỏ (≤6h)
    2. KNN Imputation (có chuẩn hóa StandardScaler) cho khoảng trống lớn

    Args:
        df: DataFrame có giá trị thiếu (phải có DatetimeIndex)
        location: Tên địa điểm để log

    Returns:
        DataFrame đã điền đầy đủ giá trị thiếu
    """
    print(f"\n🔍 [{location}] Điền Khuyết Giá Trị (Hai Giai Đoạn):")

    original_nans = df.isnull().sum().sum()
    total_values = df.size
    print(f"   Giá trị thiếu ban đầu: {original_nans:,} ({original_nans / total_values * 100:.2f}%)")

    # --- Báo cáo missing per-column cho các cột AQ ---
    aq_report_cols = ['aqi', 'pm25', 'pm10', 'co', 'no2', 'so2', 'o3']
    print(f"   📋 Missing per-column (AQ):")
    for col in aq_report_cols:
        if col in df.columns:
            col_missing = df[col].isnull().sum()
            col_total = len(df)
            col_pct = col_missing / col_total * 100
            warn = " ⚠️ >30%!" if col_pct > 30 else ""
            print(f"      {col:>6}: {col_missing:>6,} / {col_total:,} ({col_pct:.1f}%){warn}")

    if original_nans == 0:
        print("   ✓ Không có giá trị thiếu!")
        return df

    df_clean = df.copy()

    # Tách cột số
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

    # Giai đoạn 1: Nội suy tuyến tính (≤6h)
    print("   Giai đoạn 1: Nội suy tuyến tính (giới hạn = 6h)...")
    df_numeric = df_clean[numeric_cols].copy()
    df_numeric = df_numeric.interpolate(method='time', limit=6)
    df_clean[numeric_cols] = df_numeric

    after_linear = df_clean.isnull().sum().sum()
    filled_by_linear = original_nans - after_linear
    print(f"   ✓ Đã điền bằng nội suy: {filled_by_linear:,} giá trị")

    # Giai đoạn 2: KNN Imputation (có StandardScaler)
    if after_linear > 0:
        print("   Giai đoạn 2: KNN Imputation (StandardScaler + k=12)...")

        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_for_knn = df_clean[num_cols].copy()

        cols_to_drop = df_for_knn.columns[df_for_knn.isnull().all()]
        if len(cols_to_drop) > 0:
            print(f"   ⚠️  Bỏ qua {len(cols_to_drop)} cột thiếu hoàn toàn: {list(cols_to_drop)}")
            df_for_knn = df_for_knn.drop(columns=cols_to_drop)

        from sklearn.impute import KNNImputer

        # Chuẩn hóa trước KNN — tránh bias do scale khác nhau giữa các cột
        # (pm25: 0–600, rh: 0–100, precip: 0–200 → Euclidean bị dominated bởi cột lớn)
        col_means = df_for_knn.mean()
        col_stds = df_for_knn.std()
        col_stds = col_stds.replace(0, 1)  # Tránh chia cho 0
        df_scaled = (df_for_knn - col_means) / col_stds

        print(f"   ⏳ Đang chạy KNN trên {len(df_scaled.columns)} cột, {len(df_scaled):,} dòng...")
        imputer = KNNImputer(n_neighbors=12, weights='distance')
        df_imputed_scaled = imputer.fit_transform(df_scaled)

        # Inverse transform — chuyển về scale gốc
        df_imputed_values = df_imputed_scaled * col_stds.values + col_means.values

        df_imputed = pd.DataFrame(
            df_imputed_values,
            columns=df_for_knn.columns,
            index=df_clean.index
        )
        df_clean[df_imputed.columns] = df_imputed

        filled_by_knn = after_linear - df_clean.isnull().sum().sum()
        print(f"   ✓ Đã điền bằng KNN: {filled_by_knn:,} giá trị")
    else:
        filled_by_knn = 0

    # --- Sanity checks sau imputation ---
    missing_after = df_clean.isnull().sum().sum()
    total_filled = filled_by_linear + filled_by_knn
    print(f"   TỔNG ĐÃ ĐIỀN: {total_filled:,} giá trị ({total_filled / original_nans * 100:.1f}% số thiếu)")
    print(f"   Còn thiếu: {missing_after:,}")

    # Clip giá trị âm cho các cột không âm (AQ, wind, etc.)
    non_negative_cols = ['aqi', 'pm25', 'pm10', 'co', 'no2', 'so2', 'o3',
                         'wind_spd', 'wind_gusts', 'precip', 'clouds', 'rh']
    for col in non_negative_cols:
        if col in df_clean.columns:
            neg_count = (df_clean[col] < 0).sum()
            if neg_count > 0:
                df_clean[col] = df_clean[col].clip(lower=0)
                print(f"   🔧 Clip {neg_count} giá trị âm → 0 cho cột '{col}'")

    return df_clean


# ============================================================================
# STEP 6: FEATURE ENGINEERING
# ============================================================================

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo các feature theo thời gian.

    Bao gồm:
        - hour, day, month, year
        - dow (day of week): 0=Mon, 6=Sun
        - is_weekend: 1 nếu Thứ 7/Chủ nhật
        - pod (Part of Day): 1=daytime (6–17h), 0=nighttime
        - rush_hour: 1 nếu 6–9h hoặc 16–19h
        - Cyclic encoding cho hour, month, dow
    """
    df_copy = df.copy()
    df_copy['hour']  = df_copy.index.hour
    df_copy['day']   = df_copy.index.day
    df_copy['month'] = df_copy.index.month
    df_copy['year']  = df_copy.index.year

    # Day of week — quan trọng cho traffic pattern → phát thải
    df_copy['dow'] = df_copy.index.dayofweek  # 0=Mon, 6=Sun
    df_copy['is_weekend'] = (df_copy['dow'] >= 5).astype(int)

    # pod: Part of Day — 1 = daytime (6h–17h), 0 = nighttime
    df_copy['pod'] = np.where(
        (df_copy['hour'] >= 6) & (df_copy['hour'] <= 17), 1, 0
    )

    # Rush hour — 6–9h sáng, 16–19h chiều
    df_copy['rush_hour'] = np.where(
        ((df_copy['hour'] >= 6) & (df_copy['hour'] <= 9)) |
        ((df_copy['hour'] >= 16) & (df_copy['hour'] <= 19)),
        1, 0
    )

    # Cyclic encoding — tránh discontinuity
    df_copy['hour_sin']  = np.sin(2 * np.pi * df_copy['hour'] / 24)
    df_copy['hour_cos']  = np.cos(2 * np.pi * df_copy['hour'] / 24)
    df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['month'] / 12)
    df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['month'] / 12)
    df_copy['dow_sin']   = np.sin(2 * np.pi * df_copy['dow'] / 7)
    df_copy['dow_cos']   = np.cos(2 * np.pi * df_copy['dow'] / 7)

    return df_copy


def create_wind_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo feature kết hợp tốc độ và hướng gió."""
    df_copy = df.copy()
    if 'wind_spd' in df_copy.columns and 'wind_sin' in df_copy.columns:
        df_copy['spd_wind_sin'] = df_copy['wind_spd'] * df_copy['wind_sin']
        df_copy['spd_wind_cos'] = df_copy['wind_spd'] * df_copy['wind_cos']
    return df_copy


def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo các feature thời tiết và cross-pollutant.

    Bao gồm:
        - is_stagnant: gió yếu + dewpt cao → tích tụ ô nhiễm
        - ah: Absolute Humidity (g/m³)
        - dpd: Dew Point Depression
        - ratio_pm: PM2.5/PM10 (fine particle fraction)
        - o3_co: tổng O3+CO (photochemical oxidant)
        - no2_o3: NO2×O3 (photochemical indicator)
        - temp_inversion: soil_temp - temp (nghịch nhiệt proxy)
        - calm_humid: gió yếu + RH cao → điều kiện tích tụ
        - temp_change_6h: tốc độ biến đổi nhiệt (warming/cooling rate)
    """
    df_copy = df.copy()

    # --- Stagnation & atmospheric condition ---
    if 'wind_spd' in df_copy.columns and 'dewpt' in df_copy.columns:
        df_copy['is_stagnant'] = np.where(
            (df_copy['wind_spd'] < 1.5) & (df_copy['dewpt'] >= 0), 1, 0
        )

    # Calm + humid = điều kiện tích tụ ô nhiễm mạnh
    if 'wind_spd' in df_copy.columns and 'rh' in df_copy.columns:
        df_copy['calm_humid'] = np.where(
            (df_copy['wind_spd'] < 1.5) & (df_copy['rh'] > 80), 1, 0
        )

    # --- Humidity & temperature derivatives ---
    if 'dewpt' in df_copy.columns and 'temp' in df_copy.columns:
        # Absolute Humidity (g/m³)
        df_copy['ah'] = (
            6.112 * np.exp((17.67 * df_copy['dewpt']) / (df_copy['dewpt'] + 243.5))
            * 2.1674 / (273.15 + df_copy['temp'])
        )
        # DPD = Dew Point Depression = temp - dewpt
        df_copy['dpd'] = df_copy['temp'] - df_copy['dewpt']

    # Temperature inversion proxy — nghịch nhiệt giữ ô nhiễm gần mặt đất
    # Khi soil_temp > temp → khí lạnh ở trên, ô nhiễm bị kẹt
    if 'soil_temp_0_7' in df_copy.columns and 'temp' in df_copy.columns:
        df_copy['temp_inversion'] = df_copy['soil_temp_0_7'] - df_copy['temp']

    # Tốc độ biến đổi nhiệt — cooling rate nhanh → tầng nghịch nhiệt hình thành
    if 'temp' in df_copy.columns:
        df_copy['temp_change_6h'] = (df_copy['temp'] - df_copy['temp'].shift(6)).bfill()

    # --- Pollutant ratios & interactions ---
    if 'pm25' in df_copy.columns and 'pm10' in df_copy.columns:
        # Fine particle fraction — cao → nguồn phát thải đốt cháy, thấp → bụi cơ học
        df_copy['ratio_pm'] = df_copy['pm25'] / df_copy['pm10'].clip(lower=0.1)

    if 'o3' in df_copy.columns and 'co' in df_copy.columns:
        df_copy['o3_co'] = df_copy['o3'] + df_copy['co']

    # Photochemical indicator — NO2 + ánh sáng → O3, quan hệ nghịch ban ngày
    if 'no2' in df_copy.columns and 'o3' in df_copy.columns:
        df_copy['no2_o3'] = df_copy['no2'] * df_copy['o3']

    return df_copy


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo lag features, rolling statistics và trend cho AQI forecasting.

    Bao gồm:
        - PM2.5: lag 1/3/6/24, rolling mean/std/max/min, delta, trend 12h
        - AQI: lag 1/3/6/24 (target variable)
        - O3, NO2: lag 1/3 (pollutants phản ứng nhanh)
        - Temp, Wind: lag 1/24 (weather context)
        - Precipitation: rolling sum 6h, 24h
        - Interaction: wind × pm25 (ventilation effect)

    LƯU Ý VỀ DATA LEAKAGE:
        - shift() và rolling() đều backward-looking → KHÔNG leak dữ liệu tương lai.
        - Hàm này được gọi trong process_single_station() → tính PER-STATION.
        - Dùng bfill() cho các giờ đầu thiếu lag.
    """
    df_copy = df.copy()

    # =====================================================
    # PM2.5 — TARGET FEATURES
    # =====================================================
    if 'pm25' in df_copy.columns:
        # Lag features
        for lag in [1, 3, 6, 24]:
            df_copy[f'pm25_lag_{lag}'] = df_copy['pm25'].shift(lag).bfill()

        # Delta (rate of change)
        df_copy['delta_pm25'] = df_copy['pm25'].diff().bfill()

        # Rolling mean
        df_copy['ma_pm25_4']  = df_copy['pm25'].rolling(window=4, min_periods=1).mean()
        df_copy['ma_pm25_12'] = df_copy['pm25'].rolling(window=12, min_periods=1).mean()
        df_copy['ma_pm25_24'] = df_copy['pm25'].rolling(window=24, min_periods=1).mean()

        # Rolling statistics — volatility & extremes
        df_copy['std_pm25_24'] = df_copy['pm25'].rolling(window=24, min_periods=1).std().fillna(0)
        df_copy['max_pm25_24'] = df_copy['pm25'].rolling(window=24, min_periods=1).max()
        df_copy['min_pm25_24'] = df_copy['pm25'].rolling(window=24, min_periods=1).min()

        # Trend 12h — hệ số góc linear regression trên 12 giờ gần nhất
        # Dương = ô nhiễm tăng, âm = ô nhiễm giảm
        x = np.arange(12)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()
        df_copy['pm25_trend_12'] = (
            df_copy['pm25']
            .rolling(window=12, min_periods=6)
            .apply(lambda y: ((x[:len(y)] - x_mean) * (y - y.mean())).sum() / x_var, raw=True)
            .fillna(0)
        )

        # Wind × PM2.5 interaction — ventilation effect
        if 'wind_spd' in df_copy.columns:
            df_copy['w_pm25'] = df_copy['wind_spd'] * df_copy['pm25']

    # =====================================================
    # AQI — TARGET VARIABLE LAGS
    # =====================================================
    if 'aqi' in df_copy.columns:
        for lag in [1, 3, 6, 24]:
            df_copy[f'aqi_lag_{lag}'] = df_copy['aqi'].shift(lag).bfill()
        df_copy['delta_aqi'] = df_copy['aqi'].diff().bfill()
        df_copy['ma_aqi_24'] = df_copy['aqi'].rolling(window=24, min_periods=1).mean()

    # =====================================================
    # OTHER POLLUTANTS — LAGS
    # =====================================================
    # O3: chu kỳ ngày rõ rệt (peak chiều do quang hóa)
    if 'o3' in df_copy.columns:
        for lag in [1, 3]:
            df_copy[f'o3_lag_{lag}'] = df_copy['o3'].shift(lag).bfill()

    # NO2: phản ứng nhanh với giao thông
    if 'no2' in df_copy.columns:
        for lag in [1, 3]:
            df_copy[f'no2_lag_{lag}'] = df_copy['no2'].shift(lag).bfill()

    # =====================================================
    # WEATHER — LAGS
    # =====================================================
    # Temp: lag 1 (recent), lag 24 (same hour yesterday)
    if 'temp' in df_copy.columns:
        df_copy['temp_lag_1']  = df_copy['temp'].shift(1).bfill()
        df_copy['temp_lag_24'] = df_copy['temp'].shift(24).bfill()

    # Wind speed: ảnh hưởng trực tiếp đến khuếch tán
    if 'wind_spd' in df_copy.columns:
        df_copy['wind_lag_1'] = df_copy['wind_spd'].shift(1).bfill()
        df_copy['wind_lag_3'] = df_copy['wind_spd'].shift(3).bfill()

    # =====================================================
    # PRECIPITATION — ROLLING
    # =====================================================
    if 'precip' in df_copy.columns:
        df_copy['rain_sum_6']  = df_copy['precip'].rolling(window=6, min_periods=1).sum()
        df_copy['rain_sum_24'] = df_copy['precip'].rolling(window=24, min_periods=1).sum()

    return df_copy


def create_all_features(df: pd.DataFrame, location: str) -> pd.DataFrame:
    """Tạo tất cả features."""
    print(f"\n🔨 [{location}] Feature Engineering:")

    original_cols = len(df.columns)
    df = create_time_features(df)
    df = create_wind_features(df)
    df = create_weather_features(df)
    df = create_lag_features(df)

    new_cols = len(df.columns) - original_cols
    print(f"   Đã tạo {new_cols} features mới")
    print(f"   Tổng số cột: {len(df.columns)}")

    return df


# ============================================================================
# STEP 7: DISTRIBUTION ANALYSIS
# ============================================================================

def analyze_distribution(df: pd.DataFrame) -> Dict:
    """Phân tích phân phối thời gian để gợi ý chia train/val/test."""
    print("\n📊 Phân Tích Phân Phối Dữ Liệu:")

    start = df.index.min()
    end   = df.index.max()
    total_days = (end - start).days

    print(f"   Khoảng thời gian: {start} → {end}")
    print(f"   Số ngày: {total_days}")
    print(f"   Số giờ: {len(df):,}")

    train_end = start + pd.Timedelta(days=int(total_days * 0.70))
    val_end   = start + pd.Timedelta(days=int(total_days * 0.85))

    print(f"\n   📅 Gợi ý chia tập (Chronological):")
    print(f"      Train: {start.date()} → {train_end.date()} (70%)")
    print(f"      Val:   {train_end.date()} → {val_end.date()} (15%)")
    print(f"      Test:  {val_end.date()} → {end.date()} (15%)")

    return {'train_end': train_end, 'val_end': val_end}


# ============================================================================
# STEP 8: MAIN PIPELINE FOR SINGLE STATION
# ============================================================================

def process_single_station(province: str, district: str,
                            aq_file: str, weather_file: str) -> pd.DataFrame:
    """
    Xử lý dữ liệu một trạm qua toàn bộ pipeline.

    1. Load air_quality và weather_historical
    2. Rename cột weather (Open-Meteo → tên chuẩn nội bộ)
    3. Merge theo timestamp
    4. Chạy toàn bộ các bước tiền xử lý
    5. Feature engineering

    Args:
        province:     Tên tỉnh/thành
        district:     Tên quận/huyện
        aq_file:      Đường dẫn file air quality
        weather_file: Đường dẫn file weather historical

    Returns:
        DataFrame đã xử lý, sẵn sàng để ghép vào panel data
    """
    location_name = f"{province}_{district}"

    print("=" * 60)
    print(f"🌍 Đang xử lý trạm: {location_name}")
    print("=" * 60)

    # --- Load air quality ---
    print(f"   Đọc air quality: {aq_file}")
    df_air = pd.read_csv(aq_file)

    # Air quality data thường bị sort ngược (newest → oldest) → sort ascending
    df_air['timestamp_local'] = pd.to_datetime(df_air['timestamp_local'])
    df_air = df_air.sort_values('timestamp_local').reset_index(drop=True)
    print(f"   ✓ Air quality: {len(df_air):,} dòng, {len(df_air.columns)} cột")

    # Giữ lại các cột AQ cần thiết
    aq_cols_needed = ['timestamp_local', 'aqi', 'co', 'no2', 'o3', 'pm10', 'pm25', 'so2']
    aq_available = [c for c in aq_cols_needed if c in df_air.columns]
    df_air = df_air[aq_available]

    # --- Load weather ---
    print(f"   Đọc weather: {weather_file}")
    df_weather = pd.read_csv(weather_file)

    # Rename cột từ Open-Meteo format → tên chuẩn nội bộ
    df_weather = df_weather.rename(columns=WEATHER_COLUMN_MAP)

    # Parse timestamp (cột 'time' → 'timestamp_local' sau rename)
    df_weather['timestamp_local'] = pd.to_datetime(df_weather['timestamp_local'])
    df_weather = df_weather.sort_values('timestamp_local').reset_index(drop=True)
    print(f"   ✓ Weather: {len(df_weather):,} dòng, {len(df_weather.columns)} cột")

    # --- Merge ---
    print("   Ghép dữ liệu air quality và weather...")
    df = pd.merge(df_air, df_weather, on='timestamp_local', how='outer')
    print(f"   ✓ Merged: {len(df):,} dòng, {len(df.columns)} cột")

    # --- Giữ chỉ các cột cần thiết ---
    available_cols = [c for c in INPUT_COLUMNS if c in df.columns]
    missing_cols   = [c for c in INPUT_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"   ⚠️  Cột không tồn tại trong dữ liệu: {missing_cols}")
    df = df[available_cols]

    # --- Xóa duplicate timestamp trước khi set index ---
    before_dup = len(df)
    df = df.drop_duplicates(subset=['timestamp_local'], keep='first')
    removed_dup = before_dup - len(df)
    if removed_dup > 0:
        print(f"   ✓ Đã xóa {removed_dup} timestamp trùng lặp")

    # --- Pipeline chính ---
    df = create_wind_components(df)
    df = validate_time_range(df, location_name)
    df = detect_frozen_data(df)
    df = detect_outliers(df)
    df = remove_duplicates(df)
    df = impute_missing_values(df, location_name)
    df = create_all_features(df, location_name)

    # Thêm cột định danh địa điểm
    df['province'] = province
    df['district'] = district
    df['location'] = location_name

    print(f"\n✅ [{location_name}] Hoàn tất xử lý!")
    print(f"   Shape cuối: {df.shape}")
    print(f"   Cột: {list(df.columns)}")

    return df


# ============================================================================
# FINAL: MERGE AND SAVE
# ============================================================================

def merge_and_save(dfs: list, output_file: str = "data/clean_data_all.csv") -> pd.DataFrame:
    """Ghép tất cả các trạm và lưu kết quả cuối cùng."""
    print("\n" + "=" * 60)
    print("📦 Ghép Tất Cả Các Trạm")
    print("=" * 60)

    if not dfs:
        print("Không có dữ liệu để ghép!")
        return pd.DataFrame()

    # Reset DatetimeIndex về cột thường trước khi concat
    dfs_reset = [df.reset_index(names='timestamp_local') for df in dfs]

    df_all = pd.concat(dfs_reset, ignore_index=True)

    print(f"\n   Tổng kết dữ liệu cuối:")
    print(f"   Shape: {df_all.shape}")
    print(f"   Missing: {df_all.isnull().sum().sum()}")
    print(f"   Số trạm: {df_all['location'].nunique()}")

    print(f"\n   Phân phối trạm:")
    for loc, count in df_all['location'].value_counts().items():
        pct = count / len(df_all) * 100
        print(f"      {loc}: {count:,} ({pct:.1f}%)")

    # Kiểm tra quality flags
    if 'is_frozen' in df_all.columns:
        frozen_count  = df_all['is_frozen'].sum()
        outlier_count = df_all['is_outlier'].sum()
        print(f"\n   Quality Flags:")
        print(f"      Frozen:   {frozen_count:,}  ({frozen_count  / len(df_all) * 100:.2f}%)")
        print(f"      Outliers: {outlier_count:,} ({outlier_count / len(df_all) * 100:.2f}%)")

    # Phân tích phân phối
    analyze_distribution(df_all.set_index('timestamp_local'))

    # Lưu file
    out_dir = os.path.dirname(output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    df_all.to_csv(output_file, index=False)
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n💾 Đã lưu: {output_file}  ({file_size:.2f} MB)")

    return df_all


# ============================================================================
# DATA DISCOVERY
# ============================================================================

def get_all_stations(data_dir: str = "data") -> list:
    """
    Quét thư mục data để tìm tất cả trạm đo và file thời tiết tương ứng.

    Cấu trúc thư mục kỳ vọng:
        data/<Province>/air_quality/<District>.csv
        data/<Province>/weather_historical/<District>.csv

    Returns:
        List of tuples: (province, district, aq_file, weather_file)
    """
    stations = []

    for province in sorted(os.listdir(data_dir)):
        province_path = os.path.join(data_dir, province)

        if not os.path.isdir(province_path):
            continue

        aq_dir      = os.path.join(province_path, "air_quality")
        weather_dir = os.path.join(province_path, "weather_historical")

        if not os.path.exists(aq_dir):
            continue

        for file in sorted(os.listdir(aq_dir)):
            if not file.endswith(".csv"):
                continue

            district     = file.replace(".csv", "")
            aq_file      = os.path.join(aq_dir, file)
            weather_file = os.path.join(weather_dir, file)

            if os.path.exists(weather_file):
                stations.append((province, district, aq_file, weather_file))
                print(f"📍 Tìm thấy trạm: {province} - {district}")
            else:
                print(f"⚠️  Không tìm thấy weather cho: {province} - {district}")

    return stations


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def run_complete_pipeline(data_dir: str = "data",
                          output_file: str = "data/clean_data_all.csv") -> pd.DataFrame:
    """
    Chạy toàn bộ preprocessing pipeline cho tất cả trạm trong thư mục data.
    """
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  🌍 COMPLETE PREPROCESSING PIPELINE - ALL STATIONS  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")

    # 1. Quét tất cả các trạm
    stations = get_all_stations(data_dir)
    print(f"\n🚀 Tìm thấy {len(stations)} trạm để xử lý.")

    if not stations:
        print("❌ Không tìm thấy trạm nào. Kiểm tra lại cấu trúc thư mục data.")
        return pd.DataFrame()

    dfs = []

    # 2. Xử lý từng trạm
    for i, (province, district, aq_file, weather_file) in enumerate(stations):
        print(f"\n[{i+1}/{len(stations)}] {province} - {district} ...")
        try:
            df_station = process_single_station(province, district, aq_file, weather_file)
            dfs.append(df_station)
        except Exception as e:
            print(f"❌ Lỗi xử lý {province} - {district}: {str(e)}")
            import traceback
            traceback.print_exc()

    # 3. Ghép thành Panel Data và lưu
    df_final = merge_and_save(dfs, output_file)

    print("\n" + "=" * 60)
    print("✅ PIPELINE HOÀN THÀNH!")
    print("=" * 60)

    return df_final


if __name__ == "__main__":
    df = run_complete_pipeline()

    if not df.empty:
        print("\n📊 TÓM TẮT CUỐI:")
        print(f"   Các cột: {list(df.columns)}")
        print(f"\n   5 dòng đầu tiên:")
        print(df.head())
