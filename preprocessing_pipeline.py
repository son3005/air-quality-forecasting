import pandas as pd
import numpy as np
import warnings
from typing import Dict, Tuple
import os
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================================
# HẰNG SỐ & CẤU HÌNH
# ============================================================================

# Cấu hình khoảng thời gian
START_DATE = pd.Timestamp('2023-01-01 00:00:00')
END_DATE = pd.Timestamp('2025-12-01 00:00:00')

# Ngưỡng vật lý để phát hiện ngoại lai
THRESHOLDS = {
    'aqi': (0, 500),
    'pm25': (0, 600),
    'pm10': (0, 800),
    'co': (0, 30000),
    'no2': (0, 1000),
    'so2': (0, 1000),
    'o3': (0, 800),
    'temp': (5, 45),
    'rh': (0, 100),
    'pres': (950, 1050),
    'wind_spd': (0, 50),
    'clouds': (0, 100),
    'precip': (0, 200),
    'dewpt': (-10, 40)
}

# Các cột cần kiểm tra dữ liệu đóng băng (loại trừ hằng số hợp lệ)
FROZEN_CHECK_COLS = ['aqi', 'pm25', 'pm10', 'co', 'no2', 'so2', 'o3', 
                     'temp', 'rh', 'pres', 'wind_spd', 'dewpt']

# Các cột đầu vào cần giữ lại từ dữ liệu thô
INPUT_COLUMNS = [
    'timestamp_local',
    # Chất lượng không khí
    'aqi', 'co', 'no2', 'o3', 'pm10', 'pm25', 'so2',
    # Thời tiết
    'temp', 'rh', 'pres', 'wind_spd', 'wind_dir', 
    'clouds', 'precip', 'pod', 'dewpt'
]


# ============================================================================
# STEP 0: WIND COMPONENTS CALCULATION
# ============================================================================

def create_wind_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo wind_sin và wind_cos từ wind_dir.
    
    Args:
        df: DataFrame with wind_dir column (degrees)
    
    Returns:
        DataFrame with wind_sin and wind_cos columns
    """
    if 'wind_dir' not in df.columns:
        print("⚠️  Warning: wind_dir column not found! Skipping wind components.")
        return df
    
    df_copy = df.copy()
    
    # Convert degrees to radians
    wind_dir_rad = np.deg2rad(df_copy['wind_dir'])
    
    # Calculate sin and cos components
    df_copy['wind_sin'] = np.sin(wind_dir_rad)
    df_copy['wind_cos'] = np.cos(wind_dir_rad)
    
    print(f"✅ Created wind_sin and wind_cos from wind_dir")
    
    return df_copy


# ============================================================================
# STEP 1: TIME RANGE VALIDATION
# ============================================================================

def validate_time_range(df: pd.DataFrame, city_name: str) -> pd.DataFrame:
    """
    Validate time range và tạo full hourly index.
    
    Args:
        df: DataFrame with timestamp_local column
        city_name: Name of city for logging
    
    Returns:
        DataFrame with full hourly index from START_DATE to END_DATE
    """
    print(f"\n📅 [{city_name}] Time Range Validation:")
    
    # Create full hourly index
    full_index = pd.date_range(start=START_DATE, end=END_DATE, freq='h')
    
    # Parse timestamp
    df['timestamp_local'] = pd.to_datetime(df['timestamp_local'])
    
    # Check coverage
    actual_min = df['timestamp_local'].min()
    actual_max = df['timestamp_local'].max()
    
    print(f"   Expected: {START_DATE} to {END_DATE}")
    print(f"   Actual:   {actual_min} to {actual_max}")
    print(f"   Expected hours: {len(full_index):,}")
    print(f"   Actual hours:   {len(df):,}")
    
    # Set index and reindex to full range
    df.set_index('timestamp_local', inplace=True)
    df_reindexed = df.reindex(full_index)
    
    missing_count = df_reindexed.isnull().any(axis=1).sum()
    print(f"   Missing hours:  {missing_count:,} ({missing_count/len(full_index)*100:.2f}%)")
    
    return df_reindexed


# ============================================================================
# STEP 2: FROZEN DATA DETECTION + FLAGGING
# ============================================================================

def detect_frozen_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect frozen sensor data và tạo is_frozen flag.
    
    Frozen = std = 0 trong 24h liên tục
    
    Returns:
        DataFrame with 'is_frozen' column (0 = normal, 1 = frozen)
    """
    print(f"\n❄️  Frozen Data Detection:")
    
    df_clean = df.copy()
    df_clean['date'] = df_clean.index.date
    
    # Initialize flag column
    df_clean['is_frozen'] = 0
    
    frozen_details = []
    
    for col in FROZEN_CHECK_COLS:
        if col not in df_clean.columns:
            continue
            
        # Calculate daily std
        daily_std = df_clean.groupby('date')[col].transform('std')
        
        # Mark frozen (std = 0 or very close to 0)
        frozen_mask = (daily_std < 1e-6)
        frozen_count = frozen_mask.sum()
        
        if frozen_count > 0:
            df_clean.loc[frozen_mask, 'is_frozen'] = 1
            frozen_details.append(f"{col}: {frozen_count}")
    
    total_frozen = df_clean['is_frozen'].sum()
    print(f"   Total frozen flags: {total_frozen:,}")
    if frozen_details:
        print(f"   Details: {', '.join(frozen_details[:5])}")
    
    return df_clean.drop(columns=['date'])


# ============================================================================
# STEP 3: OUTLIER DETECTION + FLAGGING
# ============================================================================

def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect outliers và tạo is_outlier flag.
    
    Returns:
        DataFrame with 'is_outlier' column (0 = normal, 1 = outlier)
    """
    print(f"\n🚨 Outlier Detection:")
    
    df_clean = df.copy()
    
    # Initialize flag column
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
    
    total_outliers = df_clean['is_outlier'].sum()
    print(f"   Total outlier flags: {total_outliers:,}")
    if outlier_details:
        print(f"   Details: {', '.join(outlier_details[:5])}")
    
    return df_clean


# ============================================================================
# STEP 4: DUPLICATE REMOVAL
# ============================================================================

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate timestamps, keep first occurrence."""
    before = len(df)
    df_clean = df[~df.index.duplicated(keep='first')]
    after = len(df_clean)
    removed = before - after
    
    if removed > 0:
        print(f"\n🔄 Duplicate Removal: {removed} duplicates removed")
    
    return df_clean


# ============================================================================
# STEP 5: MISSING VALUES IMPUTATION
# ============================================================================


def impute_missing_values(df: pd.DataFrame, location: str) -> pd.DataFrame:
    """
    Điền giá trị thiếu sử dụng phương pháp điền khuyết kết hợp.

    Chiến lược hai giai đoạn:
    1. Nội suy tuyến tính theo thời gian cho khoảng trống nhỏ (≤6h)
    2. K-Láng Giềng Gần Nhất (KNN) cho khoảng trống lớn và mẫu phức tạp

    Args:
        df: DataFrame có giá trị thiếu (phải có DatetimeIndex)
        location: Tên địa điểm để ghi log

    Returns:
        DataFrame đã điền đầy đủ giá trị thiếu
    """
    print(f"\n🔍 [{location}] Điền Khuyết Giá Trị (Hai Giai Đoạn):")
    
    # Kiểm tra giá trị thiếu ban đầu
    original_nans = df.isnull().sum().sum()
    total_values = df.size
    print(f"   Giá trị thiếu ban đầu: {original_nans:,} ({original_nans / total_values * 100:.2f}%)")
    
    if original_nans == 0:
        print(f"   ✓ Không có giá trị thiếu!")
        return df
    
    df_clean = df.copy()
    
    # Tách cột số và không phải số
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df_clean.select_dtypes(exclude=[np.number]).columns
    
    # Giai đoạn 1: Nội suy tuyến tính theo thời gian cho khoảng trống nhỏ (≤6 giờ)
    # Chỉ áp dụng cho cột số
    print(f"   Giai đoạn 1: Nội suy tuyến tính (giới hạn = 6h)...")
    df_numeric = df_clean[numeric_cols].copy()
    df_numeric = df_numeric.interpolate(method='time', limit=6)
    
    # Cập nhật lại cột số
    df_clean[numeric_cols] = df_numeric
    
    after_linear = df_clean.isnull().sum().sum()
    filled_by_linear = original_nans - after_linear
    print(f"   ✓ Đã điền bằng nội suy: {filled_by_linear:,} giá trị")
    
    # Giai đoạn 2: Điền khuyết KNN cho khoảng trống còn lại
    if after_linear > 0:
        print(f"   Giai đoạn 2: Điền khuyết KNN cho khoảng trống còn lại...")
        print(f"   ⏳ Đang xử lý KNN (k=12, có thể mất vài phút)...")
        
        # Chỉ lấy cột số
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_for_knn = df_clean[num_cols]
        
        # Loại bỏ cột hoàn toàn thiếu (nếu có)
        cols_to_drop = df_for_knn.columns[df_for_knn.isnull().all()]
        if len(cols_to_drop) > 0:
            print(f"   ⚠️  Bỏ qua {len(cols_to_drop)} cột thiếu hoàn toàn: {list(cols_to_drop)}")
            df_for_knn = df_for_knn.drop(columns=cols_to_drop)
        
        # Áp dụng điền khuyết KNN với progress bar
        from sklearn.impute import KNNImputer
        print(f"   📊 Đang tính toán khoảng cách và tìm láng giềng gần nhất...")
        
        with tqdm(total=100, desc="   KNN Imputation", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
            imputer = KNNImputer(n_neighbors=12, weights='distance')
            pbar.update(20)  # Khởi tạo
            df_imputed_values = imputer.fit_transform(df_for_knn)
            pbar.update(80)  # Hoàn thành
        
        # Tái tạo DataFrame
        df_imputed = pd.DataFrame(
            df_imputed_values,
            columns=df_for_knn.columns,
            index=df_clean.index
        )
        
        # Cập nhật cột số trong df_clean
        df_clean[df_imputed.columns] = df_imputed
        
        filled_by_knn = after_linear - df_clean.isnull().sum().sum()
        print(f"   ✓ Đã điền bằng KNN: {filled_by_knn:,} giá trị")
    else:
        filled_by_knn = 0
    
    # Tổng kết cuối cùng
    missing_after = df_clean.isnull().sum().sum()
    total_filled = filled_by_linear + filled_by_knn
    print(f"   TỔNG ĐÃ ĐIỀN: {total_filled:,} giá trị ({total_filled / original_nans * 100:.1f}% số thiếu)")
    print(f"   Còn thiếu: {missing_after:,}")
    
    return df_clean



# ============================================================================
# STEP 6: FEATURE ENGINEERING
# ============================================================================

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features."""
    df_copy = df.copy()
    df_copy["hour"] = df_copy.index.hour
    df_copy["day"] = df_copy.index.day
    df_copy["month"] = df_copy.index.month
    df_copy["year"] = df_copy.index.year
    return df_copy


def create_wind_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create wind-based features."""
    df_copy = df.copy()
    if 'wind_spd' in df_copy.columns and 'wind_sin' in df_copy.columns:
        df_copy["spd_wind_sin"] = df_copy["wind_spd"] * df_copy["wind_sin"]
        df_copy["spd_wind_cos"] = df_copy["wind_spd"] * df_copy["wind_cos"]
    return df_copy


def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create weather-based features."""
    df_copy = df.copy()
    
    # Check required columns
    if 'wind_spd' in df_copy.columns and 'dewpt' in df_copy.columns:
        df_copy["is_stagnant"] = np.where(
            (df_copy["wind_spd"] < 1.5) & (df_copy["dewpt"] == 0), 1, 0
        )
    
    if 'dewpt' in df_copy.columns and 'temp' in df_copy.columns:
        df_copy["ah"] = (6.112 * np.exp((17.67 * df_copy["dewpt"]) / 
                         (df_copy["dewpt"] + 243.5)) * 2.1674 / 
                         (273.15 + df_copy["temp"]))
        df_copy["dtr"] = df_copy["temp"] - df_copy["dewpt"]
    
    if 'temp' in df_copy.columns and 'rh' in df_copy.columns:
        df_copy["dpd"] = df_copy["temp"] - df_copy["rh"]
    
    if 'pm25' in df_copy.columns and 'pm10' in df_copy.columns:
        df_copy["ratio_pm"] = df_copy["pm25"] / (df_copy["pm10"] + 1e-6)
    
    if 'o3' in df_copy.columns and 'co' in df_copy.columns:
        df_copy["o"] = df_copy["o3"] + df_copy["co"]
    
    if 'hour' in df_copy.columns:
        df_copy["rush_hour"] = np.where(
            ((df_copy["hour"] >= 6) & (df_copy["hour"] <= 9)) | 
            ((df_copy["hour"] >= 16) & (df_copy["hour"] <= 19)), 
            1, 0
        )
    
    if 'month' in df_copy.columns:
        df_copy["season_sin"] = np.sin(2 * np.pi * df_copy["month"] / 12)
    
    return df_copy


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag-based features (no groupby since single location)."""
    df_copy = df.copy()
    
    if 'pm25' not in df_copy.columns:
        return df_copy
    
    df_copy["delta_pm25"] = df_copy["pm25"].diff().fillna(0)
    df_copy["ma_pm25_4"] = df_copy["pm25"].rolling(window=4, min_periods=1).mean()
    
    if 'precip' in df_copy.columns:
        df_copy["rain_sum_6"] = df_copy["precip"].rolling(window=6, min_periods=1).sum()
    
    if 'wind_spd' in df_copy.columns:
        df_copy["w_pm25"] = df_copy["wind_spd"] * df_copy["pm25"]
    
    for lag in [1, 3, 6, 24]:
        df_copy[f"pm25_lag_{lag}"] = df_copy["pm25"].shift(lag).fillna(0)
    
    return df_copy


def create_all_features(df: pd.DataFrame, location: str) -> pd.DataFrame:
    """Create all features."""
    print(f"\n🔨 [{location}] Feature Engineering:")
    
    original_cols = len(df.columns)
    
    df = create_time_features(df)
    df = create_wind_features(df)
    df = create_weather_features(df)
    df = create_lag_features(df)
    
    new_cols = len(df.columns) - original_cols
    print(f"   Created {new_cols} new features")
    print(f"   Total columns: {len(df.columns)}")
    
    return df


# ============================================================================
# STEP 7: DISTRIBUTION ANALYSIS
# ============================================================================

def analyze_distribution(df: pd.DataFrame) -> Dict:
    """Analyze temporal distribution for train/val/test split."""
    print("\n📊 Distribution Analysis:")
    
    start = df.index.min()
    end = df.index.max()
    total_days = (end - start).days
    
    print(f"   Time Range: {start} to {end}")
    print(f"   Total Days: {total_days}")
    print(f"   Total Hours: {len(df):,}")
    
    # Suggest chronological split
    train_end = start + pd.Timedelta(days=int(total_days * 0.7))
    val_end = start + pd.Timedelta(days=int(total_days * 0.85))
    
    print(f"\n   📅 Suggested Split (Chronological):")
    print(f"      Train: {start.date()} to {train_end.date()} (70%)")
    print(f"      Val:   {train_end.date()} to {val_end.date()} (15%)")
    print(f"      Test:  {val_end.date()} to {end.date()} (15%)")
    
    return {'train_end': train_end, 'val_end': val_end}


# ============================================================================
# STEP 8: MAIN PIPELINE FOR SINGLE CITY
# ============================================================================

def process_single_city(city_code: str, data_dir: str = "data") -> pd.DataFrame:
    """
    Process data for a single city through complete pipeline.
    
    Loads raw air_quality and weather_hourly files, merges them, then processes.
    
    Args:
        city_code: 'CT', 'HCM', or 'HN'
        data_dir: Directory containing city subdirectories
    
    Returns:
        Processed DataFrame ready for merging
    """
    city_mapping = {
        'CT': ('CanTho', 'Cantho'),
        'HCM': ('HoChiMinh', 'HCM'),
        'HN': ('Hanoi', 'Hanoi')
    }
    
    location_name, folder_name = city_mapping[city_code]
    
    print("=" * 60)
    print(f"🌍 Processing: {location_name}")
    print("=" * 60)
    
    # Load air quality data
    air_quality_file = os.path.join(data_dir, folder_name, f"air_quality_{city_code}.csv")
    weather_file = os.path.join(data_dir, folder_name, f"weather_hourly_{city_code}.csv")
    
    print(f"   Loading air quality: {air_quality_file}")
    df_air = pd.read_csv(air_quality_file)
    print(f"   ✓ Air quality: {len(df_air):,} rows, {len(df_air.columns)} columns")
    
    print(f"   Loading weather: {weather_file}")
    df_weather = pd.read_csv(weather_file)
    print(f"   ✓ Weather: {len(df_weather):,} rows, {len(df_weather.columns)} columns")
    
    # Merge air quality and weather on timestamp
    # Assume both have timestamp column (might be 'timestamp_local' or 'timestamp')
    timestamp_col_air = 'timestamp_local' if 'timestamp_local' in df_air.columns else 'timestamp'
    timestamp_col_weather = 'timestamp_local' if 'timestamp_local' in df_weather.columns else 'timestamp'
    
    # Standardize timestamp column name
    df_air = df_air.rename(columns={timestamp_col_air: 'timestamp_local'})
    df_weather = df_weather.rename(columns={timestamp_col_weather: 'timestamp_local'})
    
    # Merge on timestamp
    print(f"   Merging air quality and weather data...")
    df = pd.merge(df_air, df_weather, on='timestamp_local', how='outer', suffixes=('', '_weather'))
    print(f"   ✓ Merged: {len(df):,} rows, {len(df.columns)} columns")
    
    # Filter to keep only specified columns
    available_cols = [col for col in INPUT_COLUMNS if col in df.columns]
    missing_cols = [col for col in INPUT_COLUMNS if col not in df.columns]
    
    if missing_cols:
        print(f"   ⚠️  Missing columns: {missing_cols}")
    
    df = df[available_cols]
    print(f"   Filtered to {len(available_cols)} columns")
    
    # Remove duplicates on timestamp_local (keep first occurrence)
    print(f"   Checking for duplicate timestamps...")
    df['timestamp_local'] = pd.to_datetime(df['timestamp_local'])
    before_dup = len(df)
    df = df.drop_duplicates(subset=['timestamp_local'], keep='first')
    after_dup = len(df)
    if before_dup > after_dup:
        print(f"   ✓ Removed {before_dup - after_dup} duplicate timestamps")
    else:
        print(f"   ✓ No duplicates found")
    
    # Step 0: Create wind components (wind_sin, wind_cos)
    df = create_wind_components(df)
    
    # Step 1: Time validation
    df = validate_time_range(df, location_name)
    
    # Step 2: Frozen detection
    df = detect_frozen_data(df)
    
    # Step 3: Outlier detection
    df = detect_outliers(df)
    
    # Step 4: Duplicates
    df = remove_duplicates(df)
    
    # Step 5: Missing imputation
    df = impute_missing_values(df, location_name)
    
    # Step 6: Feature engineering
    df = create_all_features(df, location_name)
    
    # Add location column
    df['location'] = location_name
    
    print(f"\n✅ [{location_name}] Processing complete!")
    print(f"   Final shape: {df.shape}")
    
    return df


# ============================================================================
# FINAL: MERGE AND SAVE
# ============================================================================

def merge_and_save(df_ct: pd.DataFrame, df_hcm: pd.DataFrame, df_hn: pd.DataFrame,
                   output_file: str = "data/clean_data_all.csv") -> pd.DataFrame:
    """Merge all cities and save final output."""
    print("\n" + "=" * 60)
    print("📦 Merging All Cities")
    print("=" * 60)
    
    # Reset index to column
    df_ct = df_ct.reset_index().rename(columns={'index': 'timestamp_local'})
    df_hcm = df_hcm.reset_index().rename(columns={'index': 'timestamp_local'})
    df_hn = df_hn.reset_index().rename(columns={'index': 'timestamp_local'})
    
    # Concat
    df_all = pd.concat([df_ct, df_hcm, df_hn], ignore_index=True)
    
    print(f"\n   Final Data Summary:")
    print(f"   Shape: {df_all.shape}")
    print(f"   Missing: {df_all.isnull().sum().sum()}")
    print(f"   Locations: {df_all['location'].nunique()}")
    print(f"\n   Location distribution:")
    for loc, count in df_all['location'].value_counts().items():
        pct = count / len(df_all) * 100
        print(f"      {loc}: {count:,} ({pct:.1f}%)")
    
    # Check quality flags
    frozen_count = df_all['is_frozen'].sum()
    outlier_count = df_all['is_outlier'].sum()
    print(f"\n   Quality Flags:")
    print(f"      Frozen: {frozen_count:,} ({frozen_count/len(df_all)*100:.2f}%)")
    print(f"      Outliers: {outlier_count:,} ({outlier_count/len(df_all)*100:.2f}%)")
    
    # Distribution analysis
    analyze_distribution(df_all.set_index('timestamp_local'))
    
    # Save
    df_all.to_csv(output_file, index=False)
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n💾 Saved to: {output_file}")
    print(f"   File size: {file_size:.2f} MB")
    
    return df_all


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def run_complete_pipeline(data_dir: str = "data",
                          output_file: str = "data/clean_data_all.csv") -> pd.DataFrame:
    """
    Run complete preprocessing pipeline for all 3 cities.
    
    Returns:
        Final merged DataFrame
    """
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  🌍 COMPLETE PREPROCESSING PIPELINE - 3 CITIES  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    
    # Process each city
    df_ct = process_single_city('CT', data_dir)
    df_hcm = process_single_city('HCM', data_dir)
    df_hn = process_single_city('HN', data_dir)
    
    # Merge and save
    df_final = merge_and_save(df_ct, df_hcm, df_hn, output_file)
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return df_final


if __name__ == "__main__":
    # Run pipeline
    df = run_complete_pipeline()
    
    # Display summary
    print("\n📊 FINAL SUMMARY:")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n   First 5 rows:")
    print(df.head())
