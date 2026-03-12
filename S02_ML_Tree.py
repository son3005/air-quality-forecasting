import pandas as pd
import numpy as np
import pickle
import glob
import warnings
from tqdm import tqdm
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')
print("XGBoost Version:", xgb.__version__)

def inverse_pm25(scaled_vals, scaler_tuple):
    """
    Hoàn tác Normalized Pipeline V3: LOG_ROBUST_ONLY_NO_CLIP
    """
    name, sc = scaler_tuple
    unscaled = sc.inverse_transform(np.array(scaled_vals).reshape(-1, 1)).flatten()
    real_vals = np.expm1(unscaled) # Inverse của log1p
    return real_vals

def get_metrics_r2(y_true, y_pred):
    """
    Trả về: RMSE, MAE, MAPE (%), R2 Score
    """
    mask = y_true > 0 # Tránh chia cho 0
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, mape, r2

def create_xgboost_dataset(horizon=1, seq_len=24):
    """
    Tạo Super-Dataset từ 16 Trạm cho thuật toán XGBoost gặm nhấm. 
    XGBoost xử lý Input 2D (Tabular) thay vì 3D, tiến hành băm phẳng (Flatten) $t-24h$.
    Horizon: Số giờ mốc tương lai (VD: 1, 6, 12)
    """
    all_X_train, all_y_train = [], []
    all_X_test, all_y_test = [], []
    
    # Danh sách chuẩn Scaler gốc của 16 trạm để lúc sau bung ra tính điểm
    scalers_dict = {}
    target_col_idx = None
    
    file_paths = sorted(glob.glob('data/normalized/norm_station_*.csv'))
    print(f"[*] Đóng gói DataSet Học Máy (Horizon = T+{horizon}) cho {len(file_paths)} Trạm...")
    
    for path in tqdm(file_paths):
        station_id = int(path.split('_')[-1].replace('.csv', ''))
        with open(f'data/normalized/scalers_{station_id}.pkl', 'rb') as f:
            scalers_dict[station_id] = pickle.load(f)['pm25']
            
        df = pd.read_csv(path, parse_dates=['timestamp']).sort_values('timestamp')
        
        # Tách nhãn split để chia tập
        split_flags = df['split'].values
        
        # Loại bỏ các cột phi logic ra khỏi phép học
        drop_cols = ['timestamp', 'split', 'province', 'district', 'station_id']
        feature_cols = [c for c in df.columns if c not in drop_cols]
        
        # Xác định vị trí Index của PM2.5 để chọc Target Y ra
        if target_col_idx is None:
            target_col_idx = feature_cols.index('pm25')
            
        data_matrix = df[feature_cols].values
        
        # One-Hot Encoding cho Tọa độ Trạm vào XGBoost
        station_one_hot = np.zeros(16)
        # (Giả định ID trạm chạy dọc từ 1-32, nếu đánh số rải rác ta có thể hash hòm hòm, ở đây ta dùng trick lookup đơn giản)
        if station_id <= 32: 
            station_one_hot[station_id % 16] = 1.0
        
        X_station, y_station, splits_station = [], [], []
        
        # Xoáy Slidung Window 24 giờ
        total_length = len(data_matrix)
        for i in range(total_length - seq_len - horizon):
            # Input X: Lấy t-24h của TẤT CẢ các cột và Băm phẳng 1D Array
            window = data_matrix[i : i + seq_len]
            x_flat = window.flatten() 
            
            # Gắn thêm biển số Trạm (Station Taggings)
            x_feature = np.concatenate([x_flat, station_one_hot])
            
            # Target Y: PM25 tại t+Horizon
            y_target = data_matrix[i + seq_len + horizon - 1, target_col_idx]
            
            # Phân loại Train hay Test dựa theo timesteps nằm ở rãnh T cuối cùng
            current_split = split_flags[i + seq_len + horizon - 1]
            
            if current_split == 'train' or current_split == 'val':
                all_X_train.append(x_feature)
                all_y_train.append(y_target)
            elif current_split == 'test':
                # Khéo léo tuồn ID Trạm vào cuối array Y để lát bung Scaler trả lại đồ thị
                all_X_test.append(x_feature)
                all_y_test.append((y_target, station_id))
                
    return np.array(all_X_train), np.array(all_y_train), np.array(all_X_test), all_y_test, scalers_dict

def train_eval_xgboost(horizon):
    print(f"\n{'='*50}\n🚀 BẮT ĐẦU TRAINING XGBOOST CHUỖI T+{horizon} HORIZON\n{'='*50}")
    X_train, y_train, X_test, test_tuples, scalers_dict = create_xgboost_dataset(horizon=horizon, seq_len=24)
    
    print(f"\n[+] Kích cỡ Tập Train (Mẫu): {X_train.shape[0]:,} Khung giờ (24h) | {X_train.shape[1]} Cột Flat Features")
    print(f"[+] Kích cỡ Tập Test (Mẫu) : {X_test.shape[0]:,} Khung giờ (24h)")
    
    # Khởi tạo mô hình Học Máy Cây Siêu Tốc (Objective MSE - Regression)
    # Dùng tree_method='hist' vì lượng Dataset là khủng khiếp (Tránh tràn RAM)
    xgb_model = xgb.XGBRegressor(
        n_estimators=100, 
        max_depth=6, 
        learning_rate=0.08,
        objective='reg:squarederror',
        tree_method='hist',  # Bật tăng tốc Histograms cho Big Data Tabular
        n_jobs=-1,
        random_state=42
    )
    
    print("[*] Training Mô hình...")
    xgb_model.fit(X_train, y_train)
    print("[*] Xong! Bắt đầu Forecast Tập Test...")
    
    y_pred_scaled = xgb_model.predict(X_test)
    
    # ---- Tính Toán Inverse Transform Cho Từng Trạm Trên Tập Test Khổng Lồ ----
    y_true_real = []
    y_pred_real = []
    
    for i, (scaled_true, sid) in enumerate(test_tuples):
        sc_tuple = scalers_dict[sid]
        
        # Phục hồi giá trị Thực tế
        real_t = inverse_pm25([scaled_true], sc_tuple)[0]
        # Phục hồi giá trị Neural Net Dự đoán
        real_p = inverse_pm25([y_pred_scaled[i]], sc_tuple)[0]
        
        y_true_real.append(real_t)
        y_pred_real.append(real_p)
        
    # Chấm Bảng Điểm
    rmse, mae, mape, r2 = get_metrics_r2(np.array(y_true_real), np.array(y_pred_real))
    
    print(f"\n🏆 KẾT QUẢ XGBOOST BĂNG TẦN T+{horizon}")
    print(f"-> RMSE : {rmse:.2f}")
    print(f"-> MAE  : {mae:.2f}")
    print(f"-> MAPE : {mape:.2f}%")
    print(f"-> R^2  : {r2:.4f}")
    
    return rmse, mae, mape, r2

# Chạy Khảo sát Đánh giá trên 3 mốc (T+1, T+6, T+12)
scores_h1 = train_eval_xgboost(horizon=1)
scores_h6 = train_eval_xgboost(horizon=6)
scores_h12 = train_eval_xgboost(horizon=12)
