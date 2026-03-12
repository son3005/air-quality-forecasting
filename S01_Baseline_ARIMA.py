import pandas as pd
import numpy as np
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

def inverse_pm25(scaled_vals, scaler_tuple):
    """
    Hoàn tác Normalized Pipeline V3: LOG_ROBUST_ONLY_NO_CLIP
    """
    name, sc = scaler_tuple
    unscaled = sc.inverse_transform(np.array(scaled_vals).reshape(-1, 1)).flatten()
    real_vals = np.expm1(unscaled) # Inverse của log1p
    return real_vals

def get_metrics(y_true, y_pred):
    """
    Trả về: RMSE, MAE, MAPE (%)
    """
    mask = y_true > 0 # Tránh chia cho 0
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return rmse, mae, mape

def run_arima_baseline(station_id, test_samples=1000, order=(2, 0, 2)):
    # 1. Đọc Data Normalized và Scalers Picker Cấu Hình Khôi Phục
    df = pd.read_csv(f'data/normalized/norm_station_{station_id}.csv', parse_dates=['timestamp']).set_index('timestamp')
    with open(f'data/normalized/scalers_{station_id}.pkl', 'rb') as f:
        scalers = pickle.load(f)
    sc_tuple = scalers['pm25']
    
    train_data = df[df['split'] == 'train']['pm25'].values
    test_data = df[df['split'] == 'test']['pm25'].values
    
    # Dùng 60 ngày cuối của Train để làm History rễ huấn luyện
    history = list(train_data[-24*60:]) 
    
    actual_t1, pred_t1 = [], []
    actual_t3, pred_t3 = [], []
    
    # Đánh giá độ dài nhất định vào khoảng Test Set (Mặc định 1000 giờ - hơn 1 tháng)
    test_eval_len = min(len(test_data) - 3, test_samples)
    
    # 2. Khởi tạo ARIMA Core
    model = ARIMA(history, order=order)
    fitted = model.fit()
    
    print(f"[*] Trạm {station_id:02d} | Đang đẩy Rolling Forecast (N={test_eval_len})...", end="")
    for t in range(test_eval_len):
        # Nhanh chóng Forecast 3 step tương lai
        forecast = fitted.forecast(steps=3)
        
        pred_t1.append(forecast[0])   # t+1
        pred_t3.append(forecast[2])   # t+3
        actual_t1.append(test_data[t])
        actual_t3.append(test_data[t+2])
        
        # Trượt chuỗi lịch sử lên bằng test data mới, KHÔNG fit lại để tăng tốc
        fitted = fitted.append([test_data[t]], refit=False)
        
    # 3. Phục hồi đơn vị Ô nhiễm gốc µg/m³ bằng Inverse Transform
    pred_t1_real = inverse_pm25(pred_t1, sc_tuple)
    actual_t1_real = inverse_pm25(actual_t1, sc_tuple)
    
    pred_t3_real = inverse_pm25(pred_t3, sc_tuple)
    actual_t3_real = inverse_pm25(actual_t3, sc_tuple)
    
    # 4. Tính toán Metrics thi đâu RMSE, MAE, MAPE
    t1_rmse, t1_mae, t1_mape = get_metrics(actual_t1_real, pred_t1_real)
    t3_rmse, t3_mae, t3_mape = get_metrics(actual_t3_real, pred_t3_real)
    print(" Xong!")
    
    return {
        "t+1": (t1_rmse, t1_mae, t1_mape),
        "t+3": (t3_rmse, t3_mae, t3_mape)
    }

stations = [1, 3, 4, 7, 9, 12, 13, 15, 16, 17, 18, 24, 27, 29, 31, 32]

results_t1 = []
results_t3 = []

# Thu nhỏ phạm vi dự báo rolling test_samples để demo nhanh (sẽ mất khoảng 1-2 phút)
for sid in stations:
    try:
        res = run_arima_baseline(station_id=sid, test_samples=500, order=(2, 0, 2))
        results_t1.append(res['t+1'])
        results_t3.append(res['t+3'])
    except Exception as e:
        print(f"Trạm {sid} gặp sự cố: {e}")

# Trích Tổng Lực 16 Trạm
t1_avg = np.mean(results_t1, axis=0)
t3_avg = np.mean(results_t3, axis=0)

print("\n" + "="*50)
print("🏆 KẾT QUẢ ĐIỂM CHUẨN S01_BASELINE (Trung bình 16 Trạm)")
print("="*50)
print(f"=> Tầm nhìn T+1 H 🕒 | RMSE: {t1_avg[0]:.2f} | MAE: {t1_avg[1]:.2f} | MAPE: {t1_avg[2]:.2f}%")
print(f"=> Tầm nhìn T+3 H 🕔 | RMSE: {t3_avg[0]:.2f} | MAE: {t3_avg[1]:.2f} | MAPE: {t3_avg[2]:.2f}%")
print("="*50)

