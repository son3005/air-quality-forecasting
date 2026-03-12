import pandas as pd
import numpy as np
import pickle
import warnings
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import glob
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# Thiết lập Device chạy GPU (CUDA) nếu có, nếu không thì CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Đang chạy Huấn luyện PyTorch trên thiết bị: {device.type.upper()}")

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

def build_sequences(data_matrix, y_idx, seq_len=24, pred_len=24):
    """
    Cắt chuỗi dữ liệu Thành Khối 3D cho Input (X) và Mảng 2D cho Output (Y)
    X: [Batch, T_past, Features]
    Y: [Batch, T_future]
    """
    X_list, Y_list = [], []
    total_len = len(data_matrix)
    for i in range(total_len - seq_len - pred_len + 1):
        X_list.append(data_matrix[i : i + seq_len])
        # Lấy riêng PM2.5 ở tương lai trải dài từ T+1 đến T+24
        Y_list.append(data_matrix[i + seq_len : i + seq_len + pred_len, y_idx])
        
    return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32)

def load_station_data(station_id, seq_len=24, pred_len=24):
    # Load file Normalized V2
    df = pd.read_csv(f'data/normalized/norm_station_{station_id}.csv', parse_dates=['timestamp']).sort_values('timestamp')
    
    # Đọc Scaler PM25
    with open(f'data/normalized/scalers_{station_id}.pkl', 'rb') as f:
        sc_tuple = pickle.load(f)['pm25']
        
    # Băm tách ranh giới file dựa trên nhãn chuẩn Split V2
    # Tập Validation dùng chung với Train để check Loss (Early Stopping)
    # Tập Test thuần túy cách ly
    train_val_df = df[df['split'].isin(['train', 'val'])]
    test_df = df[df['split'] == 'test']
    
    # Bỏ các cột rác
    drop_cols = ['timestamp', 'split', 'province', 'district', 'station_id']
    features = [c for c in df.columns if c not in drop_cols]
    target_idx = features.index('pm25')
    
    # Build Tensor Trains
    X_train, Y_train = build_sequences(train_val_df[features].values, target_idx, seq_len, pred_len)
    X_test, Y_test = build_sequences(test_df[features].values, target_idx, seq_len, pred_len)
    
    # Cutout nhỏ lại 15% của mảng đầu cho Validation Early Stop
    num_val = int(len(X_train) * 0.15)
    X_val, Y_val = X_train[-num_val:], Y_train[-num_val:]
    X_train, Y_train = X_train[:-num_val], Y_train[:-num_val]
    
    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), sc_tuple, len(features)

class PM25_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_horizon=24, dropout=0.2):
        super(PM25_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_horizon)
        
    def forward(self, x):
        # x hình dạng (Batch, T, Features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Rút trích khối tế bào Trạng thái cuối cùng của khung T-24 để bắn ra Tương lai
        last_hidden = lstm_out[:, -1, :]
        # Dự báo 1 lần ra 24 vạch tương lai vector (Batch, Output_Horizon)
        prediction = self.fc(last_hidden)
        return prediction

def train_isolate_station(station_id, epochs=100, batch_size=64, patience=10):
    print(f"\n[+] Kích hoạt Lò Cháy (Training) Trạm Độc Lập SID: {station_id:02d}...")
    
    # 1. Load Data
    (X_tr, y_tr), (X_v, y_v), (X_te, y_te), sc_tuple, input_dim = load_station_data(station_id, seq_len=24, pred_len=24)
    
    # Chuyển vào Dataloader Torch
    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_v), torch.tensor(y_v)), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_te), torch.tensor(y_te)), batch_size=batch_size, shuffle=False)
    
    # 2. Build Cấu trúc Cụm LSTM
    model = PM25_LSTM(input_size=input_dim, hidden_size=64, num_layers=2, output_horizon=24, dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 3. Vòng lặp Ép Xung (Epochs Loop) với Early Stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
        train_loss /= len(train_loader.dataset)
        
        # Kiểm thử Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                preds = model(bx)
                loss = criterion(preds, by)
                val_loss += loss.item() * bx.size(0)
        val_loss /= len(val_loader.dataset)
        
        # Cơ chế Chống Overfitting Ghi đè RAM (Early Stop)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"best_lstm_s{station_id}.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    -> Early stopping kích hoạt tại Epoch {epoch+1:03d}! Val Loss gãy đáy: {best_val_loss:.4f}")
                break
                
    # 4. Đo Lường Khảo Sát Test Mù (Blind Test Predict)
    model.load_state_dict(torch.load(f"best_lstm_s{station_id}.pth", weights_only=True))
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            preds = model(bx).cpu().numpy()
            all_preds.append(preds)
            all_trues.append(by.numpy())
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
    
    # 5. Phân hóa mảng Array 24 mốc Tương lai, đập tan Scaler Trả về Đơn vị PM2.5
    # Vị trí Index: T+1 -> 0, T+6 -> 5, T+12 -> 11, T+24 -> 23
    metrics_results = {}
    
    for h_label, h_idx in zip(['T+1', 'T+6', 'T+12', 'T+24'], [0, 5, 11, 23]):
        true_horizon = inverse_pm25(all_trues[:, h_idx], sc_tuple)
        pred_horizon = inverse_pm25(all_preds[:, h_idx], sc_tuple)
        
        rmse, mae, mape, r2 = get_metrics_r2(true_horizon, pred_horizon)
        metrics_results[h_label] = (rmse, mae, r2)
        
    return metrics_results


stations_list = [1, 3, 4, 7, 9, 12, 13, 15, 16, 17, 18, 24, 27, 29, 31, 32]
global_metrics = {'T+1': [], 'T+6': [], 'T+12': [], 'T+24': []}

# Chạy qua từng trạm. 
# LƯU Ý: Vòng lặp Full 16 Station qua 100 Epoch x 64 Batches sẽ tốn từ 10-30 phút tùy tốc độ CUDA máy bạn.
for sid in stations_list:
    res = train_isolate_station(sid, epochs=100, batch_size=128, patience=10) # Tăng batch=128 để giảm thiểu chèn ép CPU nếu không có GPU
    for h in global_metrics.keys():
        global_metrics[h].append(res[h])
        
print("\n" + "*"*60)
print("🏆 TỔNG ĐIỂM KẾT QUẢ BENCHMARK S03: LSTM (Trung bình Toàn Quốc 16 Node)")
print("*"*60)
for h in global_metrics.keys():
    # Tính trung bình các tuple (RMSE, MAE, R2)
    avg_scores = np.mean(global_metrics[h], axis=0)
    print(f"=> Băng tần Giờ {h:<5} | RMSE: {avg_scores[0]:.2f} | MAE: {avg_scores[1]:.2f} | R²: {avg_scores[2]:.4f}")
print("*"*60)
