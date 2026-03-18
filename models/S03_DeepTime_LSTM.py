import os
import pandas as pd
import numpy as np
import pickle
import warnings
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ══════════════════════════════════════════════════════════════════════════
# CONFIG & HORIZONS
# ══════════════════════════════════════════════════════════════════════════
HORIZONS = [1, 3, 6, 12, 24]
HORIZON_INDICES = [0, 2, 5, 11, 23]

REGIONS = {
    'north': [1, 3, 4, 13, 15, 16, 17, 27, 29],
    'south': [7, 9, 12, 18, 24, 31, 32],
}

def inverse_pm25(scaled_vals, scaler_tuple):
    if not scaler_tuple: return scaled_vals
    method, sc = scaler_tuple
    unscaled = sc.inverse_transform(np.array(scaled_vals).reshape(-1, 1)).flatten()
    if 'log1p' in method:
        return np.expm1(unscaled)
    return unscaled

def compute_mape(y_true, y_pred):
    mask = y_true > 1.0
    if mask.sum() == 0: return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def get_metrics(y_true, y_pred):
    return (
        np.sqrt(mean_squared_error(y_true, y_pred)),
        mean_absolute_error(y_true, y_pred),
        r2_score(y_true, y_pred),
        compute_mape(y_true, y_pred)
    )

def build_sequences(data_matrix, y_idx, seq_len=24, pred_len=24):
    """
    Cắt chuỗi dữ liệu (X: t-24h, Y: t+1h đến t+24h)
    """
    X_list, Y_list = [], []
    total_len = len(data_matrix)
    for i in range(total_len - seq_len - pred_len + 1):
        X_list.append(data_matrix[i : i + seq_len])
        Y_list.append(data_matrix[i + seq_len : i + seq_len + pred_len, y_idx])
        
    return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32)

def load_station_data(station_id, seq_len=24, pred_len=24):
    df = pd.read_csv(f'data/normalized/norm_station_{station_id}.csv', parse_dates=['timestamp']).sort_values('timestamp')
    
    with open(f'data/normalized/scalers_{station_id}.pkl', 'rb') as f:
        sc_tuple = pickle.load(f)['pm25']
        
    # Chronological Split
    ts = df['timestamp']
    is_train = ts < '2025-01-01'
    is_val   = (ts >= '2025-01-01') & (ts < '2025-05-01')
    is_test  = ts >= '2025-05-01'
    
    train_df = df[is_train]
    val_df = df[is_val]
    test_df = df[is_test]
    
    # Bỏ các cột rác
    drop_cols = ['timestamp', 'split', 'province', 'district', 'station_id']
    features = [c for c in df.columns if c not in drop_cols]
    target_idx = features.index('pm25')
    
    # Build Tensor Trains
    X_train, Y_train = build_sequences(train_df[features].values, target_idx, seq_len, pred_len)
    X_val, Y_val = build_sequences(val_df[features].values, target_idx, seq_len, pred_len)
    X_test, Y_test = build_sequences(test_df[features].values, target_idx, seq_len, pred_len)
    
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
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :] # Lấy state tại timestep cuối
        prediction = self.fc(last_hidden) # Chiếu ra 24 mốc tương lai
        return prediction

def train_isolate_station(station_id, epochs=100, batch_size=128, patience=10):
    print(f"    [*] Trạm {station_id:02d} | Đang nạp data và huấn luyện LSTM...")
    
    # 1. Load Data
    (X_tr, y_tr), (X_v, y_v), (X_te, y_te), sc_tuple, input_dim = load_station_data(station_id, seq_len=24, pred_len=24)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_v), torch.tensor(y_v)), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_te), torch.tensor(y_te)), batch_size=batch_size, shuffle=False)
    
    # 2. Build Cấu trúc Cụm LSTM
    model = PM25_LSTM(input_size=input_dim, hidden_size=64, num_layers=2, output_horizon=24, dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    best_val_loss = float('inf')
    patience_counter = 0
    save_path = f"data/models_saved/baseline_lstm_s{station_id}.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 3. Vòng lặp Ép Xung (Epochs Loop)
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
        train_loss /= max(len(train_loader.dataset), 1)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                preds = model(bx)
                loss = criterion(preds, by)
                val_loss += loss.item() * bx.size(0)
        val_loss /= max(len(val_loader.dataset), 1)
        
        # Early Stop
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    # 4. Đo Lường Khảo Sát Test Mù (Blind Test Predict)
    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            preds = model(bx).cpu().numpy()
            all_preds.append(preds)
            all_trues.append(by.numpy())
            
    if all_preds:
        all_preds = np.concatenate(all_preds, axis=0) # (N, 24)
        all_trues = np.concatenate(all_trues, axis=0) # (N, 24)
    else:
        all_preds = np.zeros((0, 24))
        all_trues = np.zeros((0, 24))
    
    # 5. Phân hóa mảng Array 24 mốc Tương lai
    metrics_results = {}
    for h, idx in zip(HORIZONS, HORIZON_INDICES):
        true_scaled = all_trues[:, idx] if len(all_trues) > 0 else []
        pred_scaled = all_preds[:, idx] if len(all_preds) > 0 else []
        
        true_real = inverse_pm25(true_scaled, sc_tuple)
        pred_real = inverse_pm25(pred_scaled, sc_tuple)
        metrics_results[h] = (true_real, pred_real)
        
    return metrics_results

def run_all():
    print("=" * 70)
    print("  S03: Deep Time LSTM Baseline (Fair Benchmark S09)")
    print("  Horizons: T+1, T+3, T+6, T+12, T+24")
    print(f"  Device: {device.type.upper()}")
    print("=" * 70)
    
    all_results = []
    
    for r_name, sids in REGIONS.items():
        print(f"\n[{r_name.upper()}] Processing {len(sids)} stations...")
        
        region_actuals = {h: [] for h in HORIZONS}
        region_preds = {h: [] for h in HORIZONS}
        
        for sid in sids:
            res = train_isolate_station(sid, epochs=100, batch_size=256, patience=15)
            for h in HORIZONS:
                region_actuals[h].extend(res[h][0])
                region_preds[h].extend(res[h][1])
                
        for h in HORIZONS:
            a_arr = np.array(region_actuals[h])
            p_arr = np.array(region_preds[h])
            rmse, mae, r2, mape = get_metrics(a_arr, p_arr)
            
            print(f"  [{r_name.upper()}] T+{h:<2d} | RMSE={rmse:.2f} | MAE={mae:.2f} | R2={r2*100:.2f}% | MAPE={mape:.2f}%")
            all_results.append({
                'region': r_name, 'horizon': f'T+{h}',
                'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape,
                'n_test': len(a_arr)
            })

    # Summary
    print("\n" + "=" * 70)
    print("  FINAL BENCHMARK — S03 Deep Time LSTM")
    print("=" * 70)
    print(f"{'Region':<10} {'Horizon':<8} {'RMSE':>8} {'MAE':>8} {'R2 %':>8} {'MAPE %':>10}")
    print("-" * 55)
    for r in all_results:
        print(f"{r['region']:<10} {r['horizon']:<8} {r['RMSE']:>8.2f} {r['MAE']:>8.2f} {r['R2']*100:>7.2f}% {r['MAPE']:>9.2f}%")

    print("\n" + "-" * 55)
    print("AGGREGATED (weighted by test size):")
    for h in HORIZONS:
        hr = [r for r in all_results if r['horizon'] == f'T+{h}']
        if hr:
            total = sum(r['n_test'] for r in hr)
            agg = lambda key: sum(r[key]*r['n_test'] for r in hr) / total
            print(f"  T+{h:<3d}  RMSE={agg('RMSE'):.2f}  MAE={agg('MAE'):.2f}  "
                  f"R2={agg('R2')*100:.2f}%  MAPE={agg('MAPE'):.2f}%")
    print("=" * 70)

if __name__ == '__main__':
    run_all()
