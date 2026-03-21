import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_mape(y_true, y_pred):
    mask = y_true != 0
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def compute_csi(y_true, y_pred, threshold=1.0):
    hits = np.sum((y_true >= threshold) & (y_pred >= threshold))
    false_alarms = np.sum((y_true < threshold) & (y_pred >= threshold))
    misses = np.sum((y_true >= threshold) & (y_pred < threshold))
    
    denominator = hits + false_alarms + misses
    if denominator == 0:
        return 0.0
    return hits / denominator

def get_metrics(y_true, y_pred):
    yt = y_true.flatten()
    yp = y_pred.flatten()
    
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    r2 = r2_score(yt, yp)
    mape = compute_mape(yt, yp)
    csi = compute_csi(yt, yp, threshold=np.mean(yt) + np.std(yt))
    
    score = (0.5 * r2) + (0.5 * csi) - (0.1 * rmse)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'CSI': csi,
        'Score': score
    }

def print_benchmark_table(results):
    print("-" * 190)
    print(f"{'Nhóm KH':<15} | {'Tên Mô Hình':<15} | {'Input Horizon':<15} | {'Output Horizon':<15} | {'RMSE ↓':<8} | {'MAE ↓':<8} | {'MAPE ↓':<8} | {'% R^2 ↑':<8} | {'CSI (Trúng) ↑':<14} | {'Score ↑':<8} | {'Hyper-Parameters (Settings ghi nhớ)':<37} | {'Ghi chú / Trạng thái':<40}")
    print("-" * 190)
    for res in results:
        print(f"{'S07_Full_SOTA':<15} | {'ASTGCN':<15} | {res['in_horizon']:<15} | {res['out_horizon']:<15} | {res['RMSE']:<8.4f} | {res['MAE']:<8.4f} | {res['MAPE']:<7.2f}% | {res['R2']*100:<7.2f}% | {res['CSI']:<14.4f} | {res['Score']:<8.4f} | {res['hp']:<37} | {res.get('note', 'OK'):<40}")
    print("-" * 190)
