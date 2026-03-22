"""
models/shared/metrics.py

Common metrics + inverse transform utilities dùng chung cho tất cả pipelines.
"""
import os
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SCALER_DIR = 'data/normalized'


def inverse_pm25(y_norm, station_id, scaler_dir=SCALER_DIR):
    """Inverse transform PM2.5 từ normalized space về µg/m³."""
    scaler_path = os.path.join(scaler_dir, f'scalers_{station_id}.pkl')
    if not os.path.exists(scaler_path):
        return y_norm
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    method_tuple = scalers.get('pm25')
    if not method_tuple:
        return y_norm
    method, sc = method_tuple[:2]
    y_inv = sc.inverse_transform(np.array(y_norm).reshape(-1, 1)).flatten()
    if 'log1p' in method:
        y_inv = np.expm1(y_inv)
    return y_inv


def compute_mape(y_true, y_pred):
    """Mean Absolute Percentage Error (bỏ qua các giá trị ≤ 1.0)."""
    mask = y_true > 1.0
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def get_metrics(y_true, y_pred):
    """Trả về (RMSE, MAE, R², MAPE)."""
    return (
        np.sqrt(mean_squared_error(y_true, y_pred)),
        mean_absolute_error(y_true, y_pred),
        r2_score(y_true, y_pred),
        compute_mape(y_true, y_pred),
    )
