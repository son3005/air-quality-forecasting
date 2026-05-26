"""
models/shared/metrics.py

Common metrics + inverse transform utilities dùng chung cho tất cả pipelines.
"""
import os
import pickle
import sys
import types
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Monkeypatch for transformers to support torch.distributed.tensor.device_mesh on Windows/PyTorch 2.4.1
try:
    import torch.distributed.tensor
    if 'torch.distributed.tensor.device_mesh' not in sys.modules:
        m = types.ModuleType('torch.distributed.tensor.device_mesh')
        try:
            from torch.distributed.device_mesh import DeviceMesh
        except ImportError:
            from torch.distributed.tensor.device_mesh import DeviceMesh
        m.DeviceMesh = DeviceMesh
        sys.modules['torch.distributed.tensor.device_mesh'] = m
        if not hasattr(torch.distributed.tensor, 'device_mesh'):
            torch.distributed.tensor.device_mesh = m
except Exception:
    pass

SCALER_DIR = 'data/normalized'


_SCALER_CACHE = {}


def inverse_pollutant(y_norm, station_id, pollutant, scaler_dir=SCALER_DIR):
    """Inverse transform a pollutant từ normalized space về không gian gốc."""
    cache_key = (station_id, scaler_dir)
    if cache_key in _SCALER_CACHE:
        scalers = _SCALER_CACHE[cache_key]
    else:
        scaler_path = os.path.join(scaler_dir, f'scalers_{station_id}.pkl')
        if not os.path.exists(scaler_path):
            return y_norm
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
        _SCALER_CACHE[cache_key] = scalers
        
    method_tuple = scalers.get(pollutant)
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


def get_per_pollutant_metrics(y_true, y_pred, pollutants=['pm25', 'pm10', 'co', 'o3', 'no2', 'so2']):
    """
    Tính metrics cho từng chất.
    y_true, y_pred: numpy array shape (samples, num_stations * len(pollutants))
    """
    num_pols = len(pollutants)
    num_stations = y_true.shape[1] // num_pols
    
    y_true_r = y_true.reshape(-1, num_stations, num_pols)
    y_pred_r = y_pred.reshape(-1, num_stations, num_pols)
    
    results = {}
    for i, pol in enumerate(pollutants):
        yt = y_true_r[:, :, i].flatten()
        yp = y_pred_r[:, :, i].flatten()
        rmse, mae, r2, mape = get_metrics(yt, yp)
        results[pol] = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}
    return results
