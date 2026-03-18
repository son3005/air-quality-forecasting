import numpy as np
import pandas as pd
import os
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SELECTED_STATIONS = [1, 3, 4, 7, 9, 12, 13, 15, 16, 17, 18, 24, 27, 29, 31, 32]
REGIONS = {
    'north': [1, 3, 4, 13, 15, 16, 17, 27, 29],
    'south': [7, 9, 12, 18, 24, 31, 32],
}

def inverse_transform(y_t, y_p, scaler_dir='data/normalized'):
    # y_t, y_p shape: (samples, pred_len, num_nodes)
    y_t_inv = np.zeros_like(y_t)
    y_p_inv = np.zeros_like(y_p)
    
    for i, sid in enumerate(SELECTED_STATIONS):
        scaler_path = os.path.join(scaler_dir, f'scalers_{sid}.pkl')
        if not os.path.exists(scaler_path):
            y_t_inv[:, :, i] = y_t[:, :, i]
            y_p_inv[:, :, i] = y_p[:, :, i]
            continue
            
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
            
        method_tuple = scalers.get('pm25')
        if method_tuple is not None:
            method = method_tuple[0]
            sc = method_tuple[1]
            try:
                shape_t = y_t[:, :, i].shape
                yt_n = y_t[:, :, i].reshape(-1, 1)
                yp_n = y_p[:, :, i].reshape(-1, 1)
                
                yt_n_inv = sc.inverse_transform(yt_n)
                yp_n_inv = sc.inverse_transform(yp_n)
                
                if 'log1p' in method:
                    yt_n_inv = np.expm1(yt_n_inv)
                    yp_n_inv = np.expm1(yp_n_inv)
                    
                y_t_inv[:, :, i] = yt_n_inv.reshape(shape_t)
                y_p_inv[:, :, i] = yp_n_inv.reshape(shape_t)
            except Exception:
                y_t_inv[:, :, i] = y_t[:, :, i]
                y_p_inv[:, :, i] = y_p[:, :, i]
        else:
            y_t_inv[:, :, i] = y_t[:, :, i]
            y_p_inv[:, :, i] = y_p[:, :, i]
            
    return y_t_inv, y_p_inv

def compute_mape(y_true, y_pred):
    mask = y_true > 1.0
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def get_metrics(y_true, y_pred):
    yt = y_true.flatten()
    yp = y_pred.flatten()
    
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    r2 = r2_score(yt, yp)
    mape = compute_mape(yt, yp)
    
    return rmse, mae, r2, mape
