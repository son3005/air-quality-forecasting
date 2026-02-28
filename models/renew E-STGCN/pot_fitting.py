"""
POT (Peaks Over Threshold) Fitting - Chuyển đổi từ R sang Python.
Fit phân phối Generalized Pareto Distribution (GPD) cho từng trạm
để tính toán các tham số xi (shape) và sigma (scale) phục vụ cho EVT Loss.
"""

import numpy as np
import pandas as pd
import os
from scipy.stats import genpareto


def fit_gpd_per_station(data_dir, threshold=60.0, target_col='aqi', num_stations=32):
    """
    Fit GPD cho từng trạm dựa trên dữ liệu vượt ngưỡng (exceedances).
    
    Tương đương code R:
        fgpd(train[[colnames(train)[i]]], u)$xi   -> shape parameter
        fgpd(train[[colnames(train)[i]]], u)$sigmau -> scale parameter
    
    Args:
        data_dir (str): Đường dẫn tới thư mục chứa các file station_*.csv đã clean.
        threshold (float): Ngưỡng u (threshold), giá trị AQI >= u được coi là "cực trị".
                           Bài báo E-STGCN dùng 60.
        target_col (str): Tên cột mục tiêu để fit GPD (ví dụ: 'aqi', 'pm25').
        num_stations (int): Số lượng trạm.
    
    Returns:
        pot_results: np.ndarray shape (num_stations, 2) -> [xi, sigma] cho mỗi trạm.
    """
    xi_list = []
    sigma_list = []
    
    for i in range(1, num_stations + 1):
        file_path = os.path.join(data_dir, f"station_{i}.csv")
        df = pd.read_csv(file_path)
        
        # Lấy dữ liệu mục tiêu
        values = df[target_col].dropna().values
        
        # Lấy 70% đầu làm training set (giống split ratio trong data_loader)
        train_size = int(len(values) * 0.7)
        train_values = values[:train_size]
        
        # Lọc các giá trị vượt ngưỡng (exceedances)
        exceedances = train_values[train_values > threshold] - threshold
        
        if len(exceedances) < 10:
            # Nếu không đủ dữ liệu cực trị, dùng giá trị mặc định an toàn
            print(f"  Station {i}: Chỉ có {len(exceedances)} exceedances (< 10). Dùng giá trị mặc định.")
            xi_list.append(0.1)   # shape parameter mặc định
            sigma_list.append(10.0)  # scale parameter mặc định
        else:
            try:
                # Fit GPD bằng scipy.stats.genpareto
                # genpareto.fit trả về (c, loc, scale) với c = xi (shape parameter)
                c, loc, scale = genpareto.fit(exceedances, floc=0)  # floc=0 vì exceedances đã trừ threshold
                
                xi_list.append(c)       # shape parameter (xi)
                sigma_list.append(scale) # scale parameter (sigma)
                
                print(f"  Station {i}: xi={c:.4f}, sigma={scale:.4f} ({len(exceedances)} exceedances)")
            except Exception as e:
                print(f"  Station {i}: Lỗi fit GPD: {e}. Dùng giá trị mặc định.")
                xi_list.append(0.1)
                sigma_list.append(10.0)
    
    pot_results = np.column_stack([xi_list, sigma_list])
    return pot_results


def save_pot_results(pot_results, output_path):
    """Lưu kết quả POT fitting ra CSV."""
    df = pd.DataFrame(pot_results, columns=['xi', 'sigma'])
    df.index.name = 'station'
    df.to_csv(output_path)
    print(f"\nPOT results saved to: {output_path}")
    print(f"Shape: {pot_results.shape}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "..", "..", "data", "clean")
    output_path = os.path.join(current_dir, "..", "..", "data", "pot_results.csv")
    
    print("=" * 60)
    print("POT (Peaks Over Threshold) Fitting - GPD Parameters")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Threshold: 60.0 AQI")
    print(f"Target column: aqi")
    print()
    
    pot_results = fit_gpd_per_station(
        data_dir=data_dir,
        threshold=60.0,
        target_col='aqi',
        num_stations=32
    )
    
    save_pot_results(pot_results, output_path)
    
    print(f"\nTrung bình xi: {pot_results[:, 0].mean():.4f}")
    print(f"Trung bình sigma: {pot_results[:, 1].mean():.4f}")
