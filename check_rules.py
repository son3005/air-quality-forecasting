import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

SELECTED = [1,2,3,4,5,6,7,16,17,18,24,27,28,30,31,32]
NORTH = [1,2,3,4,5,6,16,17,27,28]
SOUTH = [7,18,24,30,31,32]

all_air = {}
for sid in SELECTED:
    df = pd.read_csv(f'data/raw/air/air_{sid}.csv')
    df['datetime'] = pd.to_datetime(df['timestamp_local'])
    df = df.set_index('datetime').sort_index()
    all_air[sid] = df['pm25'].copy()

info_map = {
    1:'Cau Giay',2:'Thanh Xuan',3:'Tay Ho',4:'Tho Xuan',5:'Gia Lam',
    6:'Ba Vi',7:'Q1-HCM',16:'Do Son',17:'Hong Bang',18:'Ninh Kieu',
    24:'Bien Hoa',27:'Nam Dinh',28:'TP.TH',30:'Vinh Long',31:'Tra Vinh',32:'Rach Gia'
}

# =============================================
# BLOCK 5 NGAY: 3d TRAIN | 1d VAL | 1d TEST
# =============================================
TRAIN_DAYS = 3
VAL_DAYS = 1
TEST_DAYS = 1
BLOCK_HOURS = (TRAIN_DAYS + VAL_DAYS + TEST_DAYS) * 24  # 120h

print("=" * 80)
print("KIEM NGHIEM THUC TE: BLOCK 5 NGAY (3d Train | 1d Val | 1d Test)")
print("=" * 80)

# --- 1. NAIVE BASELINE ---
print("\n--- 1. NAIVE BASELINE: Copy gia tri cuoi Train -> du doan ca Test day ---")
print(f"{'Station':15s} | {'Cluster':5s} | {'Blocks':>6s} | {'MAE_val':>8s} | {'MAE_test':>9s} | {'Mean':>7s} | {'MAE/Mean_test':>13s} | Verdict")
print("-" * 95)

results = {}
for sid in SELECTED:
    pm = all_air[sid].dropna().values
    n = len(pm)
    
    train_h = TRAIN_DAYS * 24
    val_h = VAL_DAYS * 24
    test_h = TEST_DAYS * 24
    
    mae_vals = []
    mae_tests = []
    mae_persist_tests = []  # persistence: last known value
    mae_mean_tests = []     # mean of train as prediction
    blocks = 0
    
    for start in range(0, n - BLOCK_HOURS, BLOCK_HOURS):
        block = pm[start:start + BLOCK_HOURS]
        if len(block) < BLOCK_HOURS:
            break
        
        train = block[:train_h]
        val = block[train_h:train_h+val_h]
        test = block[train_h+val_h:]
        
        # Naive 1: Persistence (last train value)
        last_train_val = train[-1]
        mae_v = np.mean(np.abs(val - last_train_val))
        mae_t = np.mean(np.abs(test - last_train_val))
        
        # Naive 2: Train mean
        train_mean = np.mean(train)
        mae_mean_t = np.mean(np.abs(test - train_mean))
        
        mae_vals.append(mae_v)
        mae_tests.append(mae_t)
        mae_mean_tests.append(mae_mean_t)
        blocks += 1
    
    avg_mae_val = np.mean(mae_vals)
    avg_mae_test = np.mean(mae_tests)
    avg_mae_mean_test = np.mean(mae_mean_tests)
    mean_pm = all_air[sid].mean()
    ratio = avg_mae_test / mean_pm * 100
    
    cluster = "BAC" if sid in NORTH else "NAM"
    verdict = "OK" if ratio > 20 else ("CAUTION" if ratio > 12 else "LEAK!")
    
    results[sid] = {
        'blocks': blocks, 'mae_val': avg_mae_val, 'mae_test': avg_mae_test,
        'mae_mean_test': avg_mae_mean_test, 'mean': mean_pm, 'ratio': ratio
    }
    
    print(f"S{sid:<2d} {info_map[sid]:12s} | {cluster:5s} | {blocks:6d} | {avg_mae_val:8.2f} | {avg_mae_test:9.2f} | {mean_pm:7.1f} | {ratio:11.1f}%  | {verdict}")

# --- 2. CORRELATION TRAIN vs TEST ---
print("\n--- 2. TUONG QUAN GIUA TRAIN MEAN vs TEST MEAN (moi block) ---")
print("Neu correlation cao -> Test bi anh huong boi Train -> leak")
print(f"{'Station':15s} | {'Cluster':5s} | {'Corr(train_mean, test_mean)':>28s} | Verdict")
print("-" * 70)

for sid in SELECTED:
    pm = all_air[sid].dropna().values
    n = len(pm)
    train_h = TRAIN_DAYS * 24
    val_h = VAL_DAYS * 24
    
    train_means = []
    test_means = []
    
    for start in range(0, n - BLOCK_HOURS, BLOCK_HOURS):
        block = pm[start:start + BLOCK_HOURS]
        if len(block) < BLOCK_HOURS:
            break
        train = block[:train_h]
        test = block[train_h+val_h:]
        train_means.append(np.mean(train))
        test_means.append(np.mean(test))
    
    corr = np.corrcoef(train_means, test_means)[0, 1]
    cluster = "BAC" if sid in NORTH else "NAM"
    verdict = "OK" if corr < 0.85 else ("CAUTION" if corr < 0.95 else "LEAK!")
    print(f"S{sid:<2d} {info_map[sid]:12s} | {cluster:5s} | {corr:28.4f} | {verdict}")

# --- 3. SO SANH VOI PHUONG AN 6H ---
print("\n--- 3. SO SANH: Block 6h vs Block 5 ngay ---")
print(f"{'Station':15s} | {'6h MAE/Mean':>12s} | {'5d MAE/Mean':>12s} | {'Gain':>8s} | Comment")
print("-" * 75)

# 6h results (from previous analysis)
sixh_ratios = {
    1: 20.0, 2: 20.0, 7: 6.4, 18: 15.6, 24: 6.4, 32: 9.2
}

for sid in [1, 2, 7, 18, 24, 32]:
    r6h = sixh_ratios.get(sid, 0)
    r5d = results[sid]['ratio']
    gain = r5d - r6h
    comment = "5d tot hon nhieu!" if gain > 5 else ("5d tot hon" if gain > 0 else "Khong cai thien")
    cluster = "BAC" if sid in NORTH else "NAM"
    print(f"S{sid:<2d} {info_map[sid]:12s} | {r6h:10.1f}%  | {r5d:10.1f}%  | {gain:+6.1f}%  | {comment}")

# --- 4. PHAN BO SEASONAL TRONG BLOCK ---
print("\n--- 4. TEST COVERAGE: Block test co cover du cac mua khong? ---")
print("Dem so block roi vao tung mua (DJF=Dong, MAM=Xuan, JJA=Ha, SON=Thu)")

pm1 = all_air[1]
dates = pm1.index
total_blocks = 0
season_counts = {'DJF': 0, 'MAM': 0, 'JJA': 0, 'SON': 0}

for start_idx in range(0, len(dates) - BLOCK_HOURS, BLOCK_HOURS):
    test_start = dates[start_idx + (TRAIN_DAYS+VAL_DAYS)*24]
    month = test_start.month
    if month in [12, 1, 2]:
        season_counts['DJF'] += 1
    elif month in [3, 4, 5]:
        season_counts['MAM'] += 1
    elif month in [6, 7, 8]:
        season_counts['JJA'] += 1
    else:
        season_counts['SON'] += 1
    total_blocks += 1

print(f"  Tong blocks: {total_blocks}")
for s, c in season_counts.items():
    pct = c / total_blocks * 100
    print(f"  {s}: {c:3d} blocks ({pct:.1f}%)")

print("\n" + "=" * 80)
print("KET LUAN TONG HOP")
print("=" * 80)
