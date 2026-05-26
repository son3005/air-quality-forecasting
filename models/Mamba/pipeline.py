"""
models/Mamba/pipeline.py

Mamba Pipeline — Per-Horizon Training with Region-based Evaluation.
Uses the Hugging Face MambaModel wrapper for multi-pollutant forecasting.
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════
SELECTED_STATIONS = [1, 4, 5, 16, 17, 27, 7, 18, 24, 30, 31, 32]
REGIONS = {
    'north': [1, 4, 5, 16, 17, 27],
    'south': [7, 18, 24, 30, 31, 32],
}
POLLUTANTS = ['pm25', 'pm10', 'co', 'o3', 'no2', 'so2']
BLOCK = 'block30'
DATA_DIR = f'data/split/{BLOCK}'
SEQ_LEN = 48
HORIZONS = [1, 3, 6, 12, 24]
BATCH_SIZE = 64
EPOCHS = 7
LR = 1e-3
PATIENCE = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Mamba model configurations: (d_model, d_state, d_conv, e_layers)
MODEL_CONFIG = {
    1:  {'north': (64, 16, 4, 2), 'south': (64, 16, 4, 2)},
    3:  {'north': (64, 16, 4, 2), 'south': (64, 16, 4, 2)},
    6:  {'north': (64, 16, 4, 2), 'south': (64, 16, 4, 2)},
    12: {'north': (64, 16, 4, 2), 'south': (64, 16, 4, 2)},
    24: {'north': (64, 16, 4, 2), 'south': (64, 16, 4, 2)},
}

# ══════════════════════════════════════════════════════════════════════════
# IMPORTS AND SHARING
# ══════════════════════════════════════════════════════════════════════════
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from metrics import inverse_pollutant, get_per_pollutant_metrics
from dataset import load_station_data, MultiStationDataset

from model import HFMambaModel

print(f"[*] Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════
# LOSS
# ══════════════════════════════════════════════════════════════════════════
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.huber = nn.HuberLoss(delta=1.0)
        self.mae = nn.L1Loss()

    def forward(self, pred, target):
        return self.alpha * self.huber(pred, target) + (1 - self.alpha) * self.mae(pred, target)


# ══════════════════════════════════════════════════════════════════════════
# TRAIN + EVALUATE
# ══════════════════════════════════════════════════════════════════════════
def run():
    print("=" * 70)
    print(f"  Mamba Pipeline — Per-Horizon | Block: {BLOCK}")
    print(f"  Horizons: {HORIZONS}")
    print("=" * 70)

    all_results = []
    total_start = time.time()

    for h in HORIZONS:
        print(f"\n{'='*60}")
        print(f"  HORIZON T+{h}h")
        print(f"{'='*60}")

        for r_name, sids in REGIONS.items():
            print(f"\n  [{r_name.upper()}] T+{h}h ({len(sids)} stations)...")

            # Load datasets
            train_data = load_station_data(sids, 'train', DATA_DIR)
            val_data = load_station_data(sids, 'val', DATA_DIR)
            test_data = load_station_data(sids, 'test', DATA_DIR)

            train_ds = MultiStationDataset(train_data, sids, SEQ_LEN, h)
            val_ds = MultiStationDataset(val_data, sids, SEQ_LEN, h)
            test_ds = MultiStationDataset(test_data, sids, SEQ_LEN, h)

            print(f"    Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)} | "
                  f"Variates={train_ds.num_variates}")

            train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
            val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, pin_memory=True)
            test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, pin_memory=True)

            # Model Configuration
            d_model, d_state, d_conv, e_layers = MODEL_CONFIG[h][r_name]
            num_variates = train_ds.num_variates
            num_nodes = len(sids)
            num_targets = num_nodes * len(POLLUTANTS)

            model = HFMambaModel(
                seq_len=SEQ_LEN,
                pred_len=1,
                enc_in=num_variates,
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                e_layers=e_layers,
                use_norm=True
            ).to(DEVICE)

            n_params = sum(p.numel() for p in model.parameters())
            print(f"    Params: {n_params:,}")

            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
            criterion = CombinedLoss(alpha=0.7)

            # Training
            best_val = float('inf')
            patience_cnt = 0
            save_dir = os.path.join('models_saved', BLOCK, 'Mamba')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{r_name}_t{h}.pth')

            for epoch in range(EPOCHS):
                t0 = time.time()
                model.train()
                losses = []
                for bx, by in train_loader:
                    bx = bx.to(DEVICE, non_blocking=True)
                    by = by.to(DEVICE, non_blocking=True)

                    optimizer.zero_grad()
                    out = model(bx)  # Output: (B, 1, num_variates)
                    preds = out[:, 0, -num_targets:]  # Use last N*6 features (Targets)
                    loss = criterion(preds, by)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    losses.append(loss.item())
                
                scheduler.step()

                # Validation
                model.eval()
                vlosses = []
                all_preds, all_trues = [], []
                with torch.no_grad():
                    for bx, by in val_loader:
                        bx = bx.to(DEVICE, non_blocking=True)
                        by = by.to(DEVICE, non_blocking=True)
                        out = model(bx)
                        preds = out[:, 0, -num_targets:]
                        vlosses.append(criterion(preds, by).item())
                        all_preds.append(preds.cpu().numpy())
                        all_trues.append(by.cpu().numpy())

                tl, vl = np.mean(losses), np.mean(vlosses)
                dt = time.time() - t0
                
                # Inverse transform on Val to get metrics
                y_pred_val = np.concatenate(all_preds, axis=0)
                y_true_val = np.concatenate(all_trues, axis=0)
                
                preds_r = y_pred_val.reshape(-1, num_nodes, len(POLLUTANTS))
                trues_r = y_true_val.reshape(-1, num_nodes, len(POLLUTANTS))
                batch_preds_inv = np.zeros_like(preds_r)
                batch_trues_inv = np.zeros_like(trues_r)
                
                for node_idx, sid in enumerate(sids):
                    for p_idx, pol in enumerate(POLLUTANTS):
                        batch_preds_inv[:, node_idx, p_idx] = inverse_pollutant(preds_r[:, node_idx, p_idx], sid, pol)
                        batch_trues_inv[:, node_idx, p_idx] = inverse_pollutant(trues_r[:, node_idx, p_idx], sid, pol)
                
                val_metrics = get_per_pollutant_metrics(batch_trues_inv.reshape(-1, num_targets), 
                                                        batch_preds_inv.reshape(-1, num_targets), 
                                                        POLLUTANTS)
                pm25_val = val_metrics['pm25']
                
                print(f"    Ep {epoch+1:02d}/{EPOCHS} | T: {tl:.4f} | V: {vl:.4f} | PM2.5 RMSE: {pm25_val['RMSE']:.2f}, R2: {pm25_val['R2']*100:.2f}% | {dt:.1f}s")
                
                # Save history
                history_path = os.path.join(save_dir, f'{r_name}_t{h}_history.csv')
                hist_record = {
                    'epoch': epoch + 1, 'train_loss': tl, 'val_loss': vl, 'time': dt
                }
                for pol in POLLUTANTS:
                    m = val_metrics[pol]
                    hist_record[f'{pol}_rmse'] = m['RMSE']
                    hist_record[f'{pol}_mae'] = m['MAE']
                    hist_record[f'{pol}_r2'] = m['R2']
                    hist_record[f'{pol}_mape'] = m['MAPE']
                
                # Append to file or create new
                df_hist = pd.DataFrame([hist_record])
                if epoch == 0:
                    df_hist.to_csv(history_path, index=False)
                else:
                    df_hist.to_csv(history_path, mode='a', header=False, index=False)

                if vl < best_val:
                    best_val = vl
                    torch.save(model.state_dict(), save_path)
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= PATIENCE:
                        print(f"    Early stop at ep {epoch+1}")
                        break

            # ═══════════════════════════════════════════════════════════
            # EVALUATION
            # ═══════════════════════════════════════════════════════════
            model.load_state_dict(torch.load(save_path, weights_only=True, map_location=DEVICE))
            model.eval()

            all_preds, all_trues = [], []
            with torch.no_grad():
                for bx, by in test_loader:
                    bx = bx.to(DEVICE, non_blocking=True)
                    out = model(bx)
                    preds = out[:, 0, -num_targets:].cpu().numpy()  # (B, N*6)
                    trues = by.numpy()  # (B, N*6)

                    # Inverse transform per station per pollutant
                    preds_r = preds.reshape(-1, num_nodes, len(POLLUTANTS))
                    trues_r = trues.reshape(-1, num_nodes, len(POLLUTANTS))
                    
                    batch_preds_inv = np.zeros_like(preds_r)
                    batch_trues_inv = np.zeros_like(trues_r)
                    
                    for node_idx, sid in enumerate(sids):
                        for p_idx, pol in enumerate(POLLUTANTS):
                            batch_preds_inv[:, node_idx, p_idx] = inverse_pollutant(preds_r[:, node_idx, p_idx], sid, pol)
                            batch_trues_inv[:, node_idx, p_idx] = inverse_pollutant(trues_r[:, node_idx, p_idx], sid, pol)
                            
                    all_preds.append(batch_preds_inv.reshape(-1, num_targets))
                    all_trues.append(batch_trues_inv.reshape(-1, num_targets))

            y_true = np.concatenate(all_trues, axis=0)
            y_pred = np.concatenate(all_preds, axis=0)
            metrics = get_per_pollutant_metrics(y_true, y_pred, POLLUTANTS)
            
            # Print metrics for PM2.5 specifically as quick summary
            pm25_m = metrics['pm25']
            print(f"    [{r_name.upper()}] T+{h:2d} | PM2.5 RMSE={pm25_m['RMSE']:.2f} | MAE={pm25_m['MAE']:.2f} | "
                  f"R2={pm25_m['R2']*100:.2f}% | MAPE={pm25_m['MAPE']:.2f}%")

            # Save full metrics
            all_results.append({
                'region': r_name, 'horizon': f'T+{h}',
                'metrics': metrics,
                'n_test': len(y_true), 'train_time': round(time.time() - total_start, 2)
            })

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  FINAL BENCHMARK — Mamba (Per-Horizon)")
    print("=" * 90)
    print(f"{'Region':<10} {'Horizon':<8} {'Pol':<6} {'RMSE':>8} {'MAE':>8} {'R2 %':>8} {'MAPE %':>10}")
    print("-" * 70)
    for r in all_results:
        for pol in POLLUTANTS:
            m = r['metrics'][pol]
            print(f"{r['region']:<10} {r['horizon']:<8} {pol:<6} {m['RMSE']:>8.2f} {m['MAE']:>8.2f} "
                  f"{m['R2']*100:>7.2f}% {m['MAPE']:>9.2f}%")
        print("-" * 70)

    print("\nAGGREGATED PM2.5 (weighted by test size):")
    for h in HORIZONS:
        hr = [r for r in all_results if r['horizon'] == f'T+{h}']
        if hr:
            total = sum(r['n_test'] for r in hr)
            w = lambda key: sum(r['metrics']['pm25'][key]*r['n_test'] for r in hr) / total
            print(f"  T+{h:<3d}  RMSE={w('RMSE'):.2f}  MAE={w('MAE'):.2f}  "
                  f"R2={w('R2')*100:.2f}%  MAPE={w('MAPE'):.2f}%")

    total_time = time.time() - total_start
    print(f"\n  Total training time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print("=" * 70)

    return all_results, total_time


if __name__ == '__main__':
    run()
