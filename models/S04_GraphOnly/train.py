import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import warnings

warnings.filterwarnings('ignore')

from dataset import get_dataloaders
from graph import get_adjacency_matrix
from model import PureGCN
from utils import get_metrics, inverse_transform, SELECTED_STATIONS, REGIONS

HORIZONS = [1, 3, 6, 12, 24]

def train_model(epochs=100, batch_size=64, patience=15):
    print("="*70)
    print(f"  S04: Graph-Only (PureGCN) Baseline (Fair Benchmark S09)")
    print(f"  Horizons: T+1, T+3, T+6, T+12, T+24")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Device: {device.type.upper()}")
    
    # Dataloader
    pred_len = 24
    train_loader, val_loader, test_loader, num_features = get_dataloaders(batch_size=batch_size, pred_len=pred_len)
    
    # Graph Adjacency
    adj = get_adjacency_matrix()
    adj = torch.tensor(adj, dtype=torch.float32).to(device)
    
    num_nodes = len(SELECTED_STATIONS)
    model = PureGCN(num_nodes=num_nodes, num_features=num_features, output_horizon=pred_len)
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    save_path = "data/models_saved/baseline_gcn_s04.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("[*] Bắt đầu Training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x, adj)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            
        train_loss /= max(len(train_loader.dataset), 1)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x, adj)
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                
        val_loss /= max(len(val_loader.dataset), 1)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    # ======================================================================
    # Evaluation
    # ======================================================================
    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()
    
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x, adj)
            all_preds.append(out.cpu().numpy())
            all_targets.append(y.numpy())
            
    if all_preds:
        all_preds = np.concatenate(all_preds, axis=0) # (samples, num_nodes, 24)
        all_targets = np.concatenate(all_targets, axis=0) # (samples, num_nodes, 24)
    else:
        all_preds = np.zeros((0, num_nodes, 24))
        all_targets = np.zeros((0, num_nodes, 24))
        
    all_targets_inv, all_preds_inv = inverse_transform(all_targets, all_preds)
    
    all_results = []
    
    for r_name, sids in REGIONS.items():
        print(f"\n[{r_name.upper()}] Processing {len(sids)} stations...")
        
        # Khớp ID trạm bằng Index
        idx_list = [SELECTED_STATIONS.index(sid) for sid in sids]
        region_actuals = all_targets_inv[:, idx_list, :] # (samples, region_nodes, 24)
        region_preds = all_preds_inv[:, idx_list, :]
        
        for h in HORIZONS:
            a_arr = region_actuals[:, :, h-1].flatten()
            p_arr = region_preds[:, :, h-1].flatten()
            
            rmse, mae, r2, mape = get_metrics(a_arr, p_arr)
            
            print(f"  [{r_name.upper()}] T+{h:<2d} | RMSE={rmse:.2f} | MAE={mae:.2f} | R2={r2*100:.2f}% | MAPE={mape:.2f}%")
            
            all_results.append({
                'region': r_name, 'horizon': f'T+{h}',
                'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape,
                'n_test': len(a_arr)
            })
            
    # Summary
    print("\n" + "=" * 70)
    print("  FINAL BENCHMARK — S04 GraphOnly (PureGCN)")
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
    train_model(epochs=100, batch_size=256)
