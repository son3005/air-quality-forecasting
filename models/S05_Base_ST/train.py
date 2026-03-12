import torch
import torch.nn as nn
import torch.optim as optim
import time
from dataset import get_dataloaders
from graph import get_adjacency_matrix
from model import STGCN
from utils import get_metrics, print_benchmark_table
import numpy as np

def train_model(seq_len=24, pred_len=12, epochs=20, batch_size=32):
    print("="*60)
    print(f"Training STGCN - Input: {seq_len}h, Output: {pred_len}h")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(seq_len=seq_len, pred_len=pred_len, batch_size=batch_size)
    
    # Graph
    adj = get_adjacency_matrix()
    adj = torch.tensor(adj, dtype=torch.float32).to(device)
    
    # Model
    num_nodes = 16
    num_features = 15 # 15 raw features
    model = STGCN(num_nodes=num_nodes, num_features=num_features, seq_len=seq_len, pred_len=pred_len)
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        t1 = time.time()
        model.train()
        train_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            out = model(x, adj)
            
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                
                out = model(x, adj)
                loss = criterion(out, y)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        t2 = time.time()
        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {t2-t1:.2f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            
    # Evaluation on Test set
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x, adj)
            
            all_preds.append(out.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
    all_preds = np.concatenate(all_preds, axis=0) # (samples, pred_len, num_nodes)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics for specific horizons
    horizons = [1, 6, 12]
    results = []
    
    for h in horizons:
        if h <= pred_len:
            # Index is h-1
            idx = h - 1
            y_t = all_targets[:, idx, :]
            y_p = all_preds[:, idx, :]
            
            mets = get_metrics(y_t, y_p)
            mets['in_horizon'] = f't-{seq_len}h'
            mets['out_horizon'] = f't+{h}'
            mets['hp'] = f'E: {epochs}, B: {batch_size}, L: MSE'
            
            # Ghi chú đánh giá độ hiệu quả động
            if h <= 2:
                if mets['R2'] > 0.7:
                    mets['note'] = "Vừa vặn làm Baseline (Ngắn hạn ổn)"
                else:
                    mets['note'] = "Tệ ở ngắn hạn"
            elif h <= 6:
                if mets['R2'] > 0.5:
                    mets['note'] = "Rớt hiệu suất ở trung hạn (Oversmoothing?)"
                else:
                    mets['note'] = "Hiệu năng giảm mạnh ở trung hạn"
            else:
                mets['note'] = "Quá tệ cho dài hạn, cần Feature Engineering/Attention"
                
            results.append(mets)
            
    print("\n\n--- TEST RESULTS ---")
    print_benchmark_table(results)
    
    return results

if __name__ == '__main__':
    train_model(seq_len=24, pred_len=12, epochs=100, batch_size=64)
