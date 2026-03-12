import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np

from dataset import get_dataloaders
from graph import get_adjacency_matrix
from model import PureGCN
from utils import get_metrics, print_benchmark_table

def train_model(epochs=100, batch_size=32):
    print("="*60)
    print(f"Training Pure GCN - Input: t, Output: t+1")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataloaders - seq_len=1, pred_len=1 by default inside
    train_loader, val_loader, test_loader, num_features = get_dataloaders(batch_size=batch_size)
    print(f"Features dimension: {num_features}")
    
    # Graph
    adj = get_adjacency_matrix()
    adj = torch.tensor(adj, dtype=torch.float32).to(device)
    
    num_nodes = 16
    model = PureGCN(num_nodes=num_nodes, num_features=num_features)
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
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics
    mets = get_metrics(all_targets, all_preds)
    mets['in_horizon'] = 't'
    mets['out_horizon'] = 't+1h'
    mets['hp'] = f'E: {epochs}, B: {batch_size}, L: MSE, Opt: Adam'
    mets['note'] = 'Xem khối bụi bay từ trạm này sang trạm kia lân cận như thế nào'
    
    print("\n\n--- TEST RESULTS ---")
    print_benchmark_table([mets])
    
    return [mets]

if __name__ == '__main__':
    train_model(epochs=100, batch_size=32)
