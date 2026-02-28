import torch
import numpy as np
import scipy.sparse as sp

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    
    # Avoid division by zero
    rowsum[rowsum == 0] = 1.0 
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    
    # Return to dense or tensor
    return torch.FloatTensor(normalized_adj.toarray())

def construct_adaptive_adjacency(node_emb, relu_activation=True):
    """Construct adjacency from learnable node embeddings"""
    if relu_activation:
        A = torch.relu(node_emb @ node_emb.T)
    else:
        A = node_emb @ node_emb.T
    return torch.softmax(A, dim=-1)

def build_correlation_graph(df, target='pm25', threshold=0.3):
    """
    Build adjacency matrix from DataFrame based on Pearson Correlation.
    Expects df with ['timestamp_local', 'province', target]
    """
    pivot = df.pivot_table(index='timestamp_local', columns='province', values=target)
    corr = pivot.corr(method='pearson').abs().values
    
    # Apply threshold
    adj = (corr > threshold).astype(np.float32)
    
    # Remove self loops
    np.fill_diagonal(adj, 0)
    
    return normalize_adj(adj)
