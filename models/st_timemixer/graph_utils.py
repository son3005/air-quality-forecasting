"""
=============================================================================
Module: Graph Utilities — Xây dựng Adjacency Matrix (Ma trận kề)
=============================================================================
Vai trò:
  → Xây dựng ma trận A [N, N] thể hiện mối quan hệ giữa N tỉnh thành
  → A được dùng trong GCN (Phase 2) để lan truyền thông tin ô nhiễm

Có 3 phương pháp:
  1. Distance-based:   A_ij = exp(-d²/σ²)     — dựa trên khoảng cách GPS
  2. Correlation-based: A_ij = |corr(pm25_i, pm25_j)| — dựa trên Pearson
  3. Adaptive:          A = softmax(ReLU(E·Eᵀ))    — học trong training

Khuyến nghị:
  Bắt đầu với Correlation-based (ổn định, không cần GPS),
  sau đó thử Adaptive nếu muốn cải thiện thêm.
=============================================================================
"""

import numpy as np
import torch
from math import radians, sin, cos, sqrt, atan2

import pandas as pd


# ──────────────────────────────────────────────────────────────────────
# Tọa độ GPS trung tâm 12 tỉnh thành
# ──────────────────────────────────────────────────────────────────────

PROVINCE_COORDS = {
    'AnGiang':   (10.5215, 105.1259),
    'CanTho':    (10.0341, 105.7878),
    'DaNang':    (16.0471, 108.2068),
    'DongNai':   (10.9452, 107.1347),
    'HCM':       (10.8231, 106.6297),
    'HaiPhong':  (20.8449, 106.6881),
    'Hanoi':     (21.0285, 105.8542),
    'KhanhHoa':  (12.2388, 109.1967),
    'NgheAn':    (18.6796, 105.6813),
    'NinhBinh':  (20.2506, 105.9745),
    'ThanhHoa':  (19.8074, 105.7768),
    'VinhLong':  (10.2537, 105.9722),
}


# ──────────────────────────────────────────────────────────────────────
# Hàm phụ trợ
# ──────────────────────────────────────────────────────────────────────

def haversine(coord1: tuple, coord2: tuple) -> float:
    """
    Tính khoảng cách giữa 2 tọa độ GPS (km) bằng công thức Haversine.

    Args:
        coord1: (latitude, longitude) — tọa độ điểm 1
        coord2: (latitude, longitude) — tọa độ điểm 2

    Returns:
        Khoảng cách (km)

    Ví dụ:
        haversine((21.0285, 105.8542), (10.8231, 106.6297))
        → ≈ 1137 km (Hà Nội → HCM)
    """
    R = 6371  # Bán kính Trái Đất (km)
    lat1, lon1 = map(radians, coord1)
    lat2, lon2 = map(radians, coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def _symmetric_normalize(A: np.ndarray) -> np.ndarray:
    """
    Chuẩn hóa đối xứng: D^{-1/2} · A · D^{-1/2}

    Đây là chuẩn hóa standard cho GCN (Kipf & Welling, 2017).
    Giúp ổn định gradient và tránh node bậc cao bị dominate.

    Args:
        A: [N, N] — ma trận kề thô (chưa chuẩn hóa)

    Returns:
        A_norm: [N, N] — ma trận kề đã chuẩn hóa
    """
    # ── Tính degree matrix D ──
    d = A.sum(axis=1)  # [N] — bậc mỗi node

    # ── D^{-1/2}: tránh chia cho 0 ──
    d_inv_sqrt = np.where(d > 0, d ** (-0.5), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)

    # ── D^{-1/2} · A · D^{-1/2} ──
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    return A_norm


# ──────────────────────────────────────────────────────────────────────
# Phương pháp 1: Distance-based Adjacency
# ──────────────────────────────────────────────────────────────────────

def build_distance_adj(
    coords_dict: dict = None,
    sigma: float = 200.0,
    threshold: float = 500.0
) -> tuple:
    """
    Xây dựng adjacency matrix từ khoảng cách địa lý giữa các tỉnh.

    Công thức: A_ij = exp(-d_ij² / σ²)  nếu d_ij < threshold, else 0

    Args:
        coords_dict: dict {province: (lat, lon)}, mặc định dùng PROVINCE_COORDS
        sigma:       Bandwidth (km) — kiểm soát mức độ decay theo khoảng cách
                     σ nhỏ → chỉ tỉnh rất gần mới có cạnh mạnh
                     σ lớn → cạnh phân bố đều hơn
        threshold:   Cắt cạnh nếu khoảng cách > threshold km
                     500km → bỏ các cặp tỉnh quá xa (HN-HCM)

    Returns:
        (A_norm, node_list):
            A_norm:    torch.FloatTensor [N, N] — adjacency đã chuẩn hóa
            node_list: list[str] — thứ tự các node (tỉnh) tương ứng

    Ưu điểm: Trực quan, dựa trên kiến thức vật lý (ô nhiễm lan truyền qua không gian)
    Nhược điểm: Không tính đến hướng gió, địa hình
    """
    if coords_dict is None:
        coords_dict = PROVINCE_COORDS

    nodes = sorted(coords_dict.keys())
    N = len(nodes)
    A = np.zeros((N, N))

    # ── Tính pairwise distances và tạo adjacency ──
    for i in range(N):
        for j in range(N):
            if i != j:
                d = haversine(coords_dict[nodes[i]], coords_dict[nodes[j]])
                if d < threshold:
                    # ── Gaussian kernel: gần → giá trị lớn, xa → nhỏ ──
                    A[i, j] = np.exp(-(d ** 2) / (sigma ** 2))

    # ── Chuẩn hóa đối xứng ──
    A_norm = _symmetric_normalize(A)

    return torch.FloatTensor(A_norm), nodes


# ──────────────────────────────────────────────────────────────────────
# Phương pháp 2: Correlation-based Adjacency
# ──────────────────────────────────────────────────────────────────────

def build_correlation_adj(
    train_df: pd.DataFrame,
    target: str = 'pm25',
    threshold: float = 0.4,
    province_col: str = 'province',
    time_col: str = 'timestamp_local'
) -> tuple:
    """
    Xây dựng adjacency matrix từ Pearson correlation PM2.5 giữa các tỉnh.

    Ý tưởng: nếu PM2.5 của 2 tỉnh biến đổi tương đồng → chúng có quan hệ
    → tạo cạnh trong đồ thị

    Args:
        train_df:     DataFrame chứa dữ liệu TRAINING (chỉ dùng train để tránh leakage)
        target:       Biến mục tiêu để tính correlation (default: 'pm25')
        threshold:    Chỉ giữ cạnh nếu |corr| >= threshold
                      0.4 → loại bỏ các tương quan yếu
        province_col: Tên cột chứa tỉnh thành
        time_col:     Tên cột chứa timestamp

    Returns:
        (A_norm, node_list):
            A_norm:    torch.FloatTensor [N, N]
            node_list: list[str]

    Ưu điểm: Phản ánh mối quan hệ thực tế trong dữ liệu
    Nhược điểm: Phụ thuộc vào dữ liệu training, có thể thay đổi theo mùa
    """
    # ── Pivot: mỗi cột là một tỉnh, mỗi hàng là một timestamp ──
    pivot = train_df.pivot_table(
        index=time_col,
        columns=province_col,
        values=target
    ).sort_index()

    # ── Nội suy giá trị thiếu (tối đa 3 giờ liên tiếp) ──
    pivot = pivot.interpolate(method='time', limit=3)
    pivot = pivot.ffill().bfill()

    # ── Pearson correlation → giá trị tuyệt đối ──
    corr = pivot.corr(method='pearson').abs()

    # ── Threshold: chỉ giữ cạnh mạnh ──
    adj = corr.where(corr >= threshold, other=0.0).values

    # ── Bỏ self-loop (đường chéo = 0) ──
    np.fill_diagonal(adj, 0)

    # ── Chuẩn hóa ──
    A_norm = _symmetric_normalize(adj)

    node_list = sorted(pivot.columns.tolist())
    return torch.FloatTensor(A_norm), node_list


# ──────────────────────────────────────────────────────────────────────
# Phương pháp 3: Scaled Laplacian (cho Chebyshev GCN)
# ──────────────────────────────────────────────────────────────────────

def compute_scaled_laplacian(A: torch.Tensor) -> torch.Tensor:
    """
    Tính Scaled Laplacian: L̃ = 2L/λ_max - I

    Đây là dạng chuẩn hóa cần thiết cho Chebyshev Graph Convolution.
    Eigenvalues của L̃ nằm trong [-1, 1] → Chebyshev polynomial ổn định.

    Args:
        A: [N, N] — adjacency matrix (chưa chuẩn hóa hoặc đã chuẩn hóa)

    Returns:
        L_scaled: [N, N] — scaled Laplacian
    """
    A_np = A.numpy() if isinstance(A, torch.Tensor) else A
    N = A_np.shape[0]

    # ── L = D - A (unnormalized Laplacian) ──
    D = np.diag(A_np.sum(axis=1))
    L = D - A_np

    # ── λ_max: eigenvalue lớn nhất ──
    eigenvalues = np.linalg.eigvalsh(L)
    lambda_max = eigenvalues.max()

    # ── L̃ = 2L/λ_max - I ──
    if lambda_max > 0:
        L_scaled = 2.0 * L / lambda_max - np.eye(N)
    else:
        L_scaled = -np.eye(N)

    return torch.FloatTensor(L_scaled)
