"""
═══════════════════════════════════════════════════════════════
  RUN PARALLEL ST-TimeMixer: File chạy mô hình Parallel
═══════════════════════════════════════════════════════════════

Cách chạy:
    python run_parallel_stmixer.py

File này thực hiện các bước:
    1. Load config từ configs/parallel_stmixer_config.yaml
    2. Chuẩn bị dữ liệu (DataLoader) từ clean_data_all.csv
    3. Tạo đồ thị Adjacency Matrix
    4. Khởi tạo mô hình Parallel (Parallel Fusion + Gating)
    5. Training với Multi-task & Auxiliary Loss
    6. Đánh giá kiểm tra kết quả cuối cùng trên Test set
═══════════════════════════════════════════════════════════════
"""

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd

# -- Fix Unicode encoding on Windows console --
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# ---- Import the generic tools from original ST-TimeMixer
from models.st_timemixer.dataset import build_dataloaders, ALL_FEATURES, TARGET_COLS
from models.st_timemixer.graph_utils import build_correlation_adj, build_distance_adj

# ---- Import Parallel Model and Trainer components
from models.parallel_st_timemixer.stmixer import ParallelSTMixer
from models.parallel_st_timemixer.trainer import ParallelTrainer


def main():
    # ══════════════════════════════════════════════════════
    # BƯỚC 1: LOAD CONFIG CHO PARALLEL
    # ══════════════════════════════════════════════════════
    print("=" * 60)
    print("  BƯỚC 1: Load config Parallel")
    print("=" * 60)

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        print(f"  ✅ Loaded config từ: {config_path}")
    else:
        print(f"  ❌ Không tìm thấy {config_path}. Dừng chương trình.")
        return

    model_cfg = config["model"]
    train_cfg = config["training"]
    split_cfg = config["split"]
    graph_cfg = config.get("graph", {})
    node_list = config.get("provinces", [
        "AnGiang", "CanTho", "DaNang", "DongNai", "HCM",
        "HaiPhong", "Hanoi", "KhanhHoa", "NgheAn",
        "NinhBinh", "ThanhHoa", "VinhLong"
    ])
    csv_path = config["paths"]["data"]
    save_dir = config["paths"]["save_dir"]
    best_model_path = config["paths"]["best_model"]

    os.makedirs(save_dir, exist_ok=True)

    print(f"  Nodes:     {len(node_list)} tỉnh")
    print(f"  Lookback:  {model_cfg['seq_len']}h → Horizon: {model_cfg['pred_len']}h")
    print(f"  Data path: {csv_path}")

    # ══════════════════════════════════════════════════════
    # BƯỚC 2: CHUẨN BỊ DỮ LIỆU
    # ══════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("  BƯỚC 2: Chuẩn bị dữ liệu (CSV → DataLoader)")
    print("=" * 60)

    train_loader, val_loader, test_loader, scaler = build_dataloaders(
        csv_path=csv_path,
        node_list=node_list,
        feature_cols=ALL_FEATURES,
        target_cols=TARGET_COLS,
        seq_len=model_cfg["seq_len"],
        pred_len=model_cfg["pred_len"],
        batch_size=train_cfg["batch_size"],
        train_end=split_cfg["train_end"],
        val_end=split_cfg["val_end"],
    )

    x_sample, y_sample = next(iter(train_loader))
    print(f"\n  Sample batch:")
    print(f"    x shape: {list(x_sample.shape)}  (B, N, T, C)")
    print(f"    y shape: {list(y_sample.shape)}  (B, N, H, n_targets)")

    # ══════════════════════════════════════════════════════
    # BƯỚC 3: XÂY DỰNG ĐỒ THỊ
    # ══════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("  BƯỚC 3: Xây dựng đồ thị (Adjacency Matrix)")
    print("=" * 60)

    adj_method = graph_cfg.get("method", "correlation")
    
    if adj_method == "correlation":
        print("  Phương pháp: Pearson Correlation (PM2.5)")
        df = pd.read_csv(csv_path, parse_dates=["timestamp_local"])
        train_df = df[df["timestamp_local"] <= split_cfg["train_end"]]
        adj_matrix, _ = build_correlation_adj(
            train_df,
            target="pm25",
            threshold=graph_cfg.get("corr_threshold", 0.4),
        )
    elif adj_method == "distance":
        print("  Phương pháp: Khoảng cách địa lý (Haversine)")
        adj_matrix, _ = build_distance_adj(
            sigma=graph_cfg.get("distance_sigma", 200),
            threshold=graph_cfg.get("distance_threshold", 500),
        )
    else:
        print("  Phương pháp: Identity")
        adj_matrix = torch.eye(len(node_list))

    print(f"  Adjacency matrix shape: {list(adj_matrix.shape)}")

    # ══════════════════════════════════════════════════════
    # BƯỚC 4: KHỞI TẠO MODEL VÀ TRAINER CHO PARALLEL
    # ══════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("  BƯỚC 4: Tạo Parallel Model + Training")
    print("=" * 60)

    actual_features = x_sample.shape[-1]
    if model_cfg['num_features'] != actual_features:
        print(f"  ⚠️ Cập nhật num_features từ {model_cfg['num_features']} thành {actual_features}")
        model_cfg['num_features'] = actual_features
        
    # Tạo model Parallel STMixer
    model = ParallelSTMixer(model_cfg)

    # Khởi tạo Parallel Trainer có aux loss và entropy regularizer 
    trainer = ParallelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        adj_matrix=adj_matrix,
        config=train_cfg,
        save_path=best_model_path,
    )

    trainer.fit()

    # ══════════════════════════════════════════════════════
    # BƯỚC 5: ĐÁNH GIÁ TRÊN TEST SET
    # ══════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("  BƯỚC 5: Đánh giá trên test set")
    print("=" * 60)

    target_indices = train_loader.dataset.target_idx
    print(f"  Target indices in feature_cols: {target_indices}")

    results = trainer.evaluate(
        test_loader=test_loader,
        scaler=scaler,
        node_list=node_list,
        target_indices=target_indices,
    )

    trainer.print_results(results)

    results_path = os.path.join(save_dir, "test_results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    print(f"  💾 Kết quả đã lưu: {results_path}")

    print(f"\n{'=' * 60}")
    print("  HOÀN TẤT PARALLEL!")
    print("=" * 60)


if __name__ == "__main__":
    main()
