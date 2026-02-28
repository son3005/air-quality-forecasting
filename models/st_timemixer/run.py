"""
═══════════════════════════════════════════════════════════════
  RUN ST-TimeMixer: File duy nhất bạn cần chạy
═══════════════════════════════════════════════════════════════

Cách chạy:
    python run_st_timemixer.py

File này thực hiện 5 bước tuần tự:
    Bước 1: Load config
    Bước 2: Chuẩn bị dữ liệu (CSV → DataLoader)
    Bước 3: Xây dựng đồ thị (Adjacency Matrix)
    Bước 4: Tạo model + Training
    Bước 5: Đánh giá trên test set

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

# ── Import modules ──
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from models.st_timemixer.st_timemixer import STTimeMixer
from models.st_timemixer.dataset import build_dataloaders, ALL_FEATURES, TARGET_COLS
from models.st_timemixer.graph_utils import build_correlation_adj, build_distance_adj
from models.st_timemixer.trainer import Trainer


def main():
    # ══════════════════════════════════════════════════════
    # BƯỚC 1: LOAD CONFIG
    # ══════════════════════════════════════════════════════
    print("=" * 60)
    print("  BƯỚC 1: Load config")
    print("=" * 60)

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        print(f"  ✅ Loaded config từ: {config_path}")
    else:
        # Fallback: config mặc định nếu không có file YAML
        print(f"  ⚠️ Không tìm thấy {config_path}, dùng config mặc định")
        config = {
            "model": {
                "num_nodes": 12, "seq_len": 24, "pred_len": 12,
                "num_features": 22, "num_targets": 2,
                "d_model": 64, "d_ff": 256, "scales": [1, 4, 24],
                "K_cheby": 3, "num_gcn_layers": 2, "dropout": 0.2,
                "decomp_kernel": 25,
            },
            "graph": {"method": "correlation", "corr_threshold": 0.4},
            "training": {
                "batch_size": 32, "epochs": 100, "lr": 5e-4,
                "weight_decay": 1e-4, "grad_clip": 5.0, "patience": 15,
                "T_max": 50, "eta_min": 1e-6,
                "lambda_pm25": 0.6, "lambda_aqi": 0.4, "huber_delta": 1.0,
            },
            "split": {"train_end": "2023-09-30", "val_end": "2023-11-30"},
            "provinces": [
                "AnGiang", "CanTho", "DaNang", "DongNai", "HCM",
                "HaiPhong", "Hanoi", "KhanhHoa", "NgheAn",
                "NinhBinh", "ThanhHoa", "VinhLong"
            ],
            "paths": {
                "data": "data/clean_data_all.csv",
                "save_dir": "results/st_timemixer/",
                "best_model": "results/st_timemixer/best_model.pt",
            },
        }

    # Trích xuất config
    model_cfg = config["model"]
    train_cfg = config["training"]
    split_cfg = config["split"]
    graph_cfg = config.get("graph", {})
    node_list = config.get("provinces", [])
    csv_path = config["paths"]["data"]
    save_dir = config["paths"]["save_dir"]
    best_model_path = config["paths"]["best_model"]

    # Tạo thư mục kết quả
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

    # Kiểm tra 1 batch
    x_sample, y_sample = next(iter(train_loader))
    print(f"\n  Sample batch:")
    print(f"    x shape: {list(x_sample.shape)}  (B, N, T, C)")
    print(f"    y shape: {list(y_sample.shape)}  (B, N, H, n_targets)")

    # ══════════════════════════════════════════════════════
    # BƯỚC 3: XÂY DỰNG ĐỒ THỊ (Adjacency Matrix)
    # ══════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("  BƯỚC 3: Xây dựng đồ thị (Adjacency Matrix)")
    print("=" * 60)

    adj_method = graph_cfg.get("method", "correlation")

    if adj_method == "correlation":
        print("  Phương pháp: Pearson Correlation (PM2.5)")
        # Đọc lại CSV để tính correlation trên training data
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
        print("  Phương pháp: Identity (adaptive sẽ học trong training)")
        adj_matrix = torch.eye(len(node_list))

    print(f"  Adjacency matrix shape: {list(adj_matrix.shape)}")
    print(f"  Số cạnh (non-zero): {(adj_matrix > 0).sum().item()}")

    # ══════════════════════════════════════════════════════
    # BƯỚC 4: TẠO MODEL + TRAINING
    # ══════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("  BƯỚC 4: Tạo model + Training")
    print("=" * 60)

    # Tạo model — Auto-detect num_features từ data thực tế
    actual_features = x_sample.shape[-1]  # C dimension
    if model_cfg['num_features'] != actual_features:
        print(f"  ⚠️ Config num_features={model_cfg['num_features']} != actual={actual_features}")
        print(f"     → Auto-correcting to {actual_features}")
        model_cfg['num_features'] = actual_features
    model = STTimeMixer(model_cfg)

    # Tạo trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        adj_matrix=adj_matrix,
        config=train_cfg,
        save_path=best_model_path,
    )

    # BẮT ĐẦU TRAINING
    trainer.fit()

    # ══════════════════════════════════════════════════════
    # BƯỚC 5: ĐÁNH GIÁ TRÊN TEST SET
    # ══════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("  BƯỚC 5: Đánh giá trên test set")
    print("=" * 60)

    # Lấy target_indices từ dataset thực tế (đã filter cột không tồn tại)
    target_indices = train_loader.dataset.target_idx
    print(f"  Target indices in feature_cols: {target_indices}")

    results = trainer.evaluate(
        test_loader=test_loader,
        scaler=scaler,
        node_list=node_list,
        target_indices=target_indices,
    )

    # In kết quả
    trainer.print_results(results)

    # Lưu kết quả ra file
    results_path = os.path.join(save_dir, "test_results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    print(f"  💾 Kết quả đã lưu: {results_path}")

    print(f"\n{'=' * 60}")
    print("  HOÀN TẤT!")
    print("=" * 60)


if __name__ == "__main__":
    main()
