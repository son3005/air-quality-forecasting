"""
═══════════════════════════════════════════════════════════════
RUN SANDWICH ST-TimeMixer
Luồng chạy (Pipeline) của Kiến trúc Mạng T-S-T
═══════════════════════════════════════════════════════════════
"""

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd

# Fix console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# Import Module từ gốc Project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Tận dụng Dataloader & Dataset gốc (Đến nay Data pipeline vẫn đang là Benchmark hoàn hảo nhất)
from models.sandwich_st_timemixer.dataset import build_dataloaders, ALL_FEATURES, TARGET_COLS
from models.sandwich_st_timemixer.graph_utils import build_correlation_adj, build_distance_adj

# Import Sandwich Core Function
from models.sandwich_st_timemixer.sandwich_model import SandwichSTTimeMixer
from models.sandwich_st_timemixer.trainer import SandwichTrainer

def main():
    print("=" * 60)
    print(" KHỞI ĐỘNG KIẾN TRÚC SILO T-S-T: TÍCH CHẬP CHUỖI ")
    print("=" * 60)

    # 1. Tải lên bảng mô tả Config 
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        print(f" [V] Load Cấu trúc {config_path}")
    else:
        print(f" [X] Lỗi: Không thể tìm thấy file cấu hình {config_path}!")
        return

    model_cfg = config["model"]
    train_cfg = config["training"]
    split_cfg = config["split"]
    graph_cfg = config.get("graph", {})
    
    node_list = config.get("provinces", [])
    csv_path = config["paths"]["data"]
    save_dir = config["paths"]["save_dir"]
    best_model_path = config["paths"]["best_model"]
    
    os.makedirs(save_dir, exist_ok=True)
    print(f" [i] Lookback: {model_cfg['seq_len']}h | Horizon: {model_cfg['pred_len']}h")

    # 2. Xúc tác Chuẩn bị Dữ Liệu
    print("\n -> Xử lý Batch & Dataloader...")
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
    act_feats = x_sample.shape[-1]
    print(f" -> Tensor Shape | X: {list(x_sample.shape)} | Y: {list(y_sample.shape)}")
    
    if model_cfg['num_features'] != act_feats:
         print(f" [!] Bắt Config Lệch Data (Dự kiến {model_cfg['num_features']} | Thực tế {act_feats}). Auto Fixing...")
         model_cfg['num_features'] = act_feats

    # 3. Tạo Ma trận Kề Đồ thị
    print("\n -> Xây dựng Đồ Thị Adjacency...")
    adj_method = graph_cfg.get("method", "correlation")
    
    if adj_method == "correlation":
        df = pd.read_csv(csv_path, parse_dates=["timestamp_local"])
        train_df = df[df["timestamp_local"] <= split_cfg["train_end"]]
        adj_matrix, _ = build_correlation_adj(
            train_df,
            target="pm25",
            threshold=graph_cfg.get("corr_threshold", 0.3)
        )
    else:
        adj_matrix = torch.eye(len(node_list))
    
    # 4. Giải nén Mô Hình
    model = SandwichSTTimeMixer(model_cfg)
    print(f"\n -> Sinh Module Xong: TimeMixer({len(model_cfg['scales'])} Scales) <=> ChebConv(Khoảng ={model_cfg['K_cheby']}) <=> TempGLU")

    # 5. Ráp nối Trainer và Điểm hỏa Huấn luyện
    trainer = SandwichTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        adj_matrix=adj_matrix,
        config=train_cfg,
        save_path=best_model_path,
    )
    
    trainer.fit()

    # 6. Final Evaluation
    print("\n -> Đo lường và kết xuất Dữ liệu cuối...")
    target_indices = train_loader.dataset.target_idx
    results = trainer.evaluate(test_loader, scaler, node_list, target_indices)
    trainer.print_results(results)

    results_path = os.path.join(save_dir, "test_results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    print(f"\n [V] Xuất tệp Log -> {results_path}")

if __name__ == "__main__":
    main()
