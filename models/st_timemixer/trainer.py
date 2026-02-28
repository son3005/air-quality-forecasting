"""
=============================================================================
Module: Trainer & Evaluator — Huấn luyện và Đánh giá ST-TimeMixer
=============================================================================
Vai trò:
  → Training loop với Early Stopping, Gradient Clipping, LR Scheduling
  → Multi-task Loss: λ₁·L_PM2.5 + λ₂·L_AQI (Huber Loss)
  → Evaluation: MAE, RMSE, R², sMAPE, AQI category accuracy/F1
  → Phân tích per-horizon và per-node

Cấu trúc:
  1. st_timemixer_loss()  — hàm loss multi-task
  2. Trainer class        — training loop chính
  3. evaluate()           — đánh giá toàn diện
  4. aqi_to_category()    — chuyển AQI → mức ô nhiễm
=============================================================================
"""

import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


# ══════════════════════════════════════════════════════════════════════
# LOSS FUNCTION (v2: AQI category-weighted)
# ══════════════════════════════════════════════════════════════════════

def st_timemixer_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_pm25: float = 0.6,
    lambda_aqi: float = 0.4,
    huber_delta: float = 1.0,
    aqi_high_weight: float = 1.0,
    aqi_critical_weight: float = 1.0,
    target_aqi_real: torch.Tensor = None
) -> torch.Tensor:
    """
    Multi-task Huber Loss voi AQI category-weighted penalty.

    v2 cai tien:
      - Khi AQI thuc > 100 (nhom nhay cam): loss x aqi_high_weight
      - Khi AQI thuc > 200 (xau/nguy hiem): loss x aqi_critical_weight
      - Phat nang hon khi sai o muc o nhiem cao (quan trong cho suc khoe)

    Args:
        pred:                [B, N, H, n_targets]
        target:              [B, N, H, n_targets]
        lambda_pm25:         Trong so PM2.5 loss
        lambda_aqi:          Trong so AQI loss
        huber_delta:         Huber delta
        aqi_high_weight:     He so nhan khi AQI > 100 (scaled)
        aqi_critical_weight: He so nhan khi AQI > 200 (scaled)
        target_aqi_real:     AQI tren gia tri thuc (de xac dinh category)
                             Neu None, dung target[...,1] truc tiep (scaled)

    Returns:
        loss: scalar tensor
    """
    huber = nn.HuberLoss(reduction='none', delta=huber_delta)

    # -- Loss PM2.5: Huber binh thuong --
    loss_pm25 = huber(pred[..., 0], target[..., 0]).mean()

    # -- Loss AQI: Huber voi sample-level weighting --
    loss_aqi_raw = huber(pred[..., 1], target[..., 1])  # [B, N, H]

    # -- Tao weight map dua tren AQI category --
    if aqi_high_weight > 1.0 or aqi_critical_weight > 1.0:
        aqi_target = target[..., 1]  # [B, N, H] — scaled values
        weights = torch.ones_like(aqi_target)

        # Su dung quantile de xac dinh "high" va "critical" tren scaled data
        # Vi data da scaled (mean=0, std=1), can dung threshold tuong doi
        # AQI ~100 tuong ung voi ~top 30% data, AQI ~200 tuong ung ~top 5%
        q70 = torch.quantile(aqi_target.float(), 0.70)
        q95 = torch.quantile(aqi_target.float(), 0.95)

        weights = torch.where(aqi_target > q70, aqi_high_weight * weights, weights)
        weights = torch.where(aqi_target > q95, aqi_critical_weight * weights, weights)

        loss_aqi = (loss_aqi_raw * weights).mean()
    else:
        loss_aqi = loss_aqi_raw.mean()

    # -- Weighted sum --
    total_loss = lambda_pm25 * loss_pm25 + lambda_aqi * loss_aqi

    return total_loss


# ══════════════════════════════════════════════════════════════════════
# AQI CATEGORY MAPPING
# ══════════════════════════════════════════════════════════════════════

def aqi_to_category(aqi_value: float) -> int:
    """
    Chuyển đổi giá trị AQI → category (mức ô nhiễm).

    | AQI Range | Category | Mức         |
    |-----------|----------|-------------|
    | 0–50      | 0        | Tốt         |
    | 51–100    | 1        | Trung bình  |
    | 101–150   | 2        | Nhóm nhạy   |
    | 151–200   | 3        | Không tốt   |
    | 201–300   | 4        | Xấu         |
    | >300      | 5        | Nguy hiểm   |
    """
    if aqi_value <= 50:
        return 0
    elif aqi_value <= 100:
        return 1
    elif aqi_value <= 150:
        return 2
    elif aqi_value <= 200:
        return 3
    elif aqi_value <= 300:
        return 4
    else:
        return 5


# ══════════════════════════════════════════════════════════════════════
# TRAINER CLASS
# ══════════════════════════════════════════════════════════════════════

class Trainer:
    """
    Training loop cho ST-TimeMixer.

    Tính năng:
      - Early stopping (patience)
      - Gradient clipping (max_norm)
      - CosineAnnealingLR scheduler
      - Logging mỗi epoch
      - Lưu best model checkpoint

    Sử dụng:
        trainer = Trainer(model, train_loader, val_loader, A, config)
        trainer.fit()
        results = trainer.evaluate(test_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        adj_matrix: torch.Tensor,
        config: dict = None,
        device: str = None,
        save_path: str = 'best_st_timemixer.pt'
    ):
        """
        Args:
            model:        STTimeMixer instance
            train_loader: DataLoader cho training
            val_loader:   DataLoader cho validation
            adj_matrix:   torch.Tensor [N, N] — adjacency matrix
            config:       dict — training hyperparameters
            device:       'cuda' hoặc 'cpu'
            save_path:    Đường dẫn lưu best model
        """
        # ── Device setup ──
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # ── Model → device ──
        self.model = model.to(self.device)
        self.A = adj_matrix.to(self.device)  # Adjacency matrix

        # ── Data ──
        self.train_loader = train_loader
        self.val_loader = val_loader

        # -- Config voi defaults (v2: them warmup, aqi weights) --
        if config is None:
            config = {}
        self.epochs = config.get('epochs', 150)
        self.lr = config.get('lr', 3e-4)
        self.weight_decay = config.get('weight_decay', 1e-4)
        self.grad_clip = config.get('grad_clip', 5.0)
        self.patience = config.get('patience', 20)
        self.lambda_pm25 = config.get('lambda_pm25', 0.6)
        self.lambda_aqi = config.get('lambda_aqi', 0.4)
        self.huber_delta = config.get('huber_delta', 1.0)
        self.save_path = save_path

        # [v2] AQI category weights
        self.aqi_high_weight = config.get('aqi_high_weight', 2.0)
        self.aqi_critical_weight = config.get('aqi_critical_weight', 3.0)

        # [v2] Warmup config
        self.warmup_epochs = config.get('warmup_epochs', 5)

        # -- Optimizer: AdamW (v2: AdamW thay Adam, tot hon cho regularization) --
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # -- [v2] Scheduler: Warmup + CosineAnnealing --
        # 5 epoch dau: LR tang dan tu 0 -> lr (on dinh gradient ban dau)
        # Sau do: CosineAnnealing giam dan
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,           # Bat dau tu 0.1 * lr
            end_factor=1.0,             # Tang len 1.0 * lr
            total_iters=self.warmup_epochs
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('T_max', 100),
            eta_min=config.get('eta_min', 1e-6)
        )
        # Ket hop: warmup 5 epoch -> cosine phan con lai
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_epochs]
        )

        # -- Early stopping tracking --
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []

        print(f"[Trainer v2] Device: {self.device}")
        print(f"[Trainer v2] Epochs: {self.epochs}, LR: {self.lr}, "
              f"Patience: {self.patience}")
        print(f"[Trainer v2] Warmup: {self.warmup_epochs} epochs, "
              f"AQI weights: high={self.aqi_high_weight}x, critical={self.aqi_critical_weight}x")

    def _train_one_epoch(self) -> float:
        """Chạy 1 epoch training."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for x, y in self.train_loader:
            # x: [B, N, T, C],  y: [B, N, H, n_targets]
            x = x.to(self.device)
            y = y.to(self.device)

            # ── Forward pass ──
            self.optimizer.zero_grad()
            pred = self.model(x, self.A)  # [B, N, H, n_targets]

            # -- Loss (v2: them AQI weighted penalty) --
            loss = st_timemixer_loss(
                pred, y,
                lambda_pm25=self.lambda_pm25,
                lambda_aqi=self.lambda_aqi,
                huber_delta=self.huber_delta,
                aqi_high_weight=self.aqi_high_weight,
                aqi_critical_weight=self.aqi_critical_weight
            )

            # ── Backward + Gradient Clipping ──
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.grad_clip  # Ngăn gradient explosion
            )
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self) -> float:
        """Chạy validation (không tính gradient)."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for x, y in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            pred = self.model(x, self.A)
            loss = st_timemixer_loss(
                pred, y,
                lambda_pm25=self.lambda_pm25,
                lambda_aqi=self.lambda_aqi,
                huber_delta=self.huber_delta
            )

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def fit(self):
        """
        Training loop chính với early stopping.

        Flow mỗi epoch:
          1. Train → train_loss
          2. Validate → val_loss
          3. Scheduler step (giảm LR)
          4. Early stopping check:
             - val_loss giảm → lưu model, reset counter
             - val_loss không giảm → tăng counter
             - counter >= patience → DỪNG
        """
        print(f"\n{'='*60}")
        print(f"  BẮT ĐẦU TRAINING — {self.epochs} epochs")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # ── Train ──
            train_loss = self._train_one_epoch()
            self.train_losses.append(train_loss)

            # ── Validate ──
            val_loss = self._validate()
            self.val_losses.append(val_loss)

            # ── Scheduler step ──
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # ── Logging ──
            elapsed = time.time() - epoch_start
            print(f"Epoch {epoch:3d}/{self.epochs} | "
                  f"Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {elapsed:.1f}s", end="")

            # ── Early Stopping Check ──
            if val_loss < self.best_val_loss:
                # ✅ Improved → lưu model
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.save_path)
                print(f" ★ Best (↓{improvement:.4f})")
            else:
                # ❌ Không cải thiện → tăng counter
                self.patience_counter += 1
                print(f" (patience {self.patience_counter}/{self.patience})")

                if self.patience_counter >= self.patience:
                    print(f"\n⛔ Early stopping tại epoch {epoch}")
                    break

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"  TRAINING HOÀN TẤT")
        print(f"  Tổng thời gian: {total_time:.1f}s ({total_time/60:.1f} phút)")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"  Model saved: {self.save_path}")
        print(f"{'='*60}\n")

        # ── Load best model ──
        self.model.load_state_dict(torch.load(self.save_path, weights_only=True))

    @torch.no_grad()
    def evaluate(
        self,
        test_loader,
        scaler=None,
        node_list: list = None,
        target_indices: list = None
    ) -> dict:
        """
        Đánh giá toàn diện trên test set.

        Metrics trả về:
          1. Overall: MAE, RMSE, R², sMAPE (cho PM2.5 và AQI) — trên GIẤY TRỊ THỰC
          2. Classification: AQI category Accuracy, F1
          3. Per-horizon: MAE/RMSE tại mỗi bước H
          4. Per-node: MAE cho mỗi tỉnh

        Args:
            test_loader:    DataLoader cho test set
            scaler:         StandardScaler — để inverse transform
            node_list:      list[str] — tên các tỉnh (optional, cho per-node)
            target_indices: list[int] — index của target cols trong feature_cols
                            vd: [0, 1] nếu pm25, aqi là 2 feature đầu tiên

        Returns:
            dict chứa tất cả metrics
        """
        self.model.eval()
        all_preds, all_targets = [], []

        for x, y in test_loader:
            x = x.to(self.device)
            pred = self.model(x, self.A)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())

        # Concat: [total_samples, N, H, n_targets]
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # ══════════════════════════════════════════════════
        # INVERSE TRANSFORM: Scaled → Giá trị thực (µg/m³, AQI)
        # ══════════════════════════════════════════════════
        # Scaler fit trên 22 features, nhưng preds/targets chỉ có 2 targets.
        # Cần lấy mean_ và scale_ tương ứng với index pm25 (0) và aqi (1).

        if scaler is not None and target_indices is not None:
            for ti, feat_idx in enumerate(target_indices):
                # RobustScaler: x_scaled = (x - center) / scale
                # → x_real = x_scaled * scale + center
                # Tương thích cả StandardScaler (mean_) và RobustScaler (center_)
                if hasattr(scaler, 'center_'):
                    center = scaler.center_[feat_idx]
                    scale_val = scaler.scale_[feat_idx]
                else:
                    center = scaler.mean_[feat_idx]
                    scale_val = scaler.scale_[feat_idx]
                
                # Bước 1: Inverse Scaler
                preds[..., ti] = preds[..., ti] * scale_val + center
                targets[..., ti] = targets[..., ti] * scale_val + center
                
                # Bước 2: Inverse Log-transform (expm1 = e^x - 1)
                preds[..., ti] = np.expm1(np.clip(preds[..., ti], -20, 20))
                targets[..., ti] = np.expm1(np.clip(targets[..., ti], -20, 20))
                
            print(f"  ✅ Inverse transform applied (target indices: {target_indices})")
            print(f"     PM2.5 range: pred [{preds[...,0].min():.1f}, {preds[...,0].max():.1f}], "
                  f"true [{targets[...,0].min():.1f}, {targets[...,0].max():.1f}]")
            print(f"     AQI   range: pred [{preds[...,1].min():.1f}, {preds[...,1].max():.1f}], "
                  f"true [{targets[...,1].min():.1f}, {targets[...,1].max():.1f}]")
        elif scaler is not None:
            print("  ⚠️ Scaler provided nhưng thiếu target_indices → KHÔNG inverse transform")
            print("     Metrics sẽ trên scaled data (không phải giá trị thực)")

        metrics = {}

        # ══════════════════════════════════════════════════
        # 1. OVERALL REGRESSION METRICS
        # ══════════════════════════════════════════════════

        for ti, target_name in enumerate(['pm25', 'aqi']):
            p = preds[..., ti]      # [total, N, H]
            t = targets[..., ti]

            mae = np.abs(p - t).mean()
            rmse = np.sqrt(((p - t) ** 2).mean())

            # R² score
            ss_res = ((t - p) ** 2).sum()
            ss_tot = ((t - t.mean()) ** 2).sum()
            r2 = 1 - ss_res / (ss_tot + 1e-8)

            # sMAPE (Symmetric MAPE — robust khi y ≈ 0)
            smape = (200 * np.abs(p - t) / (np.abs(p) + np.abs(t) + 1e-8)).mean()

            metrics[f'{target_name}_MAE'] = round(float(mae), 4)
            metrics[f'{target_name}_RMSE'] = round(float(rmse), 4)
            metrics[f'{target_name}_R2'] = round(float(r2), 4)
            metrics[f'{target_name}_sMAPE'] = round(float(smape), 4)

        # ══════════════════════════════════════════════════
        # 2. AQI CLASSIFICATION METRICS
        # ══════════════════════════════════════════════════

        aqi_pred_flat = preds[..., 1].flatten()
        aqi_true_flat = targets[..., 1].flatten()

        # Chuyển sang category
        pred_cats = np.array([aqi_to_category(v) for v in aqi_pred_flat])
        true_cats = np.array([aqi_to_category(v) for v in aqi_true_flat])

        metrics['aqi_category_accuracy'] = round(
            float(accuracy_score(true_cats, pred_cats)), 4
        )
        metrics['aqi_category_f1_macro'] = round(
            float(f1_score(true_cats, pred_cats, average='macro', zero_division=0)), 4
        )
        metrics['aqi_category_f1_weighted'] = round(
            float(f1_score(true_cats, pred_cats, average='weighted', zero_division=0)), 4
        )

        # In phân bố categories để debug
        unique_true, counts_true = np.unique(true_cats, return_counts=True)
        unique_pred, counts_pred = np.unique(pred_cats, return_counts=True)
        metrics['aqi_true_category_dist'] = dict(zip(unique_true.tolist(), counts_true.tolist()))
        metrics['aqi_pred_category_dist'] = dict(zip(unique_pred.tolist(), counts_pred.tolist()))

        # ══════════════════════════════════════════════════
        # 3. PER-HORIZON METRICS
        # ══════════════════════════════════════════════════
        # Phân tích lỗi tại mỗi bước dự báo h=1..H

        H = preds.shape[2]
        per_horizon = {}
        for h in range(H):
            p_h = preds[:, :, h, 0]      # PM2.5 tại bước h
            t_h = targets[:, :, h, 0]
            mae_h = np.abs(p_h - t_h).mean()
            rmse_h = np.sqrt(((p_h - t_h) ** 2).mean())
            per_horizon[f'h{h+1}'] = {
                'MAE': round(float(mae_h), 4),
                'RMSE': round(float(rmse_h), 4)
            }
        metrics['per_horizon_pm25'] = per_horizon

        # ══════════════════════════════════════════════════
        # 4. PER-NODE METRICS
        # ══════════════════════════════════════════════════
        # Phân tích tỉnh nào khó dự báo nhất

        N = preds.shape[1]
        per_node = {}
        for ni in range(N):
            p_n = preds[:, ni, :, 0]      # PM2.5 cho node ni
            t_n = targets[:, ni, :, 0]
            mae_n = np.abs(p_n - t_n).mean()
            node_name = node_list[ni] if node_list and ni < len(node_list) else f'node_{ni}'
            per_node[node_name] = {'MAE': round(float(mae_n), 4)}
        metrics['per_node_pm25'] = per_node

        return metrics

    def print_results(self, metrics: dict):
        """In kết quả đánh giá một cách trực quan."""
        print(f"\n{'='*60}")
        print(f"  KẾT QUẢ ĐÁNH GIÁ ST-TimeMixer")
        print(f"{'='*60}")

        # ── Overall ──
        print(f"\n📊 Overall Metrics:")
        print(f"  PM2.5 — MAE: {metrics.get('pm25_MAE', '—'):>8} | "
              f"RMSE: {metrics.get('pm25_RMSE', '—'):>8} | "
              f"R²: {metrics.get('pm25_R2', '—'):>8}")
        print(f"  AQI   — MAE: {metrics.get('aqi_MAE', '—'):>8} | "
              f"RMSE: {metrics.get('aqi_RMSE', '—'):>8} | "
              f"R²: {metrics.get('aqi_R2', '—'):>8}")

        # ── Classification ──
        print(f"\n🏷️  AQI Category Classification:")
        print(f"  Accuracy:    {metrics.get('aqi_category_accuracy', '—')}")
        print(f"  Macro F1:    {metrics.get('aqi_category_f1_macro', '—')}")
        print(f"  Weighted F1: {metrics.get('aqi_category_f1_weighted', '—')}")

        # In phân bố categories
        cat_names = {0: 'Tốt(0-50)', 1: 'TB(51-100)', 2: 'Nhạy(101-150)',
                     3: 'Xấu(151-200)', 4: 'Rất xấu(201-300)', 5: 'Nguy hiểm(>300)'}
        if 'aqi_true_category_dist' in metrics:
            print(f"  True dist:  ", end="")
            for cat, cnt in sorted(metrics['aqi_true_category_dist'].items()):
                print(f" {cat_names.get(cat, cat)}={cnt}", end="")
            print()
            print(f"  Pred dist:  ", end="")
            for cat, cnt in sorted(metrics['aqi_pred_category_dist'].items()):
                print(f" {cat_names.get(cat, cat)}={cnt}", end="")
            print()

        # ── Per horizon ──
        if 'per_horizon_pm25' in metrics:
            print(f"\n📈 Per-Horizon PM2.5 MAE:")
            for h, vals in metrics['per_horizon_pm25'].items():
                bar = '█' * int(vals['MAE'] * 5)
                print(f"  {h}: MAE={vals['MAE']:.4f} RMSE={vals['RMSE']:.4f} {bar}")

        # ── Per node ──
        if 'per_node_pm25' in metrics:
            print(f"\n🗺️  Per-Node PM2.5 MAE:")
            sorted_nodes = sorted(
                metrics['per_node_pm25'].items(),
                key=lambda x: x[1]['MAE']
            )
            for name, vals in sorted_nodes:
                bar = '█' * int(vals['MAE'] * 5)
                print(f"  {name:>12}: MAE={vals['MAE']:.4f} {bar}")

        print(f"\n{'='*60}\n")
