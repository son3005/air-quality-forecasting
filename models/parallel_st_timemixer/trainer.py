"""
═══════════════════════════════════════════════════════════════
Trainer v2 cho Parallel STMixer — Loss + Training Loop
═══════════════════════════════════════════════════════════════
Cải tiến so với v1:
  1. Soft-Focal weighting: phạt mạnh hơn ở vùng error lớn nhưng
     có giới hạn trên (clamp) để tránh gradient explosion → chống overfitting
  2. CosineAnnealingWarmRestarts: LR restart mỗi 25 epoch → model
     có cơ hội thoát local minima, train lâu hơn
  3. Auxiliary Loss dùng đúng aux_decoder riêng (gradient đúng nhánh)
  4. sMAPE metric và per-horizon/per-node reports đầy đủ
═══════════════════════════════════════════════════════════════
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score


def aqi_to_category(aqi_value: float) -> int:
    if aqi_value <= 50: return 0
    elif aqi_value <= 100: return 1
    elif aqi_value <= 150: return 2
    elif aqi_value <= 200: return 3
    elif aqi_value <= 300: return 4
    else: return 5


def parallel_stmixer_loss(
    pred_dict: dict,
    target: torch.Tensor,
    lambda_pm25: float = 0.6,
    lambda_aqi: float = 0.4,
    huber_delta: float = 1.0,
    aqi_high_weight: float = 1.0,
    aqi_critical_weight: float = 1.0,
    loss_alpha: float = 0.1,
    loss_beta: float = 0.01,
    aux_tm_weight: float = 0.3,
    aux_st_weight: float = 0.3
) -> torch.Tensor:
    """
    Parallel Composite Loss:
      main = Huber(fused) + alpha*MAE(fused) - beta*Entropy(gate)
           + aux_tm * Huber(tm_branch) + aux_st * Huber(st_branch)
    
    Soft-Focal weighting: errors proportionally weighted (clamped).
    """
    pred = pred_dict['pred']
    gate = pred_dict['gate']

    huber = nn.HuberLoss(reduction='none', delta=huber_delta)
    mae_fn = nn.L1Loss(reduction='mean')

    # ── PM2.5 Loss with Soft-Focal ──
    error_pm25 = torch.abs(pred[..., 0] - target[..., 0])
    soft_weight_pm25 = 1.0 + torch.clamp(torch.log1p(error_pm25), max=1.0)
    loss_pm25 = (huber(pred[..., 0], target[..., 0]) * soft_weight_pm25).mean()

    # ── AQI Loss with Category Weighting + Soft-Focal ──
    loss_aqi_raw = huber(pred[..., 1], target[..., 1])
    error_aqi = torch.abs(pred[..., 1] - target[..., 1])
    soft_weight_aqi = 1.0 + torch.clamp(torch.log1p(error_aqi), max=1.5)

    if aqi_high_weight > 1.0 or aqi_critical_weight > 1.0:
        aqi_target = target[..., 1]
        weights = torch.ones_like(aqi_target)
        q70 = torch.quantile(aqi_target.float(), 0.70)
        q95 = torch.quantile(aqi_target.float(), 0.95)
        weights = torch.where(aqi_target > q70, torch.tensor(1.5, device=pred.device), weights)
        weights = torch.where(aqi_target > q95, torch.tensor(2.0, device=pred.device), weights)
        loss_aqi = (loss_aqi_raw * weights * soft_weight_aqi).mean()
    else:
        loss_aqi = (loss_aqi_raw * soft_weight_aqi).mean()

    # ── Main Loss ──
    main_loss = lambda_pm25 * loss_pm25 + lambda_aqi * loss_aqi

    # ── MAE Regularizer ──
    mae_loss = mae_fn(pred, target)

    # ── Entropy Reg on Gate ──
    entropy_reg = -torch.mean(
        gate * torch.log(gate + 1e-8) +
        (1 - gate) * torch.log(1 - gate + 1e-8)
    )

    total_loss = main_loss + loss_alpha * mae_loss - loss_beta * entropy_reg

    # ── Auxiliary Branch Losses ──
    if 'pred_tm' in pred_dict and 'pred_st' in pred_dict:
        loss_tm = huber(pred_dict['pred_tm'][..., 0], target[..., 0]).mean() * lambda_pm25 + \
                  huber(pred_dict['pred_tm'][..., 1], target[..., 1]).mean() * lambda_aqi
        loss_st = huber(pred_dict['pred_st'][..., 0], target[..., 0]).mean() * lambda_pm25 + \
                  huber(pred_dict['pred_st'][..., 1], target[..., 1]).mean() * lambda_aqi
        total_loss = total_loss + aux_tm_weight * loss_tm + aux_st_weight * loss_st

    return total_loss


class ParallelTrainer:
    """
    Trainer v2 cho Parallel STMixer.
    Cải tiến:
      - CosineAnnealingWarmRestarts (T_0=25, T_mult=2) → LR restart
      - Warmup từ 0.2*lr tăng dần (5 epochs)
      - Patience 25 (dài hơn vì LR restart can tạo oscillation tạm thời)
    """

    def __init__(
        self, model: nn.Module, train_loader, val_loader,
        adj_matrix: torch.Tensor, config: dict = None,
        device: str = None, save_path: str = 'best_parallel_stmixer.pt'
    ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.A = adj_matrix.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        if config is None:
            config = {}

        self.epochs = config.get('epochs', 150)
        self.lr = config.get('lr', 3e-4)
        self.weight_decay = config.get('weight_decay', 1e-4)
        self.grad_clip = config.get('grad_clip', 5.0)
        self.patience = config.get('patience', 25)

        self.lambda_pm25 = config.get('lambda_pm25', 0.6)
        self.lambda_aqi = config.get('lambda_aqi', 0.4)
        self.huber_delta = config.get('huber_delta', 1.0)
        self.loss_alpha = config.get('loss_alpha', 0.1)
        self.loss_beta = config.get('loss_beta', 0.01)
        self.aux_tm_weight = config.get('aux_tm_weight', 0.2)
        self.aux_st_weight = config.get('aux_st_weight', 0.2)
        self.aqi_high_weight = config.get('aqi_high_weight', 2.0)
        self.aqi_critical_weight = config.get('aqi_critical_weight', 3.0)

        self.save_path = save_path
        self.warmup_epochs = config.get('warmup_epochs', 5)

        # ── Optimizer ──
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # ── LR Schedule: Warmup -> CosineWarmRestarts ──
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.2, end_factor=1.0,
            total_iters=self.warmup_epochs
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=25, T_mult=2,
            eta_min=config.get('eta_min', 1e-6)
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_epochs]
        )

        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []

        print(f"[Trainer v2] Device: {self.device}")
        print(f"[Trainer v2] Epochs: {self.epochs}, LR: {self.lr}, Patience: {self.patience}")
        print(f"[Trainer v2] Warmup: {self.warmup_epochs} epochs, "
              f"Aux weights: tm={self.aux_tm_weight}, st={self.aux_st_weight}")

    def _train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            pred_dict = self.model(x, self.A)

            loss = parallel_stmixer_loss(
                pred_dict, y,
                lambda_pm25=self.lambda_pm25, lambda_aqi=self.lambda_aqi,
                huber_delta=self.huber_delta,
                aqi_high_weight=self.aqi_high_weight,
                aqi_critical_weight=self.aqi_critical_weight,
                loss_alpha=self.loss_alpha, loss_beta=self.loss_beta,
                aux_tm_weight=self.aux_tm_weight, aux_st_weight=self.aux_st_weight
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip
            )
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for x, y in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            pred_dict = self.model(x, self.A)
            
            # Validation: no aux loss, no entropy reg
            val_dict = {'pred': pred_dict['pred'], 'gate': pred_dict['gate']}
            loss = parallel_stmixer_loss(
                val_dict, y,
                lambda_pm25=self.lambda_pm25, lambda_aqi=self.lambda_aqi,
                huber_delta=self.huber_delta,
                loss_alpha=self.loss_alpha, loss_beta=0.0,
                aux_tm_weight=0.0, aux_st_weight=0.0
            )
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def fit(self):
        print(f"\n{'='*60}")
        print(f"  BẮT ĐẦU TRAINING — PARALLEL STMIXER v2 ({self.epochs} epochs)")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()
            train_loss = self._train_one_epoch()
            val_loss = self._validate()
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            elapsed = time.time() - epoch_start

            print(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s",
                end=""
            )

            if val_loss < self.best_val_loss:
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.save_path)
                print(f" ★ Best (↓{improvement:.4f})")
            else:
                self.patience_counter += 1
                print(f" (patience {self.patience_counter}/{self.patience})")
                if self.patience_counter >= self.patience:
                    print(f"\n⛔ Early stopping at epoch {epoch}")
                    break

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"  TRAINING COMPLETE")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"  Model saved: {self.save_path}")
        print(f"{'='*60}\n")
        self.model.load_state_dict(torch.load(self.save_path, weights_only=True))

    @torch.no_grad()
    def evaluate(self, test_loader, scaler=None, node_list=None, target_indices=None) -> dict:
        self.model.eval()
        all_preds, all_targets = [], []

        for x, y in test_loader:
            x = x.to(self.device)
            pred_dict = self.model(x, self.A)
            all_preds.append(pred_dict['pred'].cpu().numpy())
            all_targets.append(y.numpy())

        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # Inverse transform
        if scaler is not None and target_indices is not None:
            for ti, feat_idx in enumerate(target_indices):
                if hasattr(scaler, 'center_'):
                    center = scaler.center_[feat_idx]
                    scale_val = scaler.scale_[feat_idx]
                else:
                    center = scaler.mean_[feat_idx]
                    scale_val = scaler.scale_[feat_idx]

                preds[..., ti] = preds[..., ti] * scale_val + center
                targets[..., ti] = targets[..., ti] * scale_val + center

                preds[..., ti] = np.expm1(np.clip(preds[..., ti], -20, 20))
                targets[..., ti] = np.expm1(np.clip(targets[..., ti], -20, 20))

            print(f"  ✅ Inverse transform applied (target indices: {target_indices})")
            print(f"     PM2.5 range: pred [{preds[...,0].min():.1f}, {preds[...,0].max():.1f}], "
                  f"true [{targets[...,0].min():.1f}, {targets[...,0].max():.1f}]")
            print(f"     AQI   range: pred [{preds[...,1].min():.1f}, {preds[...,1].max():.1f}], "
                  f"true [{targets[...,1].min():.1f}, {targets[...,1].max():.1f}]")

        metrics = {}

        # ── Overall Regression ──
        for ti, target_name in enumerate(['pm25', 'aqi']):
            p, t = preds[..., ti], targets[..., ti]
            metrics[f'{target_name}_MAE'] = round(float(np.abs(p - t).mean()), 4)
            metrics[f'{target_name}_RMSE'] = round(float(np.sqrt(((p - t) ** 2).mean())), 4)
            ss_res = ((t - p) ** 2).sum()
            ss_tot = ((t - t.mean()) ** 2).sum()
            metrics[f'{target_name}_R2'] = round(float(1 - ss_res / (ss_tot + 1e-8)), 4)
            smape = (200 * np.abs(p - t) / (np.abs(p) + np.abs(t) + 1e-8)).mean()
            metrics[f'{target_name}_sMAPE'] = round(float(smape), 4)

        # ── AQI Classification ──
        aqi_pred_cats = np.array([aqi_to_category(v) for v in preds[..., 1].flatten()])
        aqi_true_cats = np.array([aqi_to_category(v) for v in targets[..., 1].flatten()])
        metrics['aqi_category_accuracy'] = round(float(accuracy_score(aqi_true_cats, aqi_pred_cats)), 4)
        metrics['aqi_category_f1_macro'] = round(float(
            f1_score(aqi_true_cats, aqi_pred_cats, average='macro', zero_division=0)), 4)
        metrics['aqi_category_f1_weighted'] = round(float(
            f1_score(aqi_true_cats, aqi_pred_cats, average='weighted', zero_division=0)), 4)

        unique_true, counts_true = np.unique(aqi_true_cats, return_counts=True)
        unique_pred, counts_pred = np.unique(aqi_pred_cats, return_counts=True)
        metrics['aqi_true_category_dist'] = dict(zip(unique_true.tolist(), counts_true.tolist()))
        metrics['aqi_pred_category_dist'] = dict(zip(unique_pred.tolist(), counts_pred.tolist()))

        # ── Per-horizon ──
        H = preds.shape[2]
        per_horizon = {}
        for h in range(H):
            p_h = preds[:, :, h, 0]
            t_h = targets[:, :, h, 0]
            mae_h = np.abs(p_h - t_h).mean()
            rmse_h = np.sqrt(((p_h - t_h) ** 2).mean())
            per_horizon[f'h{h+1}'] = {'MAE': round(float(mae_h), 4), 'RMSE': round(float(rmse_h), 4)}
        metrics['per_horizon_pm25'] = per_horizon

        # ── Per-node ──
        N = preds.shape[1]
        per_node = {}
        for ni in range(N):
            p_n = preds[:, ni, :, 0]
            t_n = targets[:, ni, :, 0]
            mae_n = np.abs(p_n - t_n).mean()
            node_name = node_list[ni] if node_list and ni < len(node_list) else f'node_{ni}'
            per_node[node_name] = {'MAE': round(float(mae_n), 4)}
        metrics['per_node_pm25'] = per_node

        return metrics

    def print_results(self, metrics: dict):
        print(f"\n{'='*60}")
        print(f"  KẾT QUẢ ĐÁNH GIÁ — PARALLEL STMIXER v2")
        print(f"{'='*60}")

        print(f"\n📊 Overall Metrics:")
        print(f"  PM2.5 — MAE: {metrics.get('pm25_MAE', '—'):>8} | "
              f"RMSE: {metrics.get('pm25_RMSE', '—'):>8} | "
              f"R²: {metrics.get('pm25_R2', '—'):>8} | "
              f"sMAPE: {metrics.get('pm25_sMAPE', '—'):>8}")
        print(f"  AQI   — MAE: {metrics.get('aqi_MAE', '—'):>8} | "
              f"RMSE: {metrics.get('aqi_RMSE', '—'):>8} | "
              f"R²: {metrics.get('aqi_R2', '—'):>8} | "
              f"sMAPE: {metrics.get('aqi_sMAPE', '—'):>8}")

        print(f"\n🏷️  AQI Category Classification:")
        print(f"  Accuracy:    {metrics.get('aqi_category_accuracy', '—')}")
        print(f"  Macro F1:    {metrics.get('aqi_category_f1_macro', '—')}")
        print(f"  Weighted F1: {metrics.get('aqi_category_f1_weighted', '—')}")

        cat_names = {
            0: 'Tốt(0-50)', 1: 'TB(51-100)', 2: 'Nhạy(101-150)',
            3: 'Xấu(151-200)', 4: 'Rất xấu(201-300)', 5: 'Nguy hiểm(>300)'
        }
        if 'aqi_true_category_dist' in metrics:
            print(f"  True dist:  ", end="")
            for cat, cnt in sorted(metrics['aqi_true_category_dist'].items()):
                print(f" {cat_names.get(cat, cat)}={cnt}", end="")
            print()
            print(f"  Pred dist:  ", end="")
            for cat, cnt in sorted(metrics['aqi_pred_category_dist'].items()):
                print(f" {cat_names.get(cat, cat)}={cnt}", end="")
            print()

        if 'per_horizon_pm25' in metrics:
            print(f"\n📈 Per-Horizon PM2.5 MAE:")
            for h, vals in list(metrics['per_horizon_pm25'].items())[:12]:
                bar = '█' * int(vals['MAE'] * 3)
                print(f"  {h}: MAE={vals['MAE']:.4f} RMSE={vals['RMSE']:.4f} {bar}")

        if 'per_node_pm25' in metrics:
            print(f"\n🗺️  Per-Node PM2.5 MAE:")
            sorted_nodes = sorted(
                metrics['per_node_pm25'].items(), key=lambda x: x[1]['MAE']
            )
            for name, vals in sorted_nodes:
                bar = '█' * int(vals['MAE'] * 3)
                print(f"  {name:>12}: MAE={vals['MAE']:.4f} {bar}")

        print(f"\n{'='*60}\n")
