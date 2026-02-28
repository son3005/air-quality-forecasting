import os
import torch
import torch.nn as nn
import numpy as np
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score


class SandwichTrainer:
    """
    Trình huấn luyện dành riêng cho mô hình T-S-T Sandwich.
    Tập trung vào Multi-task Loss (Huber Loss cho đứt gãy hồi quy, CrossEntropy cho phân loại AQI).
    Không cần Auxiliary Loss phức tạp vì gradient thẳng mạch!
    """

    def __init__(self, model, train_loader, val_loader, adj_matrix, config, save_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.adj_matrix = adj_matrix.to(self.device)
        self.save_path = save_path

        # Tách cấu hình Hyperparameters
        self.epochs = config['epochs']
        self.patience = config['patience']
        self.grad_clip = config.get('grad_clip', 5.0)

        # Multi-task Params
        self.lambda_pm25 = config.get('lambda_pm25', 0.6)
        self.lambda_aqi = config.get('lambda_aqi', 0.4)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config['lr'], 
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # LR Scheduler (Cosine Annealing with Warm Restarts chống mắc kẹt)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=config.get('T_max', 50), 
            T_mult=1, 
            eta_min=config.get('eta_min', 1e-6)
        )

        # Các hàm Loss
        # PM2.5 là bài toán Hồi quy (Regression)
        self.pm25_criterion = nn.HuberLoss(delta=config.get('huber_delta', 1.0))
        
        # AQI là bài toán Classification, ta dùng MSE kết hợp trọng số động ép viền
        # Ở đây dùng MSE nhưng đẩy weight cho các mẫu có AQI bứt ngưỡng (vùng High/Critical)
        self.aqi_high_weight = config.get('aqi_high_weight', 2.0)
        self.aqi_critical_weight = config.get('aqi_critical_weight', 3.0)

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> dict:
        """
        Tính toán Loss kết hợp cho 2 nhiệm vụ:
            - Target 0: PM2.5
            - Target 1: AQI
        """
        # (B, N, H, Targets)
        # 1. Hồi quy PM2.5 (Huber Loss vì chống outlier cực tốt)
        pred_pm25 = y_pred[..., 0]
        true_pm25 = y_true[..., 0]
        loss_pm25 = self.pm25_criterion(pred_pm25, true_pm25)
        
        # 2. Phân loại ngầm AQI (Weighted MSE)
        pred_aqi = y_pred[..., 1]
        true_aqi = y_true[..., 1]
        
        # Ép trọng số cho AQI theo phân khúc (Dành riêng cho thang đo Custom)
        weight_aqi = torch.ones_like(true_aqi)
        weight_aqi[true_aqi >= 150] = self.aqi_high_weight
        weight_aqi[true_aqi >= 200] = self.aqi_critical_weight
        
        loss_aqi = torch.mean(weight_aqi * (pred_aqi - true_aqi) ** 2)

        # Trộn tổng Loss
        loss_total = (self.lambda_pm25 * loss_pm25) + (self.lambda_aqi * loss_aqi)

        return {
            'total': loss_total,
            'pm25': loss_pm25.item(),
            'aqi': loss_aqi.item()
        }

    def train_epoch(self) -> dict:
        self.model.train()
        total_losses = {'total': 0.0, 'pm25': 0.0, 'aqi': 0.0}

        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            y_pred = self.model(x, self.adj_matrix)
            
            # Loss chính
            loss_dict = self.compute_loss(y_pred, y)
            loss = loss_dict['total']

            loss.backward()
            
            # Chặn nổ Gradient GCN
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            # Tích lũy Report log
            total_losses['total'] += loss.item()
            total_losses['pm25'] += loss_dict['pm25']
            total_losses['aqi'] += loss_dict['aqi']

        num_batches = len(self.train_loader)
        return {k: v / num_batches for k, v in total_losses.items()}

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        total_losses = {'total': 0.0, 'pm25': 0.0, 'aqi': 0.0}

        for x, y in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            y_pred = self.model(x, self.adj_matrix)
            loss_dict = self.compute_loss(y_pred, y)

            total_losses['total'] += loss_dict['total'].item()
            total_losses['pm25'] += loss_dict['pm25']
            total_losses['aqi'] += loss_dict['aqi']

        num_batches = len(self.val_loader)
        return {k: v / num_batches for k, v in total_losses.items()}

    def fit(self):
        best_val_loss = float('inf')
        patience_counter = 0
        train_times = []

        print("\n" + "="*50)
        print(" BẮT ĐẦU TRAINING: MÔ HÌNH SANDWICH T-S-T")
        print("="*50)

        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            train_loss = self.train_epoch()
            val_loss = self.validate()
            epoch_time = time.time() - start_time
            train_times.append(epoch_time)
            
            self.scheduler.step()

            print(f"Epoch {epoch:03d} | Lr: {self.scheduler.get_last_lr()[0]:.2e} | Time: {epoch_time:.2f}s")
            print(f"  Train -> Total: {train_loss['total']:.4f} | PM2.5: {train_loss['pm25']:.4f} | AQI: {train_loss['aqi']:.4f}")
            print(f"  Val   -> Total: {val_loss['total']:.4f}  | PM2.5: {val_loss['pm25']:.4f}  | AQI: {val_loss['aqi']:.4f}")

            # Lưu mô hình tốt nhất (Early stopping dựa trên Total Loss của hệ thống)
            if val_loss['total'] < best_val_loss:
                best_val_loss = val_loss['total']
                patience_counter = 0
                torch.save(self.model.state_dict(), self.save_path)
                print(f"  [*] Đã lưu mô hình Sandwich (Valid Loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"\n  [!] Early stopping ở Epoch {epoch}. Patience rớt = {self.patience}")
                break

        print(f"\nTổng thời gian Train: {sum(train_times) / 60:.2f} phút")

    @torch.no_grad()
    def evaluate(self, test_loader, scaler, node_list, target_indices):
        """
        Bước đánh giá cuối cùng.
        Nhả Inverse Transform của Scaler để tính MAE và các phép phân loại (Accuracy, F1).
        """
        print("\n" + "="*50)
        print(" ĐÁNH GIÁ TRÊN TẬP TEST GỐC (INVERSE SCALE)")
        print("="*50)
        
        self.model.load_state_dict(torch.load(self.save_path))
        self.model.eval()

        predictions, truths = [], []

        for x, y in test_loader:
            x = x.to(self.device)
            y_pred = self.model(x, self.adj_matrix)
            predictions.append(y_pred.cpu().numpy())
            truths.append(y.numpy())

        # Gộp Batch: (Total_Samples, Nodes, Horizon, Targets: 2)
        predictions = np.concatenate(predictions, axis=0)
        truths = np.concatenate(truths, axis=0)

        results = {}

        # MỤC TIÊU 1: MAE, RMSE CHO PM2.5 ============================
        pm25_idx = target_indices.index(target_indices[0])
        pred_pm25 = predictions[..., 0] 
        true_pm25 = truths[..., 0]

        # Hàm trợ giúp Inverse Transform từng cột cho RobustScaler
        def inverse_single_target(data_array, feature_idx):
            # Tạo dummy array bằng kích thước features ban đầu của scaler
            dummy = np.zeros((len(data_array.flatten()), len(self.train_loader.dataset.feature_cols)))
            dummy[:, feature_idx] = data_array.flatten()
            inv_dummy = scaler.inverse_transform(dummy)
            return inv_dummy[:, feature_idx].reshape(data_array.shape)

        # Khôi phục giá trị thực
        pred_pm25_real = inverse_single_target(pred_pm25, pm25_idx)
        true_pm25_real = inverse_single_target(true_pm25, pm25_idx)

        # Exponential inverse transform (Vì trước đó target đã bị tính np.log1p)
        pred_pm25_real = np.expm1(pred_pm25_real)
        true_pm25_real = np.expm1(true_pm25_real)

        # Metric tổng hợp cơ bản
        mae = mean_absolute_error(true_pm25_real.flatten(), pred_pm25_real.flatten())
        rmse = np.sqrt(mean_squared_error(true_pm25_real.flatten(), pred_pm25_real.flatten()))
        r2 = r2_score(true_pm25_real.flatten(), pred_pm25_real.flatten())
        
        # SMAPE (Symmetric Mean Absolute Percentage Error)
        denominator = (np.abs(true_pm25_real) + np.abs(pred_pm25_real)) / 2.0
        smape = np.mean(np.where(denominator == 0, 0, np.abs(true_pm25_real - pred_pm25_real) / denominator)) * 100
        
        results['PM2.5_MAE'] = round(mae, 4)
        results['PM2.5_RMSE'] = round(rmse, 4)
        results['PM2.5_R2'] = round(r2, 4)
        results['PM2.5_SMAPE'] = round(smape, 4)
        
        # Breakdown theo Node (Trạm)
        print("\n [Phân tích PM2.5 theo Từng Trạm]")
        for n_idx, node_name in enumerate(node_list):
            node_true = true_pm25_real[:, n_idx, :]
            node_pred = pred_pm25_real[:, n_idx, :]
            node_mae = mean_absolute_error(node_true.flatten(), node_pred.flatten())
            results[f'{node_name}_MAE'] = round(node_mae, 4)
            print(f"      - {node_name:<10}: MAE = {node_mae:.4f}")
            
        # Breakdown theo Horizon (Step thời gian)
        print("\n [Phân tích PM2.5 theo Horizon]")
        for t_idx in range(self.model.pred_len):
            hz_true = true_pm25_real[:, :, t_idx]
            hz_pred = pred_pm25_real[:, :, t_idx]
            hz_mae = mean_absolute_error(hz_true.flatten(), hz_pred.flatten())
            results[f'Horizon_{t_idx+1}_MAE'] = round(hz_mae, 4)
            print(f"      - Tương lai +{t_idx+1}h : MAE = {hz_mae:.4f}")

        # MỤC TIÊU 2: PHÂN CẤP KHÍ TƯỢNG (ACCURACY & F1) CỦA AQI ===
        # Thay đổi từ hồi quy sang Phân loại nhãn cứng
        aqi_idx = target_indices.index(target_indices[1])
        pred_aqi = predictions[..., 1]
        true_aqi = truths[..., 1]
        
        pred_aqi_real = inverse_single_target(pred_aqi, aqi_idx)
        true_aqi_real = inverse_single_target(true_aqi, aqi_idx)

        # Lùi biến đổi Exp giống PM2.5
        pred_aqi_real = np.expm1(pred_aqi_real)
        true_aqi_real = np.expm1(true_aqi_real)

        # Mũi nhọn cải thiện Accuracy:
        # Hàm biến mức AQI thành hạng mục (Category)
        def discretize_aqi(aqi_array):
            bins = [0, 50, 100, 150, 200, 300, 500]
            labels = [0, 1, 2, 3, 4, 5]
            return np.digitize(aqi_array, bins, right=False) - 1

        pred_labels = discretize_aqi(pred_aqi_real.flatten())
        true_labels = discretize_aqi(true_aqi_real.flatten())
        
        # Lọc biên out-of-bounds nếu mô hình bắn quá lố
        pred_labels = np.clip(pred_labels, 0, 5)
        true_labels = np.clip(true_labels, 0, 5)

        acc = accuracy_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, average='macro')
        
        results['AQI_Accuracy'] = round(acc, 4)
        results['AQI_F1_Macro'] = round(f1, 4)

        return results

    def print_results(self, results):
        print("\n [KẾT QUẢ TỔNG QUAN ĐÁNH GIÁ MÔ HÌNH SANDWICH T-S-T]")
        print(" -------------------------------------------")
        print(" 📌 Dự báo liên tục Hồi quy (PM2.5):")
        print(f"      - MAE       : {results.get('PM2.5_MAE', 'N/A')}")
        print(f"      - RMSE      : {results.get('PM2.5_RMSE', 'N/A')}")
        print(f"      - R2-Score  : {results.get('PM2.5_R2', 'N/A')}")
        print(f"      - SMAPE     : {results.get('PM2.5_SMAPE', 'N/A')}%")
        print(" 📌 Dự báo Phân lớp nhóm chất lượng (AQI):")
        print(f"      - Accuracy  : {results.get('AQI_Accuracy', 'N/A')}")
        print(f"      - F1 Macro  : {results.get('AQI_F1_Macro', 'N/A')}")
        print(" ===========================================")
