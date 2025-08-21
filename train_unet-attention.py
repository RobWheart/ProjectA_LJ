import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset_loader import get_dataloaders
from Unet_BAM import UNet_BAM

# ============ 设置随机种子 ============
def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============ 损失函数 ============
class WeightedMSELoss(nn.Module):
    def __init__(self, epsilon=1e-4, gamma=20.0):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = gamma

    def forward(self, pred, target):
        weight = 1.0 + self.gamma * (target.abs() > self.epsilon).float()
        loss = weight * (pred - target) ** 2
        return loss.mean()

# ============ 计算多种指标 ============
def evaluate_metrics(pred, target, epsilon=1e-4):
    with torch.no_grad():
        abs_error = (pred - target).abs()
        squared_error = (pred - target) ** 2

        mae = abs_error.mean().item()
        rmse = torch.sqrt(squared_error.mean()).item()

        change_mask = (target.abs() > epsilon)
        change_mae = abs_error[change_mask].mean().item() if change_mask.any() else 0.0

        return {
            'Loss': ((pred - target) ** 2).mean().item(),
            'MAE': mae,
            'RMSE': rmse,
            'Change_MAE': change_mae
        }

# ============ 权重初始化 ============
def initialize_weights(model, init_type='kaiming'):
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            if init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

# ============ 保存 CSV ============
def save_metrics_csv(filepath, header, rows):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

# ============ 保存最佳模型 ============
def save_model_if_best(metric_name, current_value, best_metrics, model, epoch, save_dir):
    if metric_name not in best_metrics or current_value < best_metrics[metric_name]['value']:
        best_metrics[metric_name] = {'value': current_value, 'epoch': epoch}
        save_path = os.path.join(save_dir, f"{metric_name}_best.pt")
        torch.save(model.state_dict(), save_path)

# ============ 训练主函数 ============
def train():
    set_seed(2025)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader = get_dataloaders(mix_ratio=1.0, batch_size_train=8, batch_size_val=2)

    model = UNet_BAM(in_channels=2, out_channels=1)
    initialize_weights(model, init_type='kaiming')
    model = model.to(device)

    criterion = WeightedMSELoss(epsilon=1e-4, gamma=20)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    num_epochs = 100
    patience = 10
    best_metrics = {}
    all_metrics = []
    best_change_mae = float('inf')
    no_improve_epochs = 0

    os.makedirs("unet-attention/models", exist_ok=True)
    os.makedirs("unet-attention/metric", exist_ok=True)

    for epoch in range(num_epochs):
        # ============ 训练 ============
        model.train()
        train_loss, train_mae, train_cmae, train_rmse = [], [], [], []

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics = evaluate_metrics(preds, targets)
            train_loss.append(loss.item())
            train_mae.append(metrics['MAE'])
            train_cmae.append(metrics['Change_MAE'])
            train_rmse.append(metrics['RMSE'])

        # ============ 验证 ============
        model.eval()
        val_loss, val_mae, val_cmae, val_rmse = [], [], [], []

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1} [Valid]"):
                inputs, targets = inputs.to(device), targets.to(device)
                preds = model(inputs)
                metrics = evaluate_metrics(preds, targets)
                val_loss.append(metrics['Loss'])
                val_mae.append(metrics['MAE'])
                val_cmae.append(metrics['Change_MAE'])
                val_rmse.append(metrics['RMSE'])

        # 计算均值
        row = [
            epoch + 1,
            np.mean(train_loss), np.mean(train_mae), np.mean(train_cmae), np.mean(train_rmse),
            np.mean(val_loss), np.mean(val_mae), np.mean(val_cmae), np.mean(val_rmse)
        ]
        all_metrics.append(row)

        # 打印
        tqdm.write(
            f"Epoch {epoch+1} | "
            f"Train Loss(γ=20): {row[1]:.4f}, MAE: {row[2]:.4f}, cMAE: {row[3]:.4f}, RMSE: {row[4]:.4f} | "
            f"Val Loss(γ=20): {row[5]:.4f}, MAE: {row[6]:.4f}, cMAE: {row[7]:.4f}, RMSE: {row[8]:.4f}"
        )

        # 保存最佳模型
        for name, value in zip(['Loss', 'MAE', 'Change_MAE', 'RMSE'], row[5:]):
            save_model_if_best(name, value, best_metrics, model, epoch + 1, "unet-attention/models")

        # 早停
        if row[7] < best_change_mae:
            best_change_mae = row[7]
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            tqdm.write(f"\nEarly stopping at epoch {epoch+1}.")
            break

    # 保存 CSV
    header = ["Epoch", "Train_Loss", "Train_MAE", "Train_Change_MAE", "Train_RMSE",
              "Val_Loss", "Val_MAE", "Val_Change_MAE", "Val_RMSE"]
    save_metrics_csv("unet-attention/metric/metrics.csv", header, all_metrics)

    # 汇报最佳指标
    print("\nTraining completed. Best validation results:")
    for metric_name, info in best_metrics.items():
        print(f"{metric_name} best at epoch {info['epoch']}: {info['value']:.4f}")

if __name__ == '__main__':
    train()

