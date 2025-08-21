# train_swinunetr.py
import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from monai.networks.nets import SwinUNETR
from dataset_loader import get_dataloaders

# ============ 随机种子设置 ============
def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============ Adaptive Lambda Loss ============
class AdaptiveLambdaLoss(nn.Module):
    def __init__(self, epsilon=1e-4, gamma=15.0, target_ratio=0.3):  # gamma 改为 15
        super().__init__()
        self.epsilon = epsilon
        self.gamma = gamma
        self.target_ratio = target_ratio
        self.eps = 1e-8

    def forward(self, pred, target):
        weight = 1.0 + self.gamma * (target.abs() > self.epsilon).float()
        weighted_mse = (weight * (pred - target) ** 2).mean()
        mae = (pred - target).abs().mean()
        lam = self.target_ratio * weighted_mse / ((1 - self.target_ratio) * mae + self.eps)
        total_loss = weighted_mse + lam * mae
        return total_loss

# ============ 指标计算 ============
def evaluate_metrics(pred, target, epsilon=1e-4, scale=5.0):
    with torch.no_grad():
        abs_error = (pred - target).abs()
        squared_error = (pred - target) ** 2

        mse = squared_error.mean().item()
        rmse = torch.sqrt(squared_error.mean()).item()
        mae = abs_error.mean().item()
        change_mask = (target.abs() > epsilon)
        change_mae = abs_error[change_mask].mean().item() if change_mask.any() else 0.0
        pos_mask = (target > epsilon)
        pos_mae = abs_error[pos_mask].mean().item() if pos_mask.any() else 0.0
        neg_mask = (target < -epsilon)
        neg_mae = abs_error[neg_mask].mean().item() if neg_mask.any() else 0.0
        abs_mae = (pred.abs() - target.abs()).abs().mean().item()

        pred_bin = (pred.abs() > epsilon).float()
        target_bin = (target.abs() > epsilon).float()
        intersection = (pred_bin * target_bin).sum()
        union = pred_bin.sum() + target_bin.sum()
        dice = (2. * intersection + epsilon) / (union + epsilon)

        pred_soft = torch.sigmoid(scale * pred.abs())
        target_soft = torch.sigmoid(scale * target.abs())
        soft_intersection = (pred_soft * target_soft).sum()
        soft_union = pred_soft.sum() + target_soft.sum()
        soft_dice = (2. * soft_intersection + epsilon) / (soft_union + epsilon)

        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'Change_MAE': change_mae,
            'Positive_MAE': pos_mae,
            'Negative_MAE': neg_mae,
            'Abs_MAE': abs_mae,
            'Dice': dice.item(),
            'Soft_Dice': soft_dice.item()
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

# ============ 训练函数 ============
def train_model(model, train_loader, val_loader, criterion, optimizer, device, stage, save_root, epochs=100, patience=10):
    best_metrics = {}
    best_cmae = float('inf')
    no_improve = 0
    all_rows = []

    model_path = os.path.join(save_root, "models")
    metric_path = os.path.join(save_root, "metrics")
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(metric_path, exist_ok=True)

    metric_names = ['Loss', 'MSE', 'RMSE', 'MAE', 'Change_MAE', 'Positive_MAE', 'Negative_MAE', 'Abs_MAE', 'Dice', 'Soft_Dice']

    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        train_metrics = {k: [] for k in metric_names}
        for inputs, targets in tqdm(train_loader, desc=f"{stage} Epoch {epoch+1} [Train]"):
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)
            loss = criterion(preds, targets)
            metrics = evaluate_metrics(preds, targets)
            train_metrics['Loss'].append(loss.item())
            for k in metrics:
                train_metrics[k].append(metrics[k])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ---- Validation ----
        model.eval()
        val_metrics = {k: [] for k in metric_names}
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"{stage} Epoch {epoch+1} [Valid]"):
                inputs, targets = inputs.to(device), targets.to(device)
                preds = model(inputs)
                loss = criterion(preds, targets)  # 同训练损失函数
                metrics = evaluate_metrics(preds, targets)
                val_metrics['Loss'].append(loss.item())
                for k in metrics:
                    val_metrics[k].append(metrics[k])

        train_avg = {k: np.mean(train_metrics[k]) for k in metric_names}
        val_avg = {k: np.mean(val_metrics[k]) for k in metric_names}

        # ---- 打印表格 ----
        print(f"\nEpoch {epoch+1:03d} | Stage: {stage}")
        print("(Train_Loss / Val_Loss 都是使用训练用的损失函数 AdaptiveLambdaLoss 计算)")
        header = f"{'Split':<8}" + "".join([f"{m:<13}" for m in metric_names])
        print(header)
        print("-" * len(header))
        print(f"{'Train':<8}" + "".join([f"{train_avg[m]:<13.4f}" for m in metric_names]))
        print(f"{'Valid':<8}" + "".join([f"{val_avg[m]:<13.4f}" for m in metric_names]))

        # ---- 保存最佳模型 ----
        for key in metric_names:
            if key in ["Dice", "Soft_Dice"]:  # 最大值更好
                if key not in best_metrics or val_avg[key] > best_metrics[key]['value']:
                    best_metrics[key] = {'value': val_avg[key], 'epoch': epoch+1}
                    torch.save(model.state_dict(), os.path.join(model_path, f"{stage}_{key}_best.pt"))
            else:  # 最小值更好
                if key not in best_metrics or val_avg[key] < best_metrics[key]['value']:
                    best_metrics[key] = {'value': val_avg[key], 'epoch': epoch+1}
                    torch.save(model.state_dict(), os.path.join(model_path, f"{stage}_{key}_best.pt"))

        # ---- 早停逻辑 ----
        if val_avg['Change_MAE'] < best_cmae:
            best_cmae = val_avg['Change_MAE']
            no_improve = 0
        else:
            no_improve += 1

        # ---- 保存到 CSV ----
        row = [epoch+1, stage] + [train_avg[m] for m in metric_names] + [val_avg[m] for m in metric_names]
        all_rows.append(row)

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # ---- 写入 CSV 文件 ----
    header = ["Epoch", "Stage"] + ["Train_" + m for m in metric_names] + ["Val_" + m for m in metric_names]
    with open(os.path.join(metric_path, f"{stage}_metrics.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_rows)

# ============ 主程序 ============
if __name__ == '__main__':
    set_seed(2025)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    target_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    gamma = 15.0             # gamma 改为 15
    lr_swin = 1e-3           # 学习率改为 0.001
    feature_size_swin = 24

    for tr in target_ratios:
        print(f"\n===== Training SwinUNETR with target_ratio={tr} =====")
        train_loader, val_loader = get_dataloaders(mix_ratio=1.0, batch_size_train=4, batch_size_val=2)

        model = SwinUNETR(
            in_channels=2,
            out_channels=1,
            feature_size=feature_size_swin,
            use_checkpoint=False
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr_swin)
        save_root = "Swin_UNETR_loss"
        criterion = AdaptiveLambdaLoss(epsilon=1e-4, gamma=gamma, target_ratio=tr)
        stage = f"SwinUNETR_tr{tr}"

        train_model(model, train_loader, val_loader, criterion, optimizer, device, stage, save_root, epochs=100, patience=10)

