import os
import csv
import itertools
import random
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from monai.networks.nets import SwinUNETR
from dataset_loader import get_dataloaders

# 固定随机种子
random_seed = 2025
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------- 损失函数 ----------------------
class WeightedMSELoss(nn.Module):
    def __init__(self, epsilon=1e-4, gamma=20.0):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = gamma

    def forward(self, pred, target):
        weight = 1.0 + self.gamma * (target.abs() > self.epsilon).float()
        loss = weight * (pred - target) ** 2
        return loss.mean()

# ---------------------- 评估指标 ----------------------
def evaluate_metrics(pred, target, epsilon=1e-4, gamma=None):
    with torch.no_grad():
        abs_error = (pred - target).abs()
        squared_error = (pred - target) ** 2

        mse = squared_error.mean().item()

        if gamma is not None:
            weight = 1.0 + gamma * (target.abs() > epsilon).float()
            weighted_mse = (weight * squared_error).mean().item()
        else:
            weighted_mse = mse

        rmse = torch.sqrt(squared_error.mean()).item()
        mae = abs_error.mean().item()
        change_mask = (target.abs() > epsilon)
        change_mae = abs_error[change_mask].mean().item() if change_mask.any() else 0.0
        pos_mask = (target > epsilon)
        pos_mae = abs_error[pos_mask].mean().item() if pos_mask.any() else 0.0
        neg_mask = (target < -epsilon)
        neg_mae = abs_error[neg_mask].mean().item() if neg_mask.any() else 0.0
        abs_mae = (pred.abs() - target.abs()).abs().mean().item()

        return {
            'WeightedLoss': weighted_mse,
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'Change_MAE': change_mae,
            'Positive_MAE': pos_mae,
            'Negative_MAE': neg_mae,
            'Abs_MAE': abs_mae
        }

# ---------------------- 训练函数 ----------------------
def train_swin_unetr(feature_size, gamma, lr, activation, optimizer_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_dataloaders(
        seed=random_seed,
        batch_size_train=3,  # 改成 3
        batch_size_val=2
    )

    print(f"Using batch_size: train={train_loader.batch_size}, val={val_loader.batch_size}")

    model = SwinUNETR(
        in_channels=2,
        out_channels=1,
        feature_size=feature_size,
        use_checkpoint=False
    ).to(device)

    if activation == "gelu":
        model.activation = nn.GELU()
    elif activation == "leakyrelu":
        model.activation = nn.LeakyReLU(inplace=True)
    else:
        model.activation = nn.ReLU(inplace=True)

    criterion = WeightedMSELoss(epsilon=1e-4, gamma=gamma)

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    save_root = "swinUNETR2"
    model_path = os.path.join(save_root, "models")
    metric_path = os.path.join(save_root, "metrics")
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(metric_path, exist_ok=True)

    save_name = f"fs{feature_size}_g{gamma}_lr{lr}"
    metric_file = os.path.join(metric_path, f"{save_name}.csv")

    best_metrics = {}
    all_metrics = []
    best_cmae = float('inf')
    patience, no_improve = 10, 0
    best_result = {}

    metric_names = ['WeightedLoss', 'MSE', 'MAE', 'RMSE', 'Change_MAE', 'Positive_MAE', 'Negative_MAE', 'Abs_MAE']

    for epoch in range(1, 101):
        # ----- Training -----
        model.train()
        train_metrics = {k: [] for k in metric_names}
        for inputs, targets in tqdm(train_loader, desc=f"[Train] Epoch {epoch}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics = evaluate_metrics(preds, targets, epsilon=1e-4, gamma=gamma)
            for k in metric_names:
                train_metrics[k].append(metrics[k])

        # ----- Validation -----
        model.eval()
        val_metrics = {k: [] for k in metric_names}
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"[Valid] Epoch {epoch}", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                preds = model(inputs)
                metrics = evaluate_metrics(preds, targets, epsilon=1e-4, gamma=gamma)
                for k in metric_names:
                    val_metrics[k].append(metrics[k])

        train_avg = {k: np.mean(train_metrics[k]) for k in metric_names}
        val_avg = {k: np.mean(val_metrics[k]) for k in metric_names}

        # 打印表格
        header = f"{'Metric':<15}{'Train':>12}{'Valid':>12}"
        print(f"\nEpoch {epoch:03d} | fs={feature_size}, gamma={gamma}, lr={lr}")
        print(header)
        print("-" * len(header))
        for m in metric_names:
            print(f"{m:<15}{train_avg[m]:>12.4f}{val_avg[m]:>12.4f}")

        # 保存验证指标最优模型
        for key in metric_names:
            if key not in best_metrics or val_avg[key] < best_metrics[key]['value']:
                best_metrics[key] = {'value': val_avg[key], 'epoch': epoch}
                torch.save(model.state_dict(), os.path.join(model_path, f"{save_name}_{key}_best.pt"))

        # 提前停止条件：Change_MAE 改善
        if val_avg['Change_MAE'] < best_cmae:
            best_cmae = val_avg['Change_MAE']
            no_improve = 0
            best_result = {
                "epoch": epoch,
                "Change_MAE": val_avg["Change_MAE"],
                "MAE": val_avg["MAE"],
                "WeightedLoss": val_avg["WeightedLoss"],
                "RMSE": val_avg["RMSE"]
            }
        else:
            no_improve += 1

        # 保存一行到 CSV 数据缓存
        row = [epoch] + list(train_avg.values()) + list(val_avg.values())
        all_metrics.append(row)

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # 保存到 CSV
    header = ["Epoch"] + ["Train_" + m for m in metric_names] + ["Val_" + m for m in metric_names]
    with open(metric_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_metrics)

    # 打印本组合的最佳验证指标
    print(f"\n=== Best validation metrics for {save_name} ===")
    for k, v in best_metrics.items():
        print(f"{k}: {v['value']:.4f} (epoch {v['epoch']})")

    return best_result

# ---------------------- 网格搜索入口 ----------------------
if __name__ == '__main__':
    feature_size = 24
    lr_list = [1e-4, 1e-3, 5e-3, 5e-4]
    gamma_list = [25, 15, 10, 20]
    activation = "relu"
    optimizer = "adam"

    summary_path = "swinUNETR2/summary.csv"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary_header = ["feature_size", "gamma", "lr", "epoch", "Change_MAE", "MAE", "WeightedLoss", "RMSE"]
    summary_rows = []

    for gamma, lr in itertools.product(gamma_list, lr_list):
        print(f"\n=== Training Swin UNETR with feature_size={feature_size}, gamma={gamma}, lr={lr} ===")
        result = train_swin_unetr(feature_size, gamma, lr, activation, optimizer)
        summary_rows.append([feature_size, gamma, lr, result["epoch"], result["Change_MAE"], result["MAE"], result["WeightedLoss"], result["RMSE"]])

    # 保存总表
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(summary_header)
        writer.writerows(summary_rows)

    # 打印最优参数组合
    best_row = min(summary_rows, key=lambda x: x[4])
    print("\n=== Best configuration based on Val_Change_MAE ===")
    print(f"feature_size={best_row[0]}, gamma={best_row[1]}, lr={best_row[2]}")
    print(f"Epoch {best_row[3]} | Change_MAE={best_row[4]:.4f} | MAE={best_row[5]:.4f} | WeightedLoss={best_row[6]:.4f} | RMSE={best_row[7]:.4f}")
