import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset_loader import get_dataloaders
from Unet import UNet3D

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
def evaluate_metrics(pred, target, epsilon=1e-4):
    with torch.no_grad():
        abs_error = (pred - target).abs()
        squared_error = (pred - target) ** 2

        mae = abs_error.mean().item()
        rmse = torch.sqrt(squared_error.mean()).item()

        change_mask = (target.abs() > epsilon)
        change_mae = abs_error[change_mask].mean().item() if change_mask.any() else 0.0

        pos_mask = (target > epsilon)
        pos_mae = abs_error[pos_mask].mean().item() if pos_mask.any() else 0.0

        neg_mask = (target < -epsilon)
        neg_mae = abs_error[neg_mask].mean().item() if neg_mask.any() else 0.0

        abs_mae = (pred.abs() - target.abs()).abs().mean().item()

        return {
            'Loss': squared_error.mean().item(),
            'MAE': mae,
            'RMSE': rmse,
            'Change_MAE': change_mae,
            'Positive_MAE': pos_mae,
            'Negative_MAE': neg_mae,
            'Abs_MAE': abs_mae
        }

# ---------------------- 权重初始化 ----------------------
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

# ---------------------- 工具函数 ----------------------
def save_metrics_csv(filepath, header, rows):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 可选：为了严格可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def format_lr(lr: float) -> str:
    # 用科学计数法，文件名友好
    return f"{lr:.0e}"

def pretty_epoch_log(epoch, tr, va):
    # 对齐输出
    return (
        f"Epoch {epoch:03d} | "
        f"Train: Loss {tr['Loss']:.4f}  MAE {tr['MAE']:.4f}  RMSE {tr['RMSE']:.4f}  cMAE {tr['Change_MAE']:.4f} | "
        f"Val:   Loss {va['Loss']:.4f}  MAE {va['MAE']:.4f}  RMSE {va['RMSE']:.4f}  cMAE {va['Change_MAE']:.4f}"
    )

# ---------------------- 单个组合的训练 ----------------------
def train_one_combo(lr, gamma, device='cuda', num_epochs=100, patience=10, epsilon=1e-4):
    set_seed(2025)

    # 数据加载（保持你原先设置）
    train_loader, val_loader = get_dataloaders(mix_ratio=1.0, batch_size_train=8, batch_size_val=2)

    # 模型
    model = UNet3D(in_channels=2, out_channels=1)
    initialize_weights(model, init_type='kaiming')
    model = model.to(device)

    # 针对该 gamma 的损失函数
    criterion = WeightedMSELoss(epsilon=epsilon, gamma=gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 目录
    base_dir = "Unet_train2/metric"
    os.makedirs(base_dir, exist_ok=True)

    # 日志与保存
    all_rows = []
    header = [
        "Epoch",
        "Train_Loss", "Train_MAE", "Train_RMSE", "Train_Change_MAE",
        "Val_Loss", "Val_MAE", "Val_RMSE", "Val_Change_MAE",
        "Val_Positive_MAE", "Val_Negative_MAE", "Val_Abs_MAE"
    ]

    # 针对该组合的最优记录与权重
    best = {
        "Val_MAE": {"value": float('inf'), "epoch": -1},
        "Val_Change_MAE": {"value": float('inf'), "epoch": -1},
        "Val_RMSE": {"value": float('inf'), "epoch": -1},
    }

    lr_tag = format_lr(lr)  # e.g., 5e-04
    csv_path = os.path.join(base_dir, f"metrics_lr{lr_tag}_gamma{gamma}.csv")

    best_cmae_for_early_stop = float('inf')
    no_improve_epochs = 0

    for epoch in range(1, num_epochs + 1):
        # -------- Train --------
        model.train()
        train_losses, train_maes, train_rmses, train_cmaes = [], [], [], []
        pbar = tqdm(train_loader, desc=f"[lr={lr_tag} γ={gamma}] Epoch {epoch:03d} [Train]", leave=False)
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            preds = model(inputs)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            m = evaluate_metrics(preds, targets, epsilon=epsilon)
            train_losses.append(loss.item())
            train_maes.append(m['MAE'])
            train_rmses.append(m['RMSE'])
            train_cmaes.append(m['Change_MAE'])

        train_log = {
            "Loss": float(np.mean(train_losses)),
            "MAE": float(np.mean(train_maes)),
            "RMSE": float(np.mean(train_rmses)),
            "Change_MAE": float(np.mean(train_cmaes)),
        }

        # -------- Valid --------
        model.eval()
        agg = {k: [] for k in ['Loss','MAE','RMSE','Change_MAE','Positive_MAE','Negative_MAE','Abs_MAE']}
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"[lr={lr_tag} γ={gamma}] Epoch {epoch:03d} [Valid]", leave=False):
                inputs = inputs.to(device)
                targets = targets.to(device)
                preds = model(inputs)
                mv = evaluate_metrics(preds, targets, epsilon=epsilon)
                for k in agg:
                    agg[k].append(mv[k])

        val_log = {k: float(np.mean(agg[k])) for k in agg}

        # 打印整齐的每 epoch 指标
        tqdm.write(pretty_epoch_log(epoch, train_log, val_log))

        # 保存每 epoch 行
        row = [
            epoch,
            train_log['Loss'], train_log['MAE'], train_log['RMSE'], train_log['Change_MAE'],
            val_log['Loss'], val_log['MAE'], val_log['RMSE'], val_log['Change_MAE'],
            val_log['Positive_MAE'], val_log['Negative_MAE'], val_log['Abs_MAE']
        ]
        all_rows.append(row)

        # —— 保存各指标最佳权重（存到 Unet_train2/metric）——
        def save_best(metric_key, value):
            if value < best[metric_key]["value"]:
                best[metric_key] = {"value": value, "epoch": epoch}
                save_path = os.path.join(base_dir, f"{metric_key}_best_lr{lr_tag}_gamma{gamma}.pt")
                torch.save(model.state_dict(), save_path)

        save_best("Val_MAE", val_log['MAE'])
        save_best("Val_Change_MAE", val_log['Change_MAE'])
        save_best("Val_RMSE", val_log['RMSE'])

        # —— 早停（使用 Val_Change_MAE）——
        if val_log['Change_MAE'] < best_cmae_for_early_stop:
            best_cmae_for_early_stop = val_log['Change_MAE']
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            tqdm.write(f"[lr={lr_tag} γ={gamma}] Early stopping at epoch {epoch:03d} (no improvement {patience} epochs).")
            break

    # 保存 CSV
    save_metrics_csv(csv_path, header, all_rows)

    # 返回该组合的验证集最佳记录（用于全局排名）
    return {
        "lr": lr,
        "gamma": gamma,
        "best": best,  # dict: metric -> {'value','epoch'}
        "csv": csv_path
    }

# ---------------------- 网格搜索主流程 ----------------------
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    lrs = [1e-4, 1e-3, 5e-3, 5e-4]
    gammas = [25, 15, 10, 20]

    results = []  # 每个组合的 best 记录
    tqdm.write("=== Grid Search Start ===")
    tqdm.write(f"Device: {device}")

    for lr in lrs:
        for gamma in gammas:
            tqdm.write(f"\n--- Training combo: lr={lr:.0e}, gamma={gamma} ---")
            combo_res = train_one_combo(lr=lr, gamma=gamma, device=device, num_epochs=100, patience=10, epsilon=1e-4)
            results.append(combo_res)

    # 全局最佳
    global_best = {
        "Val_MAE": {"value": float('inf'), "lr": None, "gamma": None, "epoch": None},
        "Val_Change_MAE": {"value": float('inf'), "lr": None, "gamma": None, "epoch": None},
        "Val_RMSE": {"value": float('inf'), "lr": None, "gamma": None, "epoch": None},
    }

    for r in results:
        lr, gamma = r["lr"], r["gamma"]
        for metric in global_best.keys():
            v = r["best"][metric]["value"]
            ep = r["best"][metric]["epoch"]
            if v < global_best[metric]["value"]:
                global_best[metric] = {"value": v, "lr": lr, "gamma": gamma, "epoch": ep}

    # 整齐打印全局最佳
    def fmt_lr(x): return f"{x:.0e}"
    line = "=" * 66
    print("\n" + line)
    print("Global Best (Validation)".center(66))
    print(line)
    print(f"{'Metric':<18}{'Best Value':>14}    {'lr':>8}    {'gamma':>6}    {'epoch':>6}")
    print("-" * 66)
    for metric in ["Val_MAE", "Val_Change_MAE", "Val_RMSE"]:
        gb = global_best[metric]
        print(f"{metric:<18}{gb['value']:>14.6f}    {fmt_lr(gb['lr']):>8}    {gb['gamma']:>6d}    {gb['epoch']:>6d}")
    print(line)
    print("All per-epoch logs and best weights are under: Unet_train2/metric")
    print(line + "\n")

if __name__ == '__main__':
    main()
