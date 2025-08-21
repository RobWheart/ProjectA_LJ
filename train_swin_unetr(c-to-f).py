import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from monai.networks.nets import SwinUNETR
from dataset_loader import get_dataloaders

# 固定随机种子
def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 可选: 完全确定性
    # torch.use_deterministic_algorithms(True)

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
            'CriterionLoss': None,  # 占位，后面补
            'WeightedLoss': weighted_mse,
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'Change_MAE': change_mae,
            'Positive_MAE': pos_mae,
            'Negative_MAE': neg_mae,
            'Abs_MAE': abs_mae
        }

# ---------------------- 单阶段训练 ----------------------
def train_phase(model, train_loader, val_loader, criterion, optimizer, device, tag, stage, gamma, save_root, epochs=100, patience=10):
    metric_names = ['CriterionLoss','WeightedLoss','MSE','MAE','RMSE','Change_MAE','Positive_MAE','Negative_MAE','Abs_MAE']

    model_path = os.path.join(save_root, "models")
    metric_path = os.path.join(save_root, "metric")
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(metric_path, exist_ok=True)

    metric_file = os.path.join(metric_path, f"{tag}_{stage}_metrics.csv")

    best_metrics = {}
    all_metrics = []
    best_result = {}
    best_cmae = float('inf')
    no_improve = 0

    for epoch in range(1, epochs+1):
        # ---------------------- Training ----------------------
        model.train()
        train_metrics = {k: [] for k in metric_names}
        for inputs, targets in tqdm(train_loader, desc=f"[{stage}-Train] Epoch {epoch}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics = evaluate_metrics(preds, targets, gamma=gamma)
            metrics['CriterionLoss'] = loss.item()
            for k in metric_names:
                train_metrics[k].append(metrics[k])

        # ---------------------- Validation ----------------------
        model.eval()
        val_metrics = {k: [] for k in metric_names}
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"[{stage}-Valid] Epoch {epoch}", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                preds = model(inputs)
                val_loss = criterion(preds, targets)
                metrics = evaluate_metrics(preds, targets, gamma=gamma)
                metrics['CriterionLoss'] = val_loss.item()
                for k in metric_names:
                    val_metrics[k].append(metrics[k])

        train_avg = {k: np.mean(train_metrics[k]) for k in metric_names}
        val_avg = {k: np.mean(val_metrics[k]) for k in metric_names}

        # 打印表格
        print(f"\nEpoch {epoch:03d} | {tag}-{stage}")
        print(f"{'Metric':<15}{'Train':>12}{'Valid':>12}")
        print("-"*40)
        for m in metric_names:
            print(f"{m:<15}{train_avg[m]:>12.4f}{val_avg[m]:>12.4f}")

        # 保存最优模型（每个指标单独保存）
        for key in metric_names:
            if key not in best_metrics or val_avg[key] < best_metrics[key]['value']:
                best_metrics[key] = {'value': val_avg[key], 'epoch': epoch}
                torch.save(model.state_dict(), os.path.join(model_path, f"{tag}_{stage}_{key}_best.pt"))

        # Early Stopping by Change_MAE
        if val_avg['Change_MAE'] < best_cmae:
            best_cmae = val_avg['Change_MAE']
            no_improve = 0
            best_result = {'epoch': epoch, **val_avg}
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"\nEarly stopping {stage} at epoch {epoch}")
            break

        # 记录 CSV
        row = [epoch] + [train_avg[k] for k in metric_names] + [val_avg[k] for k in metric_names]
        all_metrics.append(row)

    # 保存到 CSV
    header = ["Epoch"] + ["Train_" + m for m in metric_names] + ["Val_" + m for m in metric_names]
    with open(metric_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_metrics)

    return best_result, best_metrics

# ---------------------- 主流程 ----------------------
def main():
    set_seed(2025)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_dataloaders(seed=2025, batch_size_train=3, batch_size_val=2)

    save_root = "swinUNETR_CF"
    feature_size = 24
    lr = 1e-3  # 学习率固定 0.001

    param_sets = [
        {"gamma": 2, "alpha": 10},
        {"gamma": 5, "alpha": 10},
        {"gamma": 10, "alpha": 20}
    ]
    fine_gammas = [15, 20]

    summary_records = []
    overall_best = {}

    for p in param_sets:
        for gamma_f in fine_gammas:
            tag = f"fs{feature_size}_g{p['gamma']}_a{p['alpha']}_gf{gamma_f}"
            print(f"\n=== Training {tag} ===")

            # 初始化模型
            model = SwinUNETR(in_channels=2, out_channels=1, feature_size=feature_size, use_checkpoint=False).to(device)

            # --- Coarse Phase ---
            criterion_coarse = WeightedMSELoss(gamma=p['gamma'])
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            best_coarse, _ = train_phase(model, train_loader, val_loader, criterion_coarse, optimizer, device, tag, "coarse", p['gamma'], save_root, epochs=100, patience=5)

            # --- Fine Phase ---
            criterion_fine = WeightedMSELoss(gamma=gamma_f)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            best_fine, _ = train_phase(model, train_loader, val_loader, criterion_fine, optimizer, device, tag, "fine", gamma_f, save_root, epochs=100, patience=10)

            summary_records.append({'tag': tag, 'coarse': best_coarse, 'fine': best_fine})

            # 更新总体最佳
            for phase, result in [('coarse', best_coarse), ('fine', best_fine)]:
                for metric, value in result.items():
                    if metric == 'epoch': 
                        continue
                    if metric not in overall_best or value < overall_best[metric]['value']:
                        overall_best[metric] = {'value': value, 'tag': tag, 'phase': phase, 'epoch': result['epoch']}

    # ---------------------- 汇总输出 ----------------------
    print("\n=== Training Summary (Best Val Metrics per config) ===")
    for rec in summary_records:
        print(f"\n{rec['tag']}")
        for phase in ['coarse','fine']:
            best = rec[phase]
            print(f" {phase.capitalize()} (epoch {best['epoch']})")
            for m, v in best.items():
                if m != 'epoch':
                    print(f"   {m:<12}: {v:.4f}")

    print("\n=== Overall Best Across All Configs ===")
    for m, v in overall_best.items():
        print(f"{m:<12}: {v['value']:.4f} | {v['tag']} ({v['phase']} epoch {v['epoch']})")

if __name__ == "__main__":
    main()
