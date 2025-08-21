import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset_loader import get_dataloaders
from Unet import UNet3D

# ============ 随机种子设置 ============
def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============ Loss Functions ============
class WeightedMSELoss(nn.Module):
    def __init__(self, epsilon=1e-4, gamma=20.0):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = gamma

    def forward(self, pred, target):
        weight = 1.0 + self.gamma * (target.abs() > self.epsilon).float()
        return (weight * (pred - target) ** 2).mean()

class SoftDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-4, scale=5.0):
        super().__init__()
        self.epsilon = epsilon
        self.scale = scale

    def forward(self, pred, target):
        pred_abs = pred.abs()
        target_abs = target.abs()
        pred_norm = torch.sigmoid(self.scale * pred_abs)
        target_norm = torch.sigmoid(self.scale * target_abs)
        intersection = (pred_norm * target_norm).sum()
        union = pred_norm.sum() + target_norm.sum()
        dice = (2. * intersection + self.epsilon) / (union + self.epsilon)
        return 1 - dice

class CombinedLossWithSoftDice(nn.Module):
    def __init__(self, gamma=5.0, alpha=12.0, epsilon=1e-4, scale=5.0):
        super().__init__()
        self.wmse = WeightedMSELoss(gamma=gamma, epsilon=epsilon)
        self.softdice = SoftDiceLoss(epsilon=epsilon, scale=scale)
        self.alpha = alpha

    def forward(self, pred, target):
        return self.wmse(pred, target) + self.alpha * self.softdice(pred, target)

# ============ Metrics ============
def evaluate_metrics(pred, target, epsilon=1e-4, scale=5.0):
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

        # 硬 Dice
        pred_bin = (pred.abs() > epsilon).float()
        target_bin = (target.abs() > epsilon).float()
        intersection = (pred_bin * target_bin).sum()
        union = pred_bin.sum() + target_bin.sum()
        dice = (2. * intersection + epsilon) / (union + epsilon)

        # Soft Dice（归一化）
        pred_soft = torch.sigmoid(scale * pred.abs())
        target_soft = torch.sigmoid(scale * target.abs())
        soft_intersection = (pred_soft * target_soft).sum()
        soft_union = pred_soft.sum() + target_soft.sum()
        soft_dice = (2. * soft_intersection + epsilon) / (soft_union + epsilon)

        return {
            'Loss': squared_error.mean().item(),
            'MAE': mae,
            'RMSE': rmse,
            'Change_MAE': change_mae,
            'Positive_MAE': pos_mae,
            'Negative_MAE': neg_mae,
            'Abs_MAE': abs_mae,
            'Dice': dice.item(),
            'Soft_Dice': soft_dice.item()
        }

# ============ Training Core ============
def train_model(model, train_loader, val_loader, criterion, optimizer, device, stage, epochs, patience, save_root):
    best_cmae = float('inf')
    no_improve = 0
    all_rows = []
    best_metrics_record = {}

    model_path = os.path.join(save_root, "models")
    metric_path = os.path.join(save_root, "metrics")
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(metric_path, exist_ok=True)

    metric_names = ['Loss','MAE','RMSE','Change_MAE','Positive_MAE','Negative_MAE','Abs_MAE','Dice','Soft_Dice']

    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        train_metrics = {k: [] for k in metric_names}
        for inputs, targets in tqdm(train_loader, desc=f"{stage} Epoch {epoch+1} [Train]"):
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)
            loss = criterion(preds, targets)
            metrics = evaluate_metrics(preds, targets)
            for k in metric_names:
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
                batch_metrics = evaluate_metrics(preds, targets)
                for k in metric_names:
                    val_metrics[k].append(batch_metrics[k])

        train_avg = {k: np.mean(train_metrics[k]) for k in metric_names}
        val_avg = {k: np.mean(val_metrics[k]) for k in metric_names}

        # ---- Print formatted table ----
        print(f"\nEpoch {epoch+1:03d} | Stage: {stage}")
        header = f"{'Split':<6}" + "".join([f"{m:<13}" for m in metric_names])
        print(header)
        print("-"*len(header))
        print(f"{'Train':<6}" + "".join([f"{train_avg[m]:<13.4f}" for m in metric_names]))
        print(f"{'Valid':<6}" + "".join([f"{val_avg[m]:<13.4f}" for m in metric_names]))

        # ---- Save best by Change_MAE ----
        if val_avg['Change_MAE'] < best_cmae:
            best_cmae = val_avg['Change_MAE']
            torch.save(model.state_dict(), os.path.join(model_path, f"{stage}_Change_MAE_best.pt"))
            no_improve = 0
            best_metrics_record = {'epoch': epoch+1, **val_avg}
        else:
            no_improve += 1

        # ---- Save CSV row ----
        row = [epoch+1, stage] + [train_avg[k] for k in metric_names] + [val_avg[k] for k in metric_names]
        all_rows.append(row)

        if no_improve >= patience:
            print(f"Early stopping {stage} at epoch {epoch+1}")
            break

    header = ["Epoch", "Stage"] + ["Train_" + k for k in metric_names] + ["Val_" + k for k in metric_names]
    with open(os.path.join(metric_path, f"{stage}_metrics.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_rows)

    return best_metrics_record

# ============ Main ============
def main():
    set_seed(2025)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, val_loader = get_dataloaders(mix_ratio=1.0, batch_size_train=8, batch_size_val=2)

    param_sets = [
        {"gamma": 2,  "alpha": 10},
        {"gamma": 5,  "alpha": 10},
        {"gamma": 10, "alpha": 20}
    ]
    fine_gammas = [20, 25]
    summary_records = []

    for p in param_sets:
        for gamma_f in fine_gammas:
            tag = f"g{p['gamma']}_a{p['alpha']}_gf{gamma_f}"
            print(f"\n==== Training {tag} ====")

            model = UNet3D(in_channels=2, out_channels=1).to(device)
            model.apply(lambda m: initialize_weights(m, init_type='kaiming'))

            # --- Coarse Phase ---
            criterion_coarse = CombinedLossWithSoftDice(gamma=p['gamma'], alpha=p['alpha'])
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
            best_coarse = train_model(model, train_loader, val_loader, criterion_coarse, optimizer, device, f"{tag}_coarse", 100, 5, "soft_cs_unet_train")

            # --- Fine Phase ---
            criterion_fine = WeightedMSELoss(gamma=gamma_f)
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
            best_fine = train_model(model, train_loader, val_loader, criterion_fine, optimizer, device, f"{tag}_fine", 100, 10, "soft_cs_unet_train")

            summary_records.append({
                'tag': tag,
                'coarse_best': best_coarse,
                'fine_best': best_fine
            })

    # ---- Summary ----
    print("\n=== Training Summary (Best Val Metrics) ===")
    metric_names = ['Loss','MAE','RMSE','Change_MAE','Positive_MAE','Negative_MAE','Abs_MAE','Dice','Soft_Dice']
    header = f"{'Tag':<20}{'Phase':<10}{'Epoch':<8}" + "".join([f"{m:<13}" for m in metric_names])
    print(header)
    print("-"*len(header))
    for rec in summary_records:
        for phase, best in [('Coarse', rec['coarse_best']), ('Fine', rec['fine_best'])]:
            print(f"{rec['tag']:<20}{phase:<10}{best['epoch']:<8}" + "".join([f"{best[m]:<13.4f}" for m in metric_names]))

# ============ Utils ============
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

if __name__ == '__main__':
    main()

