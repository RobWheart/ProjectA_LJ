import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset_loader import get_dataloaders
from Unet import UNet3D

# ---------------------- 1. Adaptive Lambda Loss ----------------------
class AdaptiveLambdaLoss(nn.Module):
    def __init__(self, epsilon=1e-4, gamma=20.0, target_ratio=0.3):
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
        return total_loss, weighted_mse.item(), mae.item(), lam.item()

# ---------------------- 2. Initialize Weights ----------------------
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

# ---------------------- 3. Metric Evaluation ----------------------
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

# ---------------------- 4. Save Helpers ----------------------
def save_model_if_best(metric_name, current_value, best_metrics, model, epoch, save_dir):
    if metric_name not in best_metrics or current_value < best_metrics[metric_name]['value']:
        best_metrics[metric_name] = {'value': current_value, 'epoch': epoch}
        torch.save(model.state_dict(), os.path.join(save_dir, f"{metric_name}_best.pt"))

def save_metrics_csv(filepath, header, rows):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

# ---------------------- 5. Training Function ----------------------
def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, val_loader = get_dataloaders(mix_ratio=1.0, batch_size_train=8, batch_size_val=2)

    model = UNet3D(in_channels=2, out_channels=1)
    initialize_weights(model)
    model = model.to(device)

    criterion = AdaptiveLambdaLoss(epsilon=1e-4, gamma=20.0, target_ratio=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    num_epochs = 100
    patience = 10
    best_metrics = {}
    all_metrics = []
    best_change_mae = float('inf')
    no_improve_epochs = 0

    save_dir = "Unet_loss_train/models"
    metric_dir = "Unet_loss_train/metric"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_losses, train_maes, train_wmses, train_lams, train_cmaes = [], [], [], [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            preds = model(inputs)
            loss, w_mse, mae_val, lam_val = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics = evaluate_metrics(preds, targets)

            train_losses.append(loss.item())
            train_wmses.append(w_mse)
            train_maes.append(mae_val)
            train_lams.append(lam_val)
            train_cmaes.append(metrics['Change_MAE'])

        # ---------------- 验证 ----------------
        model.eval()
        val_metrics = {k: [] for k in ['Loss','MAE','RMSE','Change_MAE','Positive_MAE','Negative_MAE','Abs_MAE']}
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1} [Valid]"):
                inputs, targets = inputs.to(device), targets.to(device)
                preds = model(inputs)
                batch_metrics = evaluate_metrics(preds, targets)
                for k in val_metrics:
                    val_metrics[k].append(batch_metrics[k])

        row = [
            epoch + 1,
            np.mean(train_losses), np.mean(train_wmses), np.mean(train_maes), np.mean(train_lams), np.mean(train_cmaes),
            np.mean(val_metrics['Loss']), np.mean(val_metrics['MAE']), np.mean(val_metrics['RMSE']),
            np.mean(val_metrics['Change_MAE']), np.mean(val_metrics['Positive_MAE']),
            np.mean(val_metrics['Negative_MAE']), np.mean(val_metrics['Abs_MAE'])
        ]
        all_metrics.append(row)

        tqdm.write(
            f"Epoch {epoch+1} | Train Loss: {row[1]:.4f} W-MSE: {row[2]:.4f} MAE: {row[3]:.4f} λ: {row[4]:.4f} cMAE: {row[5]:.4f} | "
            f"Val Loss: {row[6]:.4f} MAE: {row[7]:.4f} RMSE: {row[8]:.4f} cMAE: {row[9]:.4f}"
        )

        for key, col in zip(['Loss','MAE','RMSE','Change_MAE','Positive_MAE','Negative_MAE','Abs_MAE'], range(6, 13)):
            save_model_if_best(key, row[col], best_metrics, model, epoch + 1, save_dir)

        if row[9] < best_change_mae:
            best_change_mae = row[9]
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            tqdm.write(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

    header = [
        "Epoch", "Train_Loss", "Train_Weighted_MSE", "Train_MAE", "Train_Lambda", "Train_Change_MAE",
        "Val_Loss", "Val_MAE", "Val_RMSE", "Val_Change_MAE",
        "Val_Positive_MAE", "Val_Negative_MAE", "Val_Abs_MAE"
    ]
    save_metrics_csv(os.path.join(metric_dir, "metrics.csv"), header, all_metrics)

    print("\nTraining completed.\nBest models saved:")
    for metric_name, info in best_metrics.items():
        print(f" - {metric_name}_best.pt: epoch {info['epoch']}, {metric_name} = {info['value']:.4f}")

if __name__ == '__main__':
    train()
