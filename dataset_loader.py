# dataset_loader.py

import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import random

class MRIDiffDatasetFlexible(Dataset):
    def __init__(self, root_dirs, sample_range, mix_ratio=1.0, random_seed=42):
        self.root_dirs = root_dirs
        self.sample_range = sample_range
        self.mix_ratio = mix_ratio
        self.random_seed = random_seed

        self.sample_entries = []
        for root in root_dirs:
            for sid in range(sample_range[0], sample_range[1] + 1):
                self.sample_entries.append({
                    "root": root,
                    "id": f"{sid:03d}"
                })

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.shuffle(self.sample_entries)
        self.mode_choices = [
            np.random.rand() < self.mix_ratio for _ in self.sample_entries
        ]

    def __len__(self):
        return len(self.sample_entries)

    def __getitem__(self, idx):
        entry = self.sample_entries[idx]
        sid = entry["id"]
        root = entry["root"]
        use_A = self.mode_choices[idx]

        noise_tag = 'A' if use_A else 'B'
        snr = 300 if use_A else 200

        pre_path = os.path.join(root, f"sample_{sid}_noisy{noise_tag}_pre_SNR{snr}.nii.gz")
        post_path = os.path.join(root, f"sample_{sid}_noisy{noise_tag}_post_SNR{snr}.nii.gz")

        pre_img = nib.load(pre_path).get_fdata().astype(np.float32)
        post_img = nib.load(post_path).get_fdata().astype(np.float32)
        input_tensor = np.stack([pre_img, post_img], axis=0)
        input_tensor = torch.from_numpy(input_tensor)

        # label 与 sample_id 相同目录中，编号相同
        label_path = os.path.join(root, f"sample_{sid}_label.nii.gz")
        label_img = nib.load(label_path).get_fdata().astype(np.float32)
        label_tensor = torch.from_numpy(label_img).unsqueeze(0)  # [1, D, H, W]

        return input_tensor, label_tensor

def get_dataloaders(
    mix_ratio=1.0,
    batch_size_train=4,
    batch_size_val=2,
    num_workers=4,
    seed=2025
):
    # 训练集
    train_dataset = MRIDiffDatasetFlexible(
        root_dirs=["output_dataset/train", "data2"],
        sample_range=(0, 299),
        mix_ratio=mix_ratio,
        random_seed=seed
    )

    # 验证集 = output_dataset/val + data2[300–339]
    val_dataset_part1 = MRIDiffDatasetFlexible(
        root_dirs=["output_dataset/val"],
        sample_range=(0, 39),
        mix_ratio=mix_ratio,
        random_seed=seed
    )

    val_dataset_part2 = MRIDiffDatasetFlexible(
        root_dirs=["data2"],
        sample_range=(300, 339),
        mix_ratio=mix_ratio,
        random_seed=seed
    )

    val_dataset = ConcatDataset([val_dataset_part1, val_dataset_part2])

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, num_workers=num_workers)

    return train_loader, val_loader
