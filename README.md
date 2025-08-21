# Brain Volume Change Detection Project

This repository contains code for generating synthetic MRI datasets and training deep learning models (U-Net and SwinUNETR) for detecting brain volume changes at sub-voxel scale. The project supports both local deformations (spherical bumps) and global atrophy simulations.

---

## 📂 Project Structure

### 1. Data Generation
- **`data_maker.py`**  
  Generates synthetic MRI data with **local spherical deformations** (expansion/shrinkage).
- **`datamakerv2.py`**  
  Generates synthetic MRI data with **global atrophy**.

### 2. Model Architectures
- **`Unet.py`**  
  Defines the U-Net architecture (encoder, decoder, and skip connections).

### 3. Training Scripts

#### SwinUNETR-based
- **`train_swin_unetr(adaptive).py`** – SwinUNETR with adaptive loss  
- **`train_swin_unetr(c-to-f).py`** – SwinUNETR with coarse-to-fine training  
- **`train_swin_unetr(WMSE).py`** – SwinUNETR with weighted MSE loss  

#### U-Net-based
- **`train_unet-attention.py`** – U-Net with attention mechanism  
- **`train_unet(adaptive).py`** – U-Net with adaptive loss  
- **`train_unet(adaptive)t.py`** – Duplicate/backup of the adaptive version  
- **`train_unet(c-to-f).py`** – U-Net with coarse-to-fine training  
- **`train_unet(WMSE).py`** – U-Net with weighted MSE loss  

---

## ⚙️ Usage

1. **Data Generation**  
   - Run `data_maker.py` for local spherical deformations.  
   - Run `datamakerv2.py` for global atrophy simulations.  
   - **Note:** Modify the output paths in the scripts to your desired dataset directory.

2. **Model Training**  
   - Choose one of the training scripts (`train_*`) depending on the model and loss function you want to use.  
   - Each training script loads data and trains the corresponding model.  
   - **Note:** Update dataset paths (`train/val/test`) inside the scripts before running.

   Example:
   ```bash
   python train_unet(WMSE).py
