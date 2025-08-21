import os
import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm

# =========================================
# 1) Core geometry & volume functions
# =========================================

def generate_centerline(t_vals, a1, b1, a2, b2):
    x = a1 * np.sin(np.pi * b1 * t_vals)
    y = t_vals
    z = a2 * np.sin(np.pi * b2 * t_vals)
    return np.stack([x, y, z], axis=1)

def radius_function(t_vals, a3, a4, b3, c, scale=1.0, min_radius=0.02):
    # quadratic segment
    r = np.where(
        t_vals <= c,
        a3 * t_vals**2,
        (a4 * scale) * np.exp(-b3 * (t_vals - c))
    )
    # truncate any tail below min_radius
    r[r < min_radius] = 0.0
    return r

def compute_arc_lengths(centerline):
    diffs = np.diff(centerline, axis=0)
    return np.linalg.norm(diffs, axis=1)

def approximate_volume(a3, a4, b3, c, centerline_params,
                       scale, min_radius, n_int=2000):
    t = np.linspace(0, 1, n_int, dtype=np.float64)
    center = generate_centerline(t, *centerline_params)
    lengths = compute_arc_lengths(center)
    radii = radius_function(t, a3, a4, b3, c, scale=scale, min_radius=min_radius)
    integrand = np.pi * radii[:-1]**2 * lengths
    return integrand.sum()

# =========================================
# 2) Voxelization
# =========================================

def generate_structure(grid_size, sub_factor, t_samples,
                       centerline_params, radius_params):
    """
    radius_params = (a3, a4, b3, c, scale, min_radius)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # subvoxel grid
    spacing = 2.0 / grid_size
    sub_s = spacing / sub_factor
    coords_1d = np.linspace(-1 + sub_s/2, 1 - sub_s/2,
                             grid_size*sub_factor, dtype=np.float32)
    X,Y,Z = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
    coords_sub = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    N_sub = coords_sub.shape[0]

    a3,a4,b3,c,scale,min_radius = radius_params
    t_vals = np.linspace(0,1,t_samples, dtype=np.float32)
    centerline = generate_centerline(t_vals, *centerline_params)
    radii = radius_function(t_vals, a3, a4, b3, c,
                            scale=scale, min_radius=min_radius)

    center_t = torch.tensor(centerline, dtype=torch.float32, device=device)
    rad2_t   = torch.tensor(radii**2, dtype=torch.float32, device=device)
    coords_t = torch.tensor(coords_sub, dtype=torch.float32, device=device)

    inside = torch.zeros(N_sub, dtype=torch.bool, device=device)
    chunk = 500_000
    for start in range(0, N_sub, chunk):
        end = min(start+chunk, N_sub)
        subc = coords_t[start:end][:,None,:]
        dist2 = ((subc - center_t[None,:,:])**2).sum(dim=2)
        inside[start:end] = dist2.lt(rad2_t).any(dim=1)

    inside_np = inside.cpu().numpy()
    sub_mask = inside_np.reshape(
        grid_size, sub_factor,
        grid_size, sub_factor,
        grid_size, sub_factor
    )
    return sub_mask.mean(axis=(1,3,5)).astype(np.float32)

# =========================================
# 3) Noise
# =========================================

def add_gaussian_noise(volume, snr):
    mask = volume > 0
    mean_signal = volume[mask].mean() if mask.any() else volume.mean()
    sigma = (mean_signal / snr) if mean_signal>0 else 0.0
    noisy = volume + np.random.normal(0, sigma, size=volume.shape)
    return np.clip(noisy, 0.0, 1.0)

# =========================================
# 4) Dataset generation
# =========================================

def generate_dataset(num_samples: int,
                     output_dir: str = "data2",
                     grid_size: int = 96,
                     sub_factor: int = 6,
                     t_samples: int = 300,
                     snrA: float = 300.0,
                     snrB: float = 200.0,
                     target_frac: float = 0.05,
                     min_radius: float = 0.02):
    os.makedirs(output_dir, exist_ok=True)
    for i in tqdm(range(num_samples), desc="Generating samples"):
        # 1) random geometry
        centerline_params = np.random.uniform(0.3, 0.6, size=4).astype(np.float64)
        a3 = np.random.uniform(0.12, 0.2)
        a4 = np.random.uniform(0.08, 0.16)
        b3 = np.random.uniform(3.0, 5.0)
        c  = np.random.uniform(0.3, 0.5)

        # 2) approximate original volume
        V0 = approximate_volume(a3, a4, b3, c,
                                centerline_params,
                                scale=1.0,
                                min_radius=min_radius)

        # 3) binary search for scale on exponential segment
        low, high, tol = 0.5, 1.0, 1e-3
        for _ in range(12):
            mid = (low + high)/2
            Vn = approximate_volume(a3, a4, b3, c,
                                    centerline_params,
                                    scale=mid,
                                    min_radius=min_radius)
            frac = (V0 - Vn)/V0
            if abs(frac - target_frac) < tol:
                break
            if frac > target_frac:
                low = mid
            else:
                high = mid
        scale = mid

        # 4) generate clean_pre / clean_post / label
        rp0 = (a3, a4, b3, c, 1.0, min_radius)
        rp1 = (a3, a4, b3, c, scale, min_radius)
        pre  = generate_structure(grid_size, sub_factor, t_samples,
                                  centerline_params, rp0)
        post = generate_structure(grid_size, sub_factor, t_samples,
                                  centerline_params, rp1)
        label = post - pre

        # 5) save volumes
        base = os.path.join(output_dir, f"sample_{i:03d}")
        nib.save(nib.Nifti1Image(pre,  np.eye(4)), base + "_clean_pre.nii.gz")
        nib.save(nib.Nifti1Image(post, np.eye(4)), base + "_clean_post.nii.gz")
        nib.save(nib.Nifti1Image(label,np.eye(4)), base + "_label.nii.gz")

        # 6) add & save noise A (SNR=snrA)
        na_pre  = add_gaussian_noise(pre,  snrA)
        na_post = add_gaussian_noise(post, snrA)
        nib.save(nib.Nifti1Image(na_pre,  np.eye(4)),
                 base + f"_noisyA_pre_SNR{int(snrA)}.nii.gz")
        nib.save(nib.Nifti1Image(na_post, np.eye(4)),
                 base + f"_noisyA_post_SNR{int(snrA)}.nii.gz")

        # 7) add & save noise B (SNR=snrB)
        nb_pre  = add_gaussian_noise(pre,  snrB)
        nb_post = add_gaussian_noise(post, snrB)
        nib.save(nib.Nifti1Image(nb_pre,  np.eye(4)),
                 base + f"_noisyB_pre_SNR{int(snrB)}.nii.gz")
        nib.save(nib.Nifti1Image(nb_post, np.eye(4)),
                 base + f"_noisyB_post_SNR{int(snrB)}.nii.gz")

if __name__ == "__main__":
    # 参数：生成多少组数据
    N = 400
    generate_dataset(N, output_dir="data2")