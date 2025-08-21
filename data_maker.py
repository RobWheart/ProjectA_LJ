import os
import time
import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm

# =========================================
# 1) Basic functions for generating tubular structures (without or with bumps)
# =========================================

def generate_centerline(t_vals, a1, b1, a2, b2):
    """
    Generate a centerline curve in 3D space based on given parameters.
    Inputs:
      - t_vals: 1D array in [0,1], used to sample points along the curve
      - a1, b1: control the amplitude and frequency of x = sin(π b1 t)
      - a2, b2: control the amplitude and frequency of z = sin(π b2 t)
      y coordinate equals t itself, ensuring the curve rises from bottom to top.
    Returns:
      - numpy array of shape (len(t_vals), 3), representing (x,y,z) coordinates.
    """
    x = a1 * np.sin(np.pi * b1 * t_vals)
    y = t_vals
    z = a2 * np.sin(np.pi * b2 * t_vals)
    return np.stack([x, y, z], axis=1)  # Each row is a (x,y,z) coordinate


def radius_function(t_vals, a3, a4, b3, c):
    """
    Define how the "radius" along the centerline changes with t:
      - When t <= c: radius = a3 * t^2 (parabolic growth)
      - When t >  c: radius = a4 * exp(-b3 * (t - c)) (exponential decay)
    Inputs:
      - t_vals: 1D array, x ∈ [0,1]
      - a3, a4, b3, c: parameters controlling the radius function shape
    Returns:
      - 1D array of the same length as t_vals, giving radius at each t.
    """
    return np.where(t_vals <= c,
                    a3 * t_vals**2,
                    a4 * np.exp(-b3 * (t_vals - c)))


def generate_structure(grid_size, sub_factor, t_samples, centerline_params, radius_params, bumps=[]):
    """
    Generate a downsampled 3D volume simulating a tubular structure,
    with optional boundary "bumps" (expansions or subtractions).

    Inputs:
      - grid_size: side length of coarse voxel grid (e.g., 96)
      - sub_factor: each coarse voxel is subdivided into sub_factor^3 subvoxels (e.g., 6)
      - t_samples: number of samples along the centerline (e.g., 300)
      - centerline_params: tuple (a1, b1, a2, b2), for generating the centerline
      - radius_params:     tuple (a3, a4, b3, c), for generating the radius profile
      - bumps: list of (bump_center, bump_radius, mode), where
          * bump_center: numpy array of length 3, center in physical coordinates
          * bump_radius: float, radius in physical space
          * mode: 'add' = add this bump outside the structure; 'remove' = subtract inside
        If bumps is empty, a clean tube is generated without deformation.

    Steps:
      1) Construct all subvoxel center coordinates in physical space: coords_sub (shape = (grid_size*sub_factor)^3, 3).
      2) Generate the centerline coordinates and corresponding radius array:
         centerline (t_samples,3), radii (t_samples,)
      3) For each subvoxel, compute squared distances to all centerline points,
         and check whether any distance < corresponding radius². If yes, it's inside.
      4) If bumps are given, modify the inside mask by adding or removing bumps:
         - 'add': mark all subvoxels within bump radius as inside (True).
         - 'remove': mark them as outside (False).
      5) Reshape inside_np to (grid_size, sub_factor, grid_size, sub_factor, grid_size, sub_factor),
         then average sub_factor^3 subvoxels within each coarse voxel to get downsampled volume (gt_volume),
         with values in [0,1], representing the fraction of voxels belonging to the structure.

    Returns:
      - gt_volume: np.ndarray, shape=(grid_size,grid_size,grid_size), dtype=float32
          Downsampled volume indicating proportion of each voxel belonging to the structure.
      - inside_np: np.ndarray, shape=((grid_size*sub_factor)^3,), dtype=bool
          Flattened mask indicating which subvoxels are inside (True) or outside (False).
      - coords_sub: np.ndarray, shape=((grid_size*sub_factor)^3,3), dtype=float32
          Physical coordinates (x,y,z) of all subvoxels, in range [-1,1]^3.
    """
    # Choose GPU if available for distance calculation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spacing = 2.0 / grid_size          # Physical size of each coarse voxel in [-1,1]
    sub_spacing = spacing / sub_factor # Size of each subvoxel

    # 1) Build all subvoxel center coordinates in physical space (grid_size*sub_factor)^3 × 3
    xs = np.linspace(-1 + sub_spacing/2, 1 - sub_spacing/2, grid_size * sub_factor)
    ys = np.linspace(-1 + sub_spacing/2, 1 - sub_spacing/2, grid_size * sub_factor)
    zs = np.linspace(-1 + sub_spacing/2, 1 - sub_spacing/2, grid_size * sub_factor)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    coords_sub = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)
    N_sub = coords_sub.shape[0]  # = (grid_size*sub_factor)^3

    # 2) Generate centerline and radii along it
    t_vals = np.linspace(0, 1, t_samples)
    centerline = generate_centerline(t_vals, *centerline_params)  # (t_samples, 3)
    radii = radius_function(t_vals, *radius_params)               # (t_samples,)

    centerline_t = torch.tensor(centerline, dtype=torch.float32, device=device)  # (t_samples,3)
    radius_sq_t  = torch.tensor(radii**2, dtype=torch.float32, device=device)    # (t_samples,)
    coords_t     = torch.tensor(coords_sub, dtype=torch.float32, device=device)  # (N_sub,3)

    # 3) Determine which subvoxels are inside the clean tube: if distance² to any centerline point < radius², mark as inside
    inside = torch.zeros(N_sub, dtype=torch.bool, device=device)
    chunk_size = 500_000  # Process in chunks to avoid memory overflow
    for start in range(0, N_sub, chunk_size):
        end = min(start + chunk_size, N_sub)
        sub_chunk = coords_t[start:end]                            # (chunk, 3)
        diff = sub_chunk[:, None, :] - centerline_t[None, :, :]   # (chunk, t_samples, 3)
        dist_sq = (diff ** 2).sum(dim=2)                           # (chunk, t_samples)
        inside[start:end] = (dist_sq < radius_sq_t).any(dim=1)    # Inside if any t satisfies the condition

    # 4) If bumps are provided, apply add/remove operations on the inside mask
    if bumps:
        for (b_center, b_radius, b_mode) in bumps:
            b_center_t = torch.tensor(b_center, dtype=torch.float32, device=device)  # (3,)
            diff_b = coords_t - b_center_t                                          # (N_sub, 3)
            dist_sq_b = (diff_b ** 2).sum(dim=1)                                     # (N_sub,)
            bump_mask = dist_sq_b < (b_radius ** 2)                                  # (N_sub,)
            if b_mode == 'add':
                inside |= bump_mask     # Add bump: mark all inside bump sphere as inside
            else:
                inside &= ~bump_mask    # Remove bump: mark all inside bump sphere as outside

    inside_np = inside.cpu().numpy()  # Convert back to numpy boolean array, shape = (N_sub,)

    # 5) Downsample: average sub_factor^3 subvoxels per coarse voxel to get antialiased result
    sub_mask = inside_np.reshape((grid_size, sub_factor,
                                  grid_size, sub_factor,
                                  grid_size, sub_factor))
    gt_volume = sub_mask.mean(axis=(1, 3, 5)).astype(np.float32)  # shape = (grid_size, grid_size, grid_size)

    return gt_volume, inside_np, coords_sub

# ===================================
# 2) Function to generate a "complete spherical bump"
# ===================================
def generate_bump_full_sphere(
    gt_volume_coarse,
    coords_sub,
    target_volume,
    grid_size,
    sub_factor,
    mode='add',
    epsilon=1e-3
):
    """
    Given a downsampled coarse voxel volume `gt_volume_coarse` and sub-voxel coordinates `coords_sub`,
    generate a "complete spherical bump" such that:
      1) The physical volume of the sphere ≈ target_volume
      2) If mode='add', most of the sphere lies outside the original structure, with a thin part inserted inside ("expansion");
         If mode='remove', most of the sphere lies inside the original structure, with a thin part extending outside ("shrinkage").
      3) The sphere is connected to the surface of the original structure, i.e., a sub-voxel within a boundary voxel is chosen as
         the "reference point", and the sphere center is placed by slightly extrapolating/inward-projecting in the direction from the origin.

    Inputs:
      - gt_volume_coarse: numpy array, shape=(grid_size,grid_size,grid_size),
          Coarse voxel proportion after downsampling, used to find boundary voxels
      - coords_sub: numpy array, shape=((grid_size*sub_factor)^3,3),
          Physical coordinates of all sub-voxels
      - target_volume: float, desired volume of the sphere (in physical volume units)
      - grid_size, sub_factor: same as above
      - mode: 'add' or 'remove'
      - epsilon: float, used to slightly shift the center to ensure the sphere is not fully buried inside or outside

    Steps:
      1) Compute radius r = (3·|target_volume| / (4π))^(1/3)
      2) Find "boundary voxels" in gt_volume_coarse: those with proportion strictly between (0,1);
         If none found, fallback to all voxels with proportion > 0.5 (approximate boundary).
      3) Randomly select one (i,j,k) from boundary_indices, then randomly select a sub-voxel within
         [i*sub_factor : (i+1)*sub_factor, ...], i.e., (sub_i, sub_j, sub_k),
         compute its flat index in coords_sub as idx_flat, and retrieve physical coordinate center_sub.
      4) Compute unit direction vector dir_vec = center_sub / ||center_sub||.
         If mode='add', sphere center = center_sub + dir_vec*(r - ε); else center = center_sub - dir_vec*(r - ε).
      5) Return (bump_center, r, mode).

    Returns:
      - bump_center: numpy array (3,); physical coordinate of the sphere center (x,y,z)
      - r: float; sphere radius
      - mode: original input 'add' or 'remove'
    """
    # 1) Compute sphere radius r
    r = (3 * abs(target_volume) / (4 * np.pi)) ** (1/3)

    # 2) Find "boundary voxels": with values in (0,1)
    boundary_mask = (gt_volume_coarse > 0.0) & (gt_volume_coarse < 1.0)
    boundary_indices = np.argwhere(boundary_mask)
    if len(boundary_indices) == 0:
        # If strict boundary voxels not found, fallback to those with values > 0.5
        boundary_indices = np.argwhere(gt_volume_coarse > 0.5)

    # 3) Randomly select one surface voxel (i,j,k)
    i, j, k = boundary_indices[np.random.choice(len(boundary_indices))]

    # Randomly select a sub-voxel within that coarse voxel
    sub_i = np.random.randint(i * sub_factor, (i + 1) * sub_factor)
    sub_j = np.random.randint(j * sub_factor, (j + 1) * sub_factor)
    sub_k = np.random.randint(k * sub_factor, (k + 1) * sub_factor)
    idx_flat = (sub_i * (grid_size * sub_factor) * (grid_size * sub_factor)
                + sub_j * (grid_size * sub_factor)
                + sub_k)
    center_sub = coords_sub[idx_flat]  # Physical coordinate (x,y,z)

    # 4) Compute direction vector dir_vec = center_sub / ||center_sub||
    norm_val = np.linalg.norm(center_sub) + 1e-12
    dir_vec = center_sub / norm_val

    # 5) Place sphere center based on mode
    if mode == 'add':
        bump_center = center_sub + dir_vec * (r - epsilon)
    else:  # 'remove'
        bump_center = center_sub - dir_vec * (r - epsilon)

    return (bump_center, r, mode)


# ===================================
# 3) Utility function to add Gaussian noise (based on SNR)
# ===================================
def add_gaussian_noise(volume, snr):
    """
    Add Gaussian noise to a 3D volume and clip it to [0,1].
    Noise standard deviation sigma = signal_mean / snr, where signal_mean = mean(volume[volume>0]),
    If the volume is all zeros, use mean(volume) instead.

    Inputs:
      - volume: numpy array, float32, value range [0,1]
      - snr: float, signal-to-noise ratio; lower value ⇒ more noise; higher value ⇒ less noise

    Returns:
      - noisy_vol: numpy array, float32, volume with added noise and clipped to [0,1]
    """
    mask_nonzero = volume > 0
    if np.any(mask_nonzero):
        signal_mean = np.mean(volume[mask_nonzero])
    else:
        signal_mean = np.mean(volume)
    sigma = signal_mean / snr if signal_mean > 0 else 0.0

    noise = np.random.normal(0, sigma, size=volume.shape)
    noisy_vol = volume + noise
    return np.clip(noisy_vol, 0.0, 1.0)



# ==============================================
# 4) Generate a single dataset (train/val): shape + snrA/sn rB + segmented delta_frac
# ==============================================
def generate_dataset(
    n_clean,
    snrA,
    snrB,
    bump_ratios,            # Strict ratio: proportion of 1/2/3 bumps, e.g. [0.5, 0.3, 0.2]
    expand_ratio,           # Probability of applying "expansion (add)" among all samples; rest use "shrinkage (remove)"
    bump_alpha,             # Dirichlet concentration parameter α
    min_bump_frac,          # Minimum fraction of total volume change per bump, e.g. 0.05
    grid_size,
    sub_factor,
    t_samples,
    # The following two parameters control segmented sampling of delta_frac:
    delta_frac_split,       # float, divide delta_frac into [low_min, split] and [split, high_max]
    delta_frac_ratio_high,  # float, sampling ratio for the high segment [split, high_max]; low segment is (1 - ratio_high)
    low_frac_range,         # tuple(min, max) = (0.02, 0.035), sampling range for low segment
    high_frac_range,        # tuple(min, max) = (0.035, 0.05), sampling range for high segment
    output_dir,
    is_training
):
    """
    Generate a dataset satisfying:
      - Each sample strictly assigned 1/2/3 bumps
      - delta_frac is first determined by delta_frac_ratio_high to sample from low or high range, then uniformly sampled in the range
      - Generate clean_pre, clean_post, label = clean_post - clean_pre
      - Generate noisyA_pre/post (snrA), and (only for training set) noisyB_pre/post (snrB)

    Output files (e.g., for sample_{i:03d}):
      clean_pre, clean_post, label,
      noisyA_pre_SNR{snrA}, noisyA_post_SNR{snrA},
      (if is_training) noisyB_pre_SNR{snrB}, noisyB_post_SNR{snrB}

    Parameters:
      - n_clean: int, number of clean samples to generate
      - snrA: float, SNR for initial noise A (used for both train and val)
      - snrB: float, SNR for enhanced noise B (used only in training)
      - bump_ratios: list of 3 floats, proportion of "1/2/3 bumps" strictly enforced
      - expand_ratio: float, proportion of samples using "add"; the rest use "remove"
      - bump_alpha: float, Dirichlet α parameter controlling randomness of bump volume distribution
      - min_bump_frac: float, ensure each bump occupies at least this fraction of the total volume change
      - grid_size, sub_factor, t_samples: grid parameters
      - delta_frac_split: float, split point e.g., 0.035
      - delta_frac_ratio_high: float, probability of choosing [split, high_frac_range[1]] range
      - low_frac_range: (0.02, 0.035), range for low segment
      - high_frac_range: (0.035, 0.05), range for high segment
      - output_dir: str, output directory (automatically created if not exist)
      - is_training: bool, True ⇒ generate noisyB; False ⇒ skip noisyB
    """
    # Validate bump_ratios
    r1, r2, r3 = bump_ratios
    assert abs(r1 + r2 + r3 - 1.0) < 1e-8, "`bump_ratios` must sum to 1.0"

    # -------- (A) Strictly assign 1/2/3 bumps to n_clean samples --------
    n1 = int(np.floor(n_clean * r1))
    n2 = int(np.floor(n_clean * r2))
    n3 = int(np.floor(n_clean * r3))
    total_assigned = n1 + n2 + n3
    remainder = n_clean - total_assigned

    # Assign remaining samples to the most prevalent category
    ratio_idx = sorted([(r1, 1), (r2, 2), (r3, 3)], key=lambda x: x[0], reverse=True)
    idx = 0
    while remainder > 0:
        target_label = ratio_idx[idx % 3][1]
        if target_label == 1:
            n1 += 1
        elif target_label == 2:
            n2 += 1
        else:
            n3 += 1
        idx += 1
        remainder -= 1

    assert n1 + n2 + n3 == n_clean, "Mismatch in strict sample assignment!"

    bump_assignment = [1] * n1 + [2] * n2 + [3] * n3
    np.random.shuffle(bump_assignment)  # Shuffle order

    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    desc = "Training set" if is_training else "Validation set"

    for i in tqdm(range(n_clean), desc=f"[{desc}] Generating samples", unit="sample"):
        # ------------------ 1) Random geometric & radius parameters ------------------
        centerline_params = np.random.uniform(0.3, 0.6, size=4)  # (a1, b1, a2, b2)
        a3 = np.random.uniform(0.12, 0.2)
        a4 = np.random.uniform(0.08, 0.16)
        b3 = np.random.uniform(3.0, 5.0)
        c  = np.random.uniform(0.3, 0.5)
        radius_params = (a3, a4, b3, c)

        # ------------------ 2) Generate "clean pre" structure ------------------
        clean_pre, inside_mask_flat, coords_sub = generate_structure(
            grid_size, sub_factor, t_samples,
            centerline_params, radius_params,
            bumps=[]
        )

        # ------------------ 3) Determine number of bumps for this sample (K = 1 / 2 / 3) ------------------
        K = bump_assignment[i]
        assert K * min_bump_frac <= 1.0, f"K={K}, min_bump_frac={min_bump_frac} ⇒ Invalid"

        # ------------------ 4) Decide between "expansion (add)" or "shrinkage (remove)" ------------------
        mode = 'add' if (np.random.rand() < expand_ratio) else 'remove'

        # -------------- 5) Compute total "physical volume" & sample delta_frac segmentally --------------
        voxel_size   = 2.0 / grid_size
        voxel_count  = np.sum(clean_pre)            # Sum of proportions inside coarse voxels
        voxel_volume = voxel_size ** 3
        true_volume  = voxel_count * voxel_volume   # Estimated physical volume

        # Choose between high and low delta_frac segment
        if np.random.rand() < delta_frac_ratio_high:
            # High segment [split, high_frac_range[1]]
            delta_frac = np.random.uniform(high_frac_range[0], high_frac_range[1])
        else:
            # Low segment [low_frac_range[0], split]
            delta_frac = np.random.uniform(low_frac_range[0], low_frac_range[1])

        target_volume = delta_frac * true_volume

        # -------------- 6) Use Dirichlet([α] * K) to distribute volume among K bumps --------------
        alpha_vec = np.ones(K, dtype=float) * bump_alpha
        while True:
            weights = np.random.dirichlet(alpha_vec)
            if (weights >= min_bump_frac).all():
                break
        small_volumes = weights * target_volume  # Physical volume per bump

        # -------------- 7) Generate bump parameter list for K bumps --------------
        bumps = []
        for vol_i in small_volumes:
            bump_center, r, _ = generate_bump_full_sphere(
                gt_volume_coarse=clean_pre,
                coords_sub=coords_sub,
                target_volume=vol_i,
                grid_size=grid_size,
                sub_factor=sub_factor,
                mode=mode,
                epsilon=1e-3
            )
            bumps.append((bump_center, r, mode))

        # ------------------ 8) Generate "clean post" structure ------------------
        clean_post, _, _ = generate_structure(
            grid_size, sub_factor, t_samples,
            centerline_params, radius_params,
            bumps=bumps
        )

        # --------------- 9) Compute and save label = clean_post - clean_pre ---------------
        label = clean_post - clean_pre
        label_name = f"sample_{i:03d}_label.nii.gz"
        nib.save(nib.Nifti1Image(label, np.eye(4)),
                 os.path.join(output_dir, label_name))

        # ---------------- 10) Save clean_pre & clean_post ----------------
        clean_pre_name  = f"sample_{i:03d}_clean_pre.nii.gz"
        clean_post_name = f"sample_{i:03d}_clean_post.nii.gz"
        nib.save(nib.Nifti1Image(clean_pre,  np.eye(4)),
                 os.path.join(output_dir, clean_pre_name))
        nib.save(nib.Nifti1Image(clean_post, np.eye(4)),
                 os.path.join(output_dir, clean_post_name))

        # ----------- 11) Generate and save "noisy A" version (snr=snrA) -----------
        noisyA_pre  = add_gaussian_noise(clean_pre,  snr=snrA)
        noisyA_post = add_gaussian_noise(clean_post, snr=snrA)
        noisyA_pre_name  = f"sample_{i:03d}_noisyA_pre_SNR{snrA:.0f}.nii.gz"
        noisyA_post_name = f"sample_{i:03d}_noisyA_post_SNR{snrA:.0f}.nii.gz"
        nib.save(nib.Nifti1Image(noisyA_pre,  np.eye(4)),
                 os.path.join(output_dir, noisyA_pre_name))
        nib.save(nib.Nifti1Image(noisyA_post, np.eye(4)),
                 os.path.join(output_dir, noisyA_post_name))

        # ----------- 12) If training set, generate and save "noisy B" version (snr=snrB) -----------
        if is_training:
            noisyB_pre  = add_gaussian_noise(clean_pre,  snr=snrB)
            noisyB_post = add_gaussian_noise(clean_post, snr=snrB)
            noisyB_pre_name  = f"sample_{i:03d}_noisyB_pre_SNR{snrB:.0f}.nii.gz"
            noisyB_post_name = f"sample_{i:03d}_noisyB_post_SNR{snrB:.0f}.nii.gz"
            nib.save(nib.Nifti1Image(noisyB_pre,  np.eye(4)),
                     os.path.join(output_dir, noisyB_pre_name))
            nib.save(nib.Nifti1Image(noisyB_post, np.eye(4)),
                     os.path.join(output_dir, noisyB_post_name))

    # Compute and print total time and average time per sample
    elapsed = time.time() - start_time
    avg_time = elapsed / n_clean
    print(f"\n[{desc}] Generated {n_clean} samples in {elapsed:.1f}s, average {avg_time:.2f}s per sample.")


# ======================================================
# 5) Wrapper function to generate training + validation sets (specify quantity directly)
# ======================================================
def generate_train_and_val(
    train_n_clean,
    val_n_clean,
    base_output_dir,
    snrA=80,
    snrB=150,
    bump_ratios=[0.5, 0.3, 0.2],
    expand_ratio=0.7,
    bump_alpha=1.0,
    min_bump_frac=0.05,
    grid_size=96,
    sub_factor=6,
    t_samples=300,
    # Newly added segmented delta_frac parameters
    delta_frac_split=0.035,      # Split point e.g. 3.5% = 0.035
    delta_frac_ratio_high=0.3,   # Sampling ratio for the high segment [0.035, 0.05]
    low_frac_range=(0.02, 0.035), # Range for the low segment
    high_frac_range=(0.035, 0.05) # Range for the high segment
):
    """
    Generate both training and validation sets at once, saving to base_output_dir/train/ and base_output_dir/val/.

    Training set (is_training=True):
      - clean_pre, clean_post, label
      - noisyA_pre, noisyA_post
      - noisyB_pre, noisyB_post

    Validation set (is_training=False):
      - clean_pre, clean_post, label
      - noisyA_pre, noisyA_post

    New feature:
      - delta_frac is divided into low (low_frac_range) and high (high_frac_range) segments,
        sampled with probability delta_frac_ratio_high from the high segment, otherwise from the low segment.
    """
    train_dir = os.path.join(base_output_dir, "train")
    val_dir   = os.path.join(base_output_dir, "val")

    print("Start generating training set:")
    generate_dataset(
        n_clean=train_n_clean,
        snrA=snrA,
        snrB=snrB,
        bump_ratios=bump_ratios,
        expand_ratio=expand_ratio,
        bump_alpha=bump_alpha,
        min_bump_frac=min_bump_frac,
        grid_size=grid_size,
        sub_factor=sub_factor,
        t_samples=t_samples,
        delta_frac_split=delta_frac_split,
        delta_frac_ratio_high=delta_frac_ratio_high,
        low_frac_range=low_frac_range,
        high_frac_range=high_frac_range,
        output_dir=train_dir,
        is_training=True
    )

    print("\nStart generating validation set:")
    generate_dataset(
        n_clean=val_n_clean,
        snrA=snrA,
        snrB=None,
        bump_ratios=bump_ratios,
        expand_ratio=expand_ratio,
        bump_alpha=bump_alpha,
        min_bump_frac=min_bump_frac,
        grid_size=grid_size,
        sub_factor=sub_factor,
        t_samples=t_samples,
        delta_frac_split=delta_frac_split,
        delta_frac_ratio_high=delta_frac_ratio_high,
        low_frac_range=low_frac_range,
        high_frac_range=high_frac_range,
        output_dir=val_dir,
        is_training=False
    )

    print(f"\nTraining set saved to: {train_dir}\nValidation set saved to: {val_dir}")


# ============================
# 6) Run example (modify as needed)
# ============================
if __name__ == "__main__":
    # 1) Specify the number of clean samples in the training/validation sets
    train_n_clean = 400  # Number of clean samples in training set
    val_n_clean   = 100   # Number of clean samples in validation set

    # 2) Call the wrapper function to start generation
    generate_train_and_val(
        train_n_clean=train_n_clean,
        val_n_clean=val_n_clean,
        base_output_dir="output_dataset",
        snrA=300,                    # Initial SNR A
        snrB=200,                   # Enhanced SNR B (for training only)
        bump_ratios=[0.5, 0.3, 0.2],# Strict ratio: 50% samples with 1 bump, 30% with 2 bumps, 20% with 3 bumps
        expand_ratio=0.7,           # 70% samples with expansion (add), 30% with shrinkage (remove)
        bump_alpha=1.0,             # Dirichlet concentration parameter α=1.0 (uniform distribution)
        min_bump_frac=0.05,         # Each bump must account for at least 5% of total volume change
        grid_size=96,
        sub_factor=6,
        t_samples=300,
        # The following two control delta_frac segment ratio between [0.02, 0.035] and [0.035, 0.05]
        delta_frac_split=0.035,       # Split point 3.5%
        delta_frac_ratio_high=0.4,    # 40% samples from [0.035, 0.05], the rest from [0.02, 0.035]
        low_frac_range=(0.02, 0.035),  # Low segment range 2% ~ 3.5%
        high_frac_range=(0.035, 0.05)  # High segment range 3.5% ~ 5%
    )
