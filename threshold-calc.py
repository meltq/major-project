import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from tqdm import tqdm
from diffusers import UNet1DModel, DDIMScheduler

MODEL_PATH = "deep-ddpm-tiny/other/default/2/geometry_deep_ecg_model"
NORMAL_DATA_DIR = "non-anomalous-ecg/npy_hr"

T_START = 400
TRAIN_SPLIT_INDEX = 7000 
BATCH_SIZE = 64
T_START = 400
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- 2. GET GLOBAL STATS (From Training Split) ---
def get_global_stats(npy_dir, limit=7000):
    files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))[:limit]
    print(f"Calculating global stats from {len(files)} training files...")
    
    total_sum, total_sq_sum, total_count = 0, 0, 0
    for f in tqdm(files, desc="Stats", leave=False):
        arr = np.load(f).flatten()
        total_sum += arr.sum()
        total_sq_sum += (arr**2).sum()
        total_count += len(arr)
        
    mean = total_sum / total_count
    std = np.sqrt((total_sq_sum / total_count) - (mean**2))
    print(f"Global Mean: {mean:.5f}, Global Std: {std:.5f}")
    return mean, std

GLOBAL_MEAN, GLOBAL_STD = get_global_stats(NORMAL_DATA_DIR, limit=TRAIN_SPLIT_INDEX)

print(f"loading Model from {MODEL_PATH}...")
model = UNet1DModel.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_schedule="squaredcos_cap_v2",
    prediction_type="sample"
)
scheduler.set_timesteps(num_inference_steps=50)

import seaborn as sns

def get_validation_files(npy_dir, start_idx):
    files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))[start_idx:]
    print(f"Found {len(files)} files for Threshold Calibration.")
    return files

def process_batch(file_batch, mean, std):
    """Loads a batch of files, normalizes, and moves to GPU."""
    data_list = []
    for f in file_batch:
        arr = np.load(f)[:4992, :].transpose(1, 0)
        # Global Z-Score Normalization + Clipping
        arr = (arr - mean) / std
        arr = np.clip(arr, -3, 3) / 3.0
        data_list.append(arr)
        
    return torch.tensor(np.array(data_list), dtype=torch.float32).to(DEVICE)

def calculate_mae_for_batch(data_batch, model, scheduler, t_start):
    """Runs the diffusion manifold projection and returns MAE scores."""
    noise = torch.randn_like(data_batch)
    t = torch.tensor([t_start], device=DEVICE).long()
    noisy_latents = scheduler.add_noise(data_batch, noise, t)
    
    target_timesteps = [ts for ts in scheduler.timesteps if ts <= t_start]
    curr_sample = noisy_latents
    
    with torch.no_grad():
        for t_step in target_timesteps:
            model_output = model(curr_sample, t_step).sample
            curr_sample = scheduler.step(model_output, t_step, curr_sample).prev_sample
            
    # Calculate MAE (Absolute difference between original and reconstructed)
    scores = torch.abs(data_batch - curr_sample).mean(dim=(1, 2))
    return scores.cpu().numpy()


# --- 5. RUN CALIBRATION LOOP ---
val_files = get_validation_files(NORMAL_DATA_DIR, TRAIN_SPLIT_INDEX)
all_healthy_scores = []

print(f"Running Inference on {len(val_files)} Healthy Validation Samples...")
# Loop through files in chunks of BATCH_SIZE
for i in tqdm(range(0, len(val_files), BATCH_SIZE), desc="Processing Batches"):
    file_batch = val_files[i : i + BATCH_SIZE]
    
    # 1. Load and prepare data
    data_tensor = process_batch(file_batch, GLOBAL_MEAN, GLOBAL_STD)
    
    # 2. Run inference and get scores
    batch_scores = calculate_mae_for_batch(data_tensor, model, scheduler, T_START)
    
    # 3. Store scores
    all_healthy_scores.extend(batch_scores)

all_healthy_scores = np.array(all_healthy_scores)


# --- 6. CALCULATE FINAL THRESHOLD ---
calib_mean = all_healthy_scores.mean()
calib_std = all_healthy_scores.std()

# 99.7% Confidence Interval Cutoff (Standard in anomaly detection)
DYNAMIC_THRESHOLD = calib_mean + (3 * calib_std)

print("\n" + "="*50)
print("="*50)
print(f"Total Samples Processed : {len(all_healthy_scores)}")
print(f"Healthy MAE Mean        : {calib_mean:.6f}")
print(f"Healthy MAE Std Dev     : {calib_std:.6f}")
print("-" * 50)
print(f"RECOMMENDED THRESHOLD : {DYNAMIC_THRESHOLD:.6f}")
print("="*50)


# --- 7. VISUALIZE THE DISTRIBUTION ---
plt.figure(figsize=(10, 6))
sns.histplot(all_healthy_scores, bins=50, kde=True, color='green', alpha=0.6)

# Plot the Mean and Threshold lines
plt.axvline(calib_mean, color='blue', linestyle='dashed', linewidth=2, label=f'Mean ({calib_mean:.4f})')
plt.axvline(DYNAMIC_THRESHOLD, color='red', linestyle='solid', linewidth=2.5, label=f'Threshold [Mean + 3σ] ({DYNAMIC_THRESHOLD:.4f})')

plt.title("Distribution of MAE Scores on Healthy Validation Data")
plt.xlabel("Reconstruction Error (MAE)")
plt.ylabel("Number of Patients")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
