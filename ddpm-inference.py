import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from tqdm import tqdm
from diffusers import UNet1DModel, DDIMScheduler

MODEL_PATH = "deep-ddpm-tiny/other/default/2/geometry_deep_ecg_model"
TRAIN_DIR_FOR_STATS = "non-anomalous-ecg/npy_hr"
NORMAL_VAL_DIR = "non-anomalous-ecg/npy_hr"
ABNORMAL_DIR = "anomalous-ecg-major/abnormal_npy_hr"

T_START = 400
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LIMIT_STATS = 1000

def get_global_stats(npy_dir, limit=7000):
    """Calculates Mean and Std from the training set."""
    files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))[:limit]
    
    total_sum = 0
    total_sq_sum = 0
    total_count = 0
    
    for f in tqdm(files, desc="Calc Stats"):
        arr = np.load(f).flatten()
        total_sum += arr.sum()
        total_sq_sum += (arr**2).sum()
        total_count += len(arr)
        
    mean = total_sum / total_count
    var = (total_sq_sum / total_count) - (mean**2)
    std = np.sqrt(var)
    
    print(f"Global Mean: {mean:.5f}, Global Std: {std:.5f}")
    return mean, std

def load_random_samples(npy_dir, mean, std, n=5):
    files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))
    
    # Pick random files
    selected_files = files[0:15]
    data_list = []
    
    print(f"loading samples from {npy_dir}")
    
    for f in selected_files:
        arr = np.load(f)
        
        arr = arr[:4992, :]
        arr = arr.transpose(1, 0)
        
        arr = (arr - mean) / std
        arr = np.clip(arr, -3, 3) / 3.0
        data_list.append(arr)
        
    return torch.tensor(np.array(data_list), dtype=torch.float32).to(DEVICE)


GLOBAL_MEAN, GLOBAL_STD = get_global_stats(TRAIN_DIR_FOR_STATS, limit=LIMIT_STATS)

norm_batch = load_random_samples(NORMAL_VAL_DIR, GLOBAL_MEAN, GLOBAL_STD, n=5)
abnorm_batch = load_random_samples(ABNORMAL_DIR, GLOBAL_MEAN, GLOBAL_STD, n=5)

print(f"loading Model from {MODEL_PATH}...")
model = UNet1DModel.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_schedule="squaredcos_cap_v2",
    prediction_type="sample"
)
scheduler.set_timesteps(num_inference_steps=50)

THRESHOLD = 0.189578

def analyze_heartbeats(data_batch, model, scheduler, t_start=400):
    noise = torch.randn_like(data_batch)
    t = torch.tensor([t_start], device=DEVICE).long()
    noisy_latents = scheduler.add_noise(data_batch, noise, t)
    
    target_timesteps = [ts for ts in scheduler.timesteps if ts <= t_start]
    
    curr_sample = noisy_latents
    with torch.no_grad():
        for t in target_timesteps:
            model_output = model(curr_sample, t).sample
            curr_sample = scheduler.step(model_output, t, curr_sample).prev_sample
            
    reconstructions = curr_sample
    scores = torch.abs(data_batch - reconstructions).mean(dim=(1, 2))
    
    return reconstructions, scores


print("\nAnalyzing Healthy Samples...")
recon_norm, scores_norm = analyze_heartbeats(norm_batch, model, scheduler, T_START)

print("Analyzing Abnormal Samples...")
recon_abnorm, scores_abnorm = analyze_heartbeats(abnorm_batch, model, scheduler, T_START)


print("\n" + "="*70)
print(f"{'SAMPLE TYPE':<15} | {'SCORE (MAE)':<12} | {'STATUS':<15} | {'VERDICT':<15}")
print("-" * 70)

# Report Healthy
for i, score in enumerate(scores_norm):
    verdict = "PASSED" if score < THRESHOLD else "FALSE ALARM"
    print(f"{'Healthy':<15} | {score.item():.4f}       | {'Normal':<15} | {verdict}")

print("-" * 70)

# Report Abnormal
for i, score in enumerate(scores_abnorm):
    verdict = "🚨 DETECTED" if score >= THRESHOLD else "❌ MISSED"
    print(f"{'Abnormal':<15} | {score.item():.4f}       | {'Anomaly':<15} | {verdict}")
print("="*70)


def plot_projection(original, reconstructed, score, title, color_code):
    plt.figure(figsize=(12, 5))
    
    # Convert to numpy (First channel only)
    orig = original[0, 0, :].cpu().numpy()
    recon = reconstructed[0, 0, :].cpu().numpy()
    
    plt.plot(orig, label="Original Patient Signal", color="black", alpha=0.7, linewidth=1.5)
    plt.plot(recon, label="Projected Healthy Reconstruction", color=color_code, linestyle="--", linewidth=2)
    plt.fill_between(range(len(orig)), orig, recon, color="red", alpha=0.3, label="Anomaly Error")
    
    plt.title(f"{title}\nAnomaly Score (MAE): {score:.4f}  |  Threshold: {THRESHOLD}")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Visualize
print("\nVisualizing Healthy Manifold Projection:")
plot_projection(norm_batch[0:1], recon_norm[0:1], scores_norm[0], "HEALTHY PATIENT CASE", "green")

print("\nVisualizing Abnormal Manifold Projection:")
plot_projection(abnorm_batch[0:1], recon_abnorm[0:1], scores_abnorm[0], "HEART ATTACK PATIENT CASE", "orange")
