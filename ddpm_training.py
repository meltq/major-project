import torch
import numpy as np
import glob
from tqdm.notebook import tqdm
import os

def get_global_stats(npy_dir, limit=7000):
    files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))[:limit]
    print(f"Calculating global stats on {len(files)} training files...")
    
    total_sum = 0
    total_sq_sum = 0
    total_count = 0
    
    for f in tqdm(files):
        arr = np.load(f).flatten()
        total_sum += arr.sum()
        total_sq_sum += (arr**2).sum()
        total_count += len(arr)
        
    mean = total_sum / total_count
    var = (total_sq_sum / total_count) - (mean**2)
    std = np.sqrt(var)
    
    print(f"Global Mean: {mean:.5f}, Global Std: {std:.5f}")
    return mean, std

train_dir = "non-anomalous-ecg/npy_hr"
GLOBAL_MEAN, GLOBAL_STD = get_global_stats(train_dir, limit=7000)

def load_data_global_norm(npy_dir, mean, std, mode='train', split_idx=7000):
    files = sorted(glob.glob(f"{npy_dir}/*.npy"))
    
    if mode == 'train': files = files[:split_idx]
    elif mode == 'val': files = files[split_idx:]
    
    print(f"Loading {len(files)} files ({mode})...")
    data_list = []
    
    for f in tqdm(files):
        arr = np.load(f)
        arr = arr[:4992, :]
        arr = arr.transpose(1, 0)

        arr = (arr - mean) / std
        arr = np.clip(arr, -3, 3) / 3.0
        data_list.append(arr)
        
    return torch.tensor(np.array(data_list), dtype=torch.float32).to("cuda")

train_data_gpu = load_data_global_norm(train_dir, GLOBAL_MEAN, GLOBAL_STD, mode='train')
val_data_gpu = load_data_global_norm(train_dir, GLOBAL_MEAN, GLOBAL_STD, mode='val')

from diffusers import UNet1DModel, DDPMScheduler
from torch.optim import AdamW
import torch.nn.functional as F
import torch.fft
from tqdm.notebook import tqdm
import torch

model = UNet1DModel(
    sample_size=4992,
    in_channels=12,
    out_channels=12,
    layers_per_block=2,
    # 5 Levels of depth: 64 -> 128 -> 256 -> 512 -> 1024
    block_out_channels=(64, 128, 256, 512, 1024), 
    down_block_types=(
        "DownBlock1D", 
        "DownBlock1D", 
        "DownBlock1D",
        "AttnDownBlock1D",
        "DownBlock1D"
    ),
    up_block_types=(
        "UpBlock1D", 
        "AttnUpBlock1D", 
        "UpBlock1D", 
        "UpBlock1D",
        "UpBlock1D"
    )
).cuda()

noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000, 
    beta_schedule="squaredcos_cap_v2",
    prediction_type="sample" 
)

optimizer = AdamW(model.parameters(), lr=1e-4)

def geometric_loss(pred, target):
    mse_loss = F.mse_loss(pred, target)

    pred_fft = torch.fft.rfft(pred, dim=2).abs()
    target_fft = torch.fft.rfft(target, dim=2).abs()
    fft_loss = F.l1_loss(torch.log(pred_fft + 1e-8), torch.log(target_fft + 1e-8))
    
    return mse_loss + 0.1 * fft_loss

epochs = 40
batch_size = 16 

print(f"Retraining DEEP (5-Level) Model with x0-Prediction...")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    indices = torch.randperm(train_data_gpu.shape[0], device="cuda")
    pbar = tqdm(range(0, train_data_gpu.shape[0], batch_size), desc=f"Epoch {epoch+1}/{epochs}", leave=False)
    
    for i in pbar:
        batch_indices = indices[i : i + batch_size]
        clean_ecg = train_data_gpu[batch_indices]
        
        noise = torch.randn_like(clean_ecg)
        timesteps = torch.randint(0, 1000, (clean_ecg.shape[0],), device="cuda").long()
        noisy_ecg = noise_scheduler.add_noise(clean_ecg, noise, timesteps)
        
        pred_clean = model(noisy_ecg, timesteps).sample
        
        loss = geometric_loss(pred_clean, clean_ecg)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * clean_ecg.shape[0]
        pbar.set_postfix({'loss': loss.item()})
        
    avg_loss = total_loss / train_data_gpu.shape[0]
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i in range(0, len(val_data_gpu), batch_size):
            clean = val_data_gpu[i:i+batch_size]
            noise = torch.randn_like(clean)
            t = torch.randint(0, 1000, (clean.shape[0],), device="cuda").long()
            noisy = noise_scheduler.add_noise(clean, noise, t)
            
            pred = model(noisy, t).sample
            val_loss += geometric_loss(pred, clean).item() * clean.shape[0]
            
    avg_val_loss = val_loss / len(val_data_gpu)
    
    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

model.save_pretrained("geometry_deep_ecg_model")
print("Training Complete.")
