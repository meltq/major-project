import os, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# -----------------------
# Config / paths / seed
# -----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

NORMAL_DIR   = "the-npy-files-from-hea-and-dat/npy_hr"
ABNORMAL_DIR = "anomolous-npy-files/abnormal_npy_hr"

BATCH_SIZE = 16
NUM_EPOCHS = 50
LR = 1e-3
HIDDEN_DIM = 128
NUM_LEADS = 12
TRAIN_FIRST_N = 7000  # keep first 7000 files for training (deterministic)

# -----------------------
# Robust dataset (recursive + per-signal zscore)
# -----------------------
class RobustPlainECGDataset(Dataset):
    def __init__(self, root_dir, suffix_filter="_hr.npy", limit_first_n=None, recursive=True):
        root = Path(root_dir)
        if not root.exists():
            raise ValueError(f"Path does not exist: {root_dir}")
        files = list(root.rglob("*.npy")) if recursive else list(root.glob("*.npy"))
        if suffix_filter:
            files = [f for f in files if f.name.lower().endswith(suffix_filter.lower())]
        files = sorted(files)
        if limit_first_n is not None:
            files = files[:limit_first_n]
        if len(files) == 0:
            raise ValueError(f"No .npy files found in {root_dir} (checked recursive={recursive}, suffix='{suffix_filter}')")
        self.npy_files = files
    def __len__(self): return len(self.npy_files)
    def __getitem__(self, idx):
        sig = np.load(self.npy_files[idx]).astype(np.float32)   # [timesteps, leads]
        # per-signal z-score normalization (per lead)
        mean = sig.mean(axis=0, keepdims=True)
        std  = sig.std(axis=0, keepdims=True) + 1e-6
        sig = (sig - mean) / std
        return torch.from_numpy(sig)  # [timesteps, leads]

# -----------------------
# Model: simple conv autoencoder
# -----------------------
class AE1D(nn.Module):
    def __init__(self, num_leads=12, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(num_leads, hidden_dim, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, num_leads, kernel_size=7, stride=2, padding=3, output_padding=1)
        )
    def forward(self, x):   # x: [batch, channels, seq]
        return self.decoder(self.encoder(x))

# -----------------------
# Prepare datasets & loaders (deterministic first-7000 -> train)
# -----------------------
print("Discovering files...")
normal_ds_all = RobustPlainECGDataset(NORMAL_DIR, suffix_filter="_hr.npy", recursive=True)
abnormal_ds   = RobustPlainECGDataset(ABNORMAL_DIR, suffix_filter="_hr.npy", recursive=True)

total_normal = len(normal_ds_all)
print(f"Total normal samples found: {total_normal}")
if total_normal < TRAIN_FIRST_N:
    raise ValueError(f"Not enough normal samples ({total_normal}) for TRAIN_FIRST_N={TRAIN_FIRST_N}")

# deterministic: use first TRAIN_FIRST_N files for training, remaining for test
train_indices = list(range(0, TRAIN_FIRST_N))
test_indices  = list(range(TRAIN_FIRST_N, total_normal))

train_ds = Subset(normal_ds_all, train_indices)
test_ds  = Subset(normal_ds_all, test_indices)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
abnorm_loader = DataLoader(abnormal_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train size: {len(train_ds)}  Test(normal) size: {len(test_ds)}  Abnormal size: {len(abnormal_ds)}")
print("Device:", DEVICE)

# -----------------------
# Instantiate model, loss, optimizer
# -----------------------
model = AE1D(num_leads=NUM_LEADS, hidden_dim=HIDDEN_DIM).to(DEVICE)
criterion = nn.MSELoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------
# Training loop
# -----------------------
best_val_loss = float("inf")
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    n = 0
    for sig in tqdm(train_loader, desc=f"Epoch {epoch} train"):
        # sig: [batch, timesteps, leads] -> permute
        sig = sig.permute(0,2,1).to(DEVICE)  # [batch, channels, seq]
        optimizer.zero_grad()
        out = model(sig)
        loss = criterion(out, sig)
        loss.backward()
        optimizer.step()
        bs = sig.size(0)
        running_loss += loss.item() * bs
        n += bs
    train_loss = running_loss / max(1,n)

    # quick validation on test(normal) set to monitor (not used for early stopping)
    model.eval()
    val_running = 0.0
    vn = 0
    with torch.no_grad():
        for vsig in test_loader:
            vsig = vsig.permute(0,2,1).to(DEVICE)
            vout = model(vsig)
            l = criterion(vout, vsig)
            val_running += l.item() * vsig.size(0)
            vn += vsig.size(0)
    val_loss = val_running / max(1,vn)
    print(f"Epoch {epoch}/{NUM_EPOCHS}  train_loss={train_loss:.6f}  val_loss(normal_test)={val_loss:.6f}")

    # checkpoint best by val_loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "AE1D_best.pth")
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"AE1D_epoch{epoch}.pth")

# -----------------------
# Evaluation: compute per-sample MAE scores on normal test set and abnormal set
# -----------------------
def compute_mae_loader(loader):
    model.eval()
    scores = []
    with torch.no_grad():
        for sig in loader:
            sig = sig.permute(0,2,1).to(DEVICE)
            out = model(sig)
            mae_per_sample = torch.mean(torch.abs(out - sig), dim=[1,2])  # [batch]
            scores.extend(mae_per_sample.cpu().numpy().tolist())
    return np.array(scores)

normal_test_scores = compute_mae_loader(test_loader)
abnormal_scores    = compute_mae_loader(abnorm_loader)

print("Normal test mean MAE:", float(normal_test_scores.mean()), "std:", float(normal_test_scores.std()))
print("Abnormal mean MAE:", float(abnormal_scores.mean()), "std:", float(abnormal_scores.std()))

# ROC AUC (label normal=0, abnormal=1)
y_scores = np.concatenate([normal_test_scores, abnormal_scores])
y_true   = np.concatenate([np.zeros_like(normal_test_scores), np.ones_like(abnormal_scores)])
auc = roc_auc_score(y_true, y_scores)
print("ROC AUC (reconstruction MAE):", float(auc))

# -----------------------
# Save scores CSV
# -----------------------
df = pd.DataFrame({
    "score": np.concatenate([normal_test_scores, abnormal_scores]),
    "label": np.concatenate([np.zeros_like(normal_test_scores), np.ones_like(abnormal_scores)])
})
df.to_csv("ae_reconstruction_scores.csv", index=False)
print("Saved ae_reconstruction_scores.csv (rows:", len(df), ")")

# ---- SAVE EVERYTHING FOR LATER ----
torch.save(model.state_dict(), "autoencoder_final.pth")
np.save("normal_scores.npy", normal_test_scores)
np.save("abnormal_scores.npy", abnormal_scores)

print("Saved: autoencoder_final.pth, normal_scores.npy, abnormal_scores.npy")

# ---- LOAD MODEL ----
model = AE1D(num_leads=12, hidden_dim=128).to(DEVICE)
model.load_state_dict(torch.load("autoencoder_final.pth", map_location=DEVICE))
model.eval()

# ---- LOAD PRECOMPUTED SCORES ----
normal_test_scores = np.load("normal_scores.npy")
abnormal_scores = np.load("abnormal_scores.npy")

print("Model + scores successfully loaded!")

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
sns.kdeplot(normal_test_scores, label="Normal", fill=True)
sns.kdeplot(abnormal_scores, label="Abnormal", fill=True)
plt.xlabel("Reconstruction MAE")
plt.title("KDE Plot: Separation Between Normal & Abnormal")
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
plt.hist(normal_test_scores, bins=50, alpha=0.6, label="Normal", density=True)
plt.hist(abnormal_scores, bins=50, alpha=0.6, label="Abnormal", density=True)
plt.xlabel("Reconstruction MAE")
plt.ylabel("Density")
plt.title("Score Distribution: Normal vs Abnormal ECGs")
plt.legend()
plt.show()

plt.figure(figsize=(8,5))
plt.boxplot([normal_test_scores, abnormal_scores], labels=["Normal", "Abnormal"])
plt.ylabel("Reconstruction MAE")
plt.title("Boxplot: Normal vs Abnormal Reconstruction Error")
plt.show()
