import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

# Device: GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class ECGDataset(Dataset):
    def __init__(self, npy_dir, mask_ratio=0.2, limit_first_n=None):
        npy_dir = Path(npy_dir).resolve()
        self.npy_files = sorted([f for f in npy_dir.glob("*.npy") if f.name.lower().endswith("_hr.npy")])

        # Restrict to first N samples if specified
        if limit_first_n is not None:
            self.npy_files = self.npy_files[:limit_first_n]

        self.mask_ratio = mask_ratio

        if len(self.npy_files) == 0:
            raise ValueError(f"No .npy files found in {npy_dir}")

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
        signal = np.load(self.npy_files[idx]).astype(np.float32)
        
        # Create random mask
        mask = np.ones_like(signal)
        num_masked = int(signal.shape[0] * self.mask_ratio)
        start = np.random.randint(0, signal.shape[0] - num_masked)
        mask[start:start+num_masked, :] = 0
        masked_signal = signal * mask

        return torch.from_numpy(masked_signal), torch.from_numpy(signal), torch.from_numpy(mask)


# Path to your Kaggle dataset
dataset_path = "the-npy-files-from-hea-and-dat/npy_hr"

# Hyperparameters
batch_size = 16
mask_ratio = 0.3
limit_first_n = 7000  # train + val + test only on first 7000 normal ECGs

# Create dataset
dataset = ECGDataset(dataset_path, mask_ratio=mask_ratio, limit_first_n=limit_first_n)

# Train/Val/Test split: 70/15/15
train_size = int(0.70 * len(dataset))
val_size   = int(0.15 * len(dataset))
test_size  = len(dataset) - train_size - val_size

train_set, val_set, test_set = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)  # deterministic split
)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False)

print(f"Total normal ECGs (restricted): {len(dataset)}")
print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

class MAE1D(nn.Module):
    def __init__(self, num_leads=12, hidden_dim=128):
        super(MAE1D, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(num_leads, hidden_dim, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, num_leads, kernel_size=7, stride=2, padding=3, output_padding=1)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

# Initialize model and move to GPU
model = MAE1D().to(device)
print(model)

# Loss and optimizer
criterion = nn.MSELoss(reduction='none')  # we’ll apply masking manually
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training settings
num_epochs = 100
batch_size = 16  # already set in DataLoader
mask_ratio = 0.3

for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0.0
    
    for masked_signal, signal, mask in tqdm(train_loader, desc=f"Epoch {epoch}"):
        masked_signal = masked_signal.permute(0, 2, 1).to(device)  # [batch, channels, seq]
        signal = signal.permute(0, 2, 1).to(device)
        mask = mask.permute(0, 2, 1).to(device)

        optimizer.zero_grad()
        output = model(masked_signal)
        
        # Masked MSE loss
        loss = (criterion(output, signal) * (1 - mask)).mean()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * masked_signal.size(0)
    
    epoch_loss /= len(train_set)
    print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.6f}")

# Save the trained MAE model
save_path = "mae1d_ecg.pth"
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")

from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# --- Paths ---
normal_path = Path("the-npy-files-from-hea-and-dat/npy_hr")
abnormal_path = Path("anomolous-npy-files/abnormal_npy_hr")

def compute_mae_for_folder(folder_path, model, device):
    npy_files = sorted(folder_path.glob("*.npy"))
    print(f"Found {len(npy_files)} samples in {folder_path}")

    maes = []

    for f in tqdm(npy_files, desc=f"Processing " + folder_path.name):
        x = np.load(f).astype(np.float32)   # shape [T, 12]

        # Convert to torch shape [1, 12, T]
        if x.ndim == 2:
            x_t = torch.tensor(x).float().permute(1, 0).unsqueeze(0).to(device)
        else:
            raise ValueError(f"Unexpected shape {x.shape} in file {f}")

        with torch.no_grad():
            out = model(x_t)

        mae = torch.mean(torch.abs(out - x_t)).item()
        maes.append(mae)

    return np.array(maes)


# RUN VALIDATION
normal_mae = compute_mae_for_folder(normal_path, model, device)
abnormal_mae = compute_mae_for_folder(abnormal_path, model, device)

print("\n--- VALIDATION RESULTS ---")
print(f"Normal MAE:   mean={normal_mae.mean():.6f}, std={normal_mae.std():.6f}")
print(f"Abnormal MAE: mean={abnormal_mae.mean():.6f}, std={abnormal_mae.std():.6f}")

from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
import numpy as np

# Combine scores and labels
scores = np.concatenate([normal_mae, abnormal_mae])
labels = np.concatenate([np.zeros_like(normal_mae), np.ones_like(abnormal_mae)])

# ROC curve
fpr, tpr, thresholds = roc_curve(labels, scores)
roc_auc = auc(fpr, tpr)

# Best threshold = Youden's J (tpr - fpr)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]

print(best_threshold)

# Use the threshold you found
threshold = best_threshold

# Predictions
normal_pred = normal_mae < threshold     # expected: True (normal)
abnormal_pred = abnormal_mae > threshold # expected: True (abnormal)

# Accuracy for each class
normal_acc = normal_pred.mean()
abnormal_acc = abnormal_pred.mean()

# Overall accuracy
overall_acc = (normal_pred.sum() + abnormal_pred.sum()) / (len(normal_mae) + len(abnormal_mae))

print(normal_acc, abnormal_acc, overall_acc)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))   # wider (12), taller (6)

sns.kdeplot(normal_mae, label="Normal ECG", linewidth=2)
sns.kdeplot(abnormal_mae, label="Abnormal ECG", linewidth=2)

plt.xlabel("Reconstruction MAE")
plt.ylabel("Density")
plt.title("MAE Density Distribution: Normal vs Abnormal")
plt.legend()
plt.grid(True)
plt.show()
