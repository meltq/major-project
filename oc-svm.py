import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def extract_features(ecg):
    # ecg shape: [T, 12]
    feats = []
    feats.append(ecg.mean(axis=0))
    feats.append(ecg.std(axis=0))
    feats.append(np.max(ecg, axis=0))
    feats.append(np.min(ecg, axis=0))
    feats.append(np.ptp(ecg, axis=0))          # peak-to-peak
    feats.append(np.mean(np.abs(ecg), axis=0)) # signal energy proxy
    return np.concatenate(feats)

normal_path = Path("the-npy-files-from-hea-and-dat/npy_hr")

X_normal = []
for f in tqdm(sorted(normal_path.glob("*.npy"))):
    ecg = np.load(f)
    X_normal.append(extract_features(ecg))

X_normal = np.array(X_normal)

# Use only first 7000 normal samples
X_normal_subset = X_normal[:7000]

# Train/Val/Test split: 70/15/15
n = len(X_normal_subset)
train_end = int(0.7 * n)
val_end = int(0.85 * n)

X_train = X_normal_subset[:train_end]
X_val_normal = X_normal_subset[train_end:val_end]
X_normal_test_subset = X_normal_subset[val_end:]

# Remaining 1933 normals for final testing
X_normal_test_remaining = X_normal[7000:]

abnormal_path = Path("anomolous-npy-files/abnormal_npy_hr")

X_abnormal_test = []
for f in tqdm(sorted(abnormal_path.glob("*.npy"))):
    ecg = np.load(f)
    X_abnormal_test.append(extract_features(ecg))

X_abnormal_test = np.array(X_abnormal_test)

# Take first 200 abnormal samples for validation
X_val_abnormal = X_abnormal_test[:200]

# Remaining for final testing
X_abnormal_test_final = X_abnormal_test

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val_normal = scaler.transform(X_val_normal)
X_normal_test_subset = scaler.transform(X_normal_test_subset)
X_normal_test_remaining = scaler.transform(X_normal_test_remaining)
X_val_abnormal = scaler.transform(X_val_abnormal)
X_abnormal_test_final = scaler.transform(X_abnormal_test_final)

gammas = [0.05, 0.08, 0.1, 0.12, 0.15]
nus = [0.05, 0.1, 0.15, 0.2, 0.25]

best_score = -1
best_params = None
best_svm = None

for gamma in gammas:
    for nu in nus:
        svm = OneClassSVM(kernel="rbf", gamma=gamma, nu=nu)
        svm.fit(X_train)
        
        y_pred_normal = svm.predict(X_val_normal)
        y_pred_abnormal = svm.predict(X_val_abnormal)
        
        normal_acc = (y_pred_normal == 1).mean()
        abnormal_acc = (y_pred_abnormal == -1).mean()
        
        score = np.sqrt(normal_acc * abnormal_acc)
        
        if score > best_score:
            best_score = score
            best_params = (gamma, nu)
            best_svm = svm

print(f"Best balanced score: {best_score:.3f} with gamma={best_params[0]}, nu={best_params[1]}")

# Combine normal test: 15% subset + remaining normals
X_normal_test_final = np.concatenate([X_normal_test_subset, X_normal_test_remaining], axis=0)

y_pred_normal = best_svm.predict(X_normal_test_final)
y_pred_abnormal = best_svm.predict(X_abnormal_test_final)

normal_acc = (y_pred_normal == 1).mean()
abnormal_acc = (y_pred_abnormal == -1).mean()
overall_acc = (
    (y_pred_normal == 1).sum() + (y_pred_abnormal == -1).sum()
) / (len(y_pred_normal) + len(y_pred_abnormal))

print(normal_acc, abnormal_acc, overall_acc)

# ROC
y_true = np.concatenate([
    np.zeros(len(X_normal_test_final)),
    np.ones(len(X_abnormal_test_final))
])
y_scores = np.concatenate([
    -best_svm.decision_function(X_normal_test_final),
    -best_svm.decision_function(X_abnormal_test_final)
])
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, linewidth=2, label=f"SVM ROC (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – One-Class SVM")
plt.legend()
plt.grid(True)
plt.show()
