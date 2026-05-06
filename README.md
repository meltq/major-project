# DDPM for ECG Anomaly Detection

## Authors
Tejas Vipin, Shreya R, Tarun Akash, Dr. S. Prabakeran  
SRM Institute of Science and Technology, Chennai

## Abstract
This work uses Denoising Diffusion Probabilistic Models (DDPMs) trained on normal ECG signals to detect anomalies. By combining reconstruction error and likelihood estimation, the model enables accurate and scalable unsupervised cardiac anomaly detection.

## Method
- Dataset: PTB-XL (12-lead ECG)
- Model: 1D U-Net DDPM
- Training: Only normal ECG signals
- Detection: Reconstruction error + likelihood

## Variants
- Full Denoising  
- Inpainting  
- Partial Denoising (best balance)  
- NLL-based (best accuracy, slower)

## Results (ROC-AUC)
- Autoencoder: 0.6766  
- Masked AE: 0.6849  
- OC-SVM: 0.7581  
- **DDPM: 0.7682**

## Conclusion
DDPM outperforms traditional methods by learning normal ECG distributions. Partial denoising offers the best trade-off between speed and performance.

## Future Work
- Multi-dataset validation  
- Faster inference  
- Deployment on low-resource systems
