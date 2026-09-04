"""
train_dl.py - PyTorch Deep Learning Architectures (1D CNN Raw & 2D CNN Spectrogram)
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from data_loader import prepare_cwru_dataset, LABEL_NAMES
from signal_processing import compute_spectrogram
from dataset_split import split_dataset_by_groups


# ---------------------------------------------------------
# 1. PyTorch Dataset Classes
# ---------------------------------------------------------
class RawVibrationDataset(Dataset):
    """Dataset for 1D Raw Vibration Windows"""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # Shape: (N, 1, 2048)
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SpectrogramDataset(Dataset):
    """Dataset for 2D STFT Spectrogram Images"""
    def __init__(self, X_raw: np.ndarray, y: np.ndarray, fs: float = 12000.0):
        specs = []
        for win in X_raw:
            _, _, Sxx_log = compute_spectrogram(win, fs=fs, nperseg=128, noverlap=64)
            # Normalize per spectrogram matrix
            norm_Sxx = (Sxx_log - np.mean(Sxx_log)) / (np.std(Sxx_log) + 1e-8)
            specs.append(norm_Sxx)
        
        # Shape: (N, 1, Freq_bins, Time_steps)
        self.X = torch.tensor(np.array(specs), dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------
# 2. Neural Network Architectures
# ---------------------------------------------------------
class Conv1DNet(nn.Module):
    """1D Convolutional Neural Network for Raw Vibration Signals"""
    def __init__(self, num_classes: int = 4):
        super(Conv1DNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(16, 32, kernel_size=8, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class Conv2DSpectrogramNet(nn.Module):
    """2D Convolutional Neural Network for Time-Frequency Spectrograms"""
    def __init__(self, num_classes: int = 4):
        super(Conv2DSpectrogramNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ---------------------------------------------------------
# 3. Training & Evaluation Pipeline
# ---------------------------------------------------------
def train_model(model, train_loader, val_loader, epochs: int = 15, lr: float = 0.001, device: str = "cpu"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.to(device)
    best_val_acc = 0.0
    best_weights = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)

        # Validation phase
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())

        val_acc = accuracy_score(val_targets, val_preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = model.state_dict()

    if best_weights:
        model.load_state_dict(best_weights)
    return model, best_val_acc


def evaluate_dl_models(models_dir: str = "models", epochs: int = 12):
    """
    Trains and compares 1D CNN (Raw Signal) vs 2D CNN (Spectrogram)
    """
    os.makedirs(models_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Deep Learning Training Hardware Device: {device}")

    X_raw, y, meta_df = prepare_cwru_dataset()
    X_train, y_train, X_val, y_val, X_test, y_test, _ = split_dataset_by_groups(
        X_raw, y, meta_df, test_size=0.2, val_size=0.15
    )

    # ---------------------------------------------------------
    # Train 1D CNN on Raw Signal
    # ---------------------------------------------------------
    print("Training 1D ConvNet on Raw Vibration Windows...")
    train_ds_1d = RawVibrationDataset(X_train, y_train)
    val_ds_1d = RawVibrationDataset(X_val, y_val)
    test_ds_1d = RawVibrationDataset(X_test, y_test)

    train_loader_1d = DataLoader(train_ds_1d, batch_size=32, shuffle=True)
    val_loader_1d = DataLoader(val_ds_1d, batch_size=32, shuffle=False)
    test_loader_1d = DataLoader(test_ds_1d, batch_size=32, shuffle=False)

    net_1d = Conv1DNet(num_classes=4)
    net_1d, _ = train_model(net_1d, train_loader_1d, val_loader_1d, epochs=epochs, device=device)

    # Evaluate 1D CNN
    net_1d.eval()
    y_pred_1d = []
    with torch.no_grad():
        for X_batch, _ in test_loader_1d:
            X_batch = X_batch.to(device)
            out = net_1d(X_batch)
            y_pred_1d.extend(torch.argmax(out, dim=1).cpu().numpy())
    
    acc_1d = accuracy_score(y_test, y_pred_1d)
    _, _, f1_1d, _ = precision_recall_fscore_support(y_test, y_pred_1d, average="macro")
    print(f"1D CNN (Raw Signal) -> Test Accuracy: {acc_1d:.4f} | Test F1 Macro: {f1_1d:.4f}")

    # ---------------------------------------------------------
    # Train 2D CNN on Spectrogram Images
    # ---------------------------------------------------------
    print("Training 2D ConvNet on STFT Log-Spectrogram Images...")
    train_ds_2d = SpectrogramDataset(X_train, y_train)
    val_ds_2d = SpectrogramDataset(X_val, y_val)
    test_ds_2d = SpectrogramDataset(X_test, y_test)

    train_loader_2d = DataLoader(train_ds_2d, batch_size=32, shuffle=True)
    val_loader_2d = DataLoader(val_ds_2d, batch_size=32, shuffle=False)
    test_loader_2d = DataLoader(test_ds_2d, batch_size=32, shuffle=False)

    net_2d = Conv2DSpectrogramNet(num_classes=4)
    net_2d, _ = train_model(net_2d, train_loader_2d, val_loader_2d, epochs=epochs, device=device)

    # Evaluate 2D CNN
    net_2d.eval()
    y_pred_2d = []
    with torch.no_grad():
        for X_batch, _ in test_loader_2d:
            X_batch = X_batch.to(device)
            out = net_2d(X_batch)
            y_pred_2d.extend(torch.argmax(out, dim=1).cpu().numpy())
    
    acc_2d = accuracy_score(y_test, y_pred_2d)
    _, _, f1_2d, _ = precision_recall_fscore_support(y_test, y_pred_2d, average="macro")
    print(f"2D CNN (Spectrogram) -> Test Accuracy: {acc_2d:.4f} | Test F1 Macro: {f1_2d:.4f}")

    # Save best PyTorch DL model
    if f1_2d >= f1_1d:
        torch.save(net_2d.state_dict(), os.path.join(models_dir, "best_dl_model.pt"))
        best_dl_name = "2D ConvNet (Spectrogram)"
    else:
        torch.save(net_1d.state_dict(), os.path.join(models_dir, "best_dl_model.pt"))
        best_dl_name = "1D ConvNet (Raw Signal)"

    print(f"Saved best DL model weights ({best_dl_name}) to {models_dir}/best_dl_model.pt")
    return {
        "1D_CNN": {"accuracy": float(acc_1d), "f1_macro": float(f1_1d)},
        "2D_CNN_Spectrogram": {"accuracy": float(acc_2d), "f1_macro": float(f1_2d)},
        "best_dl_model": best_dl_name
    }


if __name__ == "__main__":
    evaluate_dl_models()
