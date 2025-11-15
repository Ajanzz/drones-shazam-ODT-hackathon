!pip install torch torchvision torchaudio librosa onnx onnxruntime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import librosa
from pathlib import Path
from typing import List, Tuple
import random

# ===== CONFIG =====
DRONE_DIR = Path("/content/drone_training")
NON_DRONE_DIR = Path("/content/non_drone_training")

TARGET_SR = 16000
MAX_SECONDS = 3
N_MELS = 64
BATCH_SIZE = 16
EPOCHS = 20
VAL_SPLIT = 0.2
ONNX_OUTPUT_PATH = "model_detector.onnx"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


def list_wavs(folder: Path) -> List[Path]:
    return [p for p in folder.glob("**/*") if p.suffix.lower() == ".wav"]


class DroneDataset(Dataset):
    def __init__(self, files_labels: List[Tuple[Path, int]]):
        self.files_labels = files_labels

    def __len__(self):
        return len(self.files_labels)

    def __getitem__(self, idx):
        path, label = self.files_labels[idx]

        # 1) load audio
        y, sr = librosa.load(path, sr=TARGET_SR, mono=True)

        # 2) trim/pad to MAX_SECONDS
        max_samples = TARGET_SR * MAX_SECONDS
        if len(y) > max_samples:
            y = y[:max_samples]
        else:
            y = np.pad(y, (0, max_samples - len(y)), mode="constant")

        # 3) mel spectrogram -> log-mel
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=TARGET_SR,
            n_mels=N_MELS,
            n_fft=1024,
            hop_length=512,
        )
        logmel = librosa.power_to_db(mel).astype(np.float32)  # (64, time)

        # 4) tensor with shape (1, 1, 64, T) for Conv2d
        x = torch.from_numpy(logmel).unsqueeze(0).unsqueeze(0)  # (1,1,64,T)

        return x, torch.tensor(label, dtype=torch.long)


class DroneCNN(nn.Module):
    """
    Simple 2D CNN over log-mel 'image' of shape (1, 64, T).
    Uses AdaptiveAvgPool2d so it works with variable time dimension.
    """
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),          # (16, 32, T/2)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),          # (32, 16, T/4)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # (32,1,1)
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):
        # x: (B,1,64,T)
        x = self.conv_block(x)
        x = self.global_pool(x)  # (B,32,1,1)
        x = x.view(x.size(0), -1)  # (B,32)
        logits = self.fc(x)        # (B,2)
        return logits


def make_datasets(drone_dir: Path, non_drone_dir: Path):
    drone_files = list_wavs(drone_dir)
    non_drone_files = list_wavs(non_drone_dir)

    if not drone_files:
        raise RuntimeError(f"No .wav files found in {drone_dir}")
    if not non_drone_files:
        raise RuntimeError(f"No .wav files found in {non_drone_dir}")

    print(f"Found {len(drone_files)} drone files")
    print(f"Found {len(non_drone_files)} non-drone files")

    data = []
    data += [(p, 1) for p in drone_files]      # 1 -> drone
    data += [(p, 0) for p in non_drone_files]  # 0 -> non-drone

    random.shuffle(data)

    n_total = len(data)
    n_val = int(n_total * VAL_SPLIT)
    n_train = n_total - n_val

    class _ListDataset(Dataset):
        def __init__(self, lst): self.lst = lst
        def __len__(self): return len(self.lst)
        def __getitem__(self, i): return self.lst[i]

    base = _ListDataset(data)
    train_raw, val_raw = random_split(
        base, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_list = [train_raw.dataset.lst[i] for i in train_raw.indices]
    val_list   = [val_raw.dataset.lst[i] for i in val_raw.indices]

    train_ds = DroneDataset(train_list)
    val_ds   = DroneDataset(val_list)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    return train_ds, val_ds


train_ds, val_ds = make_datasets(DRONE_DIR, NON_DRONE_DIR)

from torch.utils.data import DataLoader

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

model = DroneCNN(n_classes=2).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Model and data ready.")

# FIXED DroneDataset (overrides the old one)

class DroneDataset(Dataset):
    def __init__(self, files_labels: List[Tuple[Path, int]]):
        self.files_labels = files_labels

    def __len__(self):
        return len(self.files_labels)

    def __getitem__(self, idx):
        path, label = self.files_labels[idx]

        y, sr = librosa.load(path, sr=TARGET_SR, mono=True)

        max_samples = TARGET_SR * MAX_SECONDS
        if len(y) > max_samples:
            y = y[:max_samples]
        else:
            y = np.pad(y, (0, max_samples - len(y)), mode="constant")

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=TARGET_SR,
            n_mels=N_MELS,
            n_fft=1024,
            hop_length=512,
        )
        logmel = librosa.power_to_db(mel).astype(np.float32)  # (64, T)

        
        x = torch.from_numpy(logmel).unsqueeze(0)

        return x, torch.tensor(label, dtype=torch.long)

from torch.utils.data import DataLoader

train_ds, val_ds = make_datasets(DRONE_DIR, NON_DRONE_DIR)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = DroneCNN(n_classes=2).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Model and data ready.")
for x_batch, y_batch in train_loader:
    print("One batch shape:", x_batch.shape)  # should be (B, 1, 64, T)
    break

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


def eval_one_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return total_loss / total, correct / total


print("Starting training...")

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = eval_one_epoch(model, val_loader, criterion)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
        f"val_loss={val_loss:.4f} acc={val_acc:.3f}"
    )

import torch.onnx as onnx

model.eval()

# grab one batch to see the real input shape
example_batch = next(iter(train_loader))[0]   # shape: (B, 1, 64, T)
dummy_input = example_batch[0:1].to(DEVICE)  # shape: (1, 1, 64, T)

dynamic_axes = {
    "input": {0: "batch", 3: "time"},  # batch & time are dynamic
    "output": {0: "batch"}
}

ONNX_OUTPUT_PATH = "model_detector.onnx"

onnx.export(
    model,
    dummy_input,
    ONNX_OUTPUT_PATH,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes=dynamic_axes,
    opset_version=13,
)

print("Saved ONNX model to:", ONNX_OUTPUT_PATH)

from google.colab import files
files.download("model_detector.onnx")
