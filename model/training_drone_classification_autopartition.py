import os
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchaudio

# ----------------------------
# 1. CONFIG
# ----------------------------

ROOT_DIR   = "/content"  # change if your data is somewhere else
DATA_DIR   = os.path.join(ROOT_DIR, "train_data")  # only this folder exists

SAMPLE_RATE   = 16000    # target sample rate (Hz)
DURATION_SEC  = 3        # clip length in seconds (pad/trim)
NUM_SAMPLES   = SAMPLE_RATE * DURATION_SEC

N_MELS        = 64
BATCH_SIZE    = 32
NUM_EPOCHS    = 15
LEARNING_RATE = 1e-3
TEST_RATIO    = 0.2      # 20% test split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ----------------------------
# 2. DATASET
# ----------------------------

class AudioFolderDataset(Dataset):
    """
    Expects structure:
      train_data/
        drone0/
          *.wav
        drone1/
          *.wav
        ...
        drone9/
          *.wav
    """

    def __init__(self, root_dir: str,
                 class_names=None,
                 sample_rate: int = SAMPLE_RATE,
                 num_samples: int = NUM_SAMPLES,
                 n_mels: int = N_MELS):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.num_samples = num_samples

        # If class_names not provided, infer from folders
        if class_names is None:
            self.class_names = sorted(
                [d for d in os.listdir(root_dir)
                 if os.path.isdir(os.path.join(root_dir, d))]
            )
        else:
            self.class_names = class_names

        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        # Collect (filepath, label_idx)
        self.items = []
        for cls in self.class_names:
            class_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(".wav"):
                    fpath = os.path.join(class_dir, fname)
                    self.items.append((fpath, self.class_to_idx[cls]))

        # Audio transforms
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=n_mels,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        print(f"Loaded {len(self.items)} files from {root_dir}")
        print(f"Classes ({len(self.class_names)}): {self.class_names}")

    def __len__(self):
        return len(self.items)

    def _load_audio(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)   # [channels, samples]

        # Convert to mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        # Pad/trim to fixed length
        if wav.shape[1] < self.num_samples:
            pad_amount = self.num_samples - wav.shape[1]
            wav = nn.functional.pad(wav, (0, pad_amount))
        elif wav.shape[1] > self.num_samples:
            wav = wav[:, :self.num_samples]

        return wav  # [1, NUM_SAMPLES]

    def __getitem__(self, idx):
        path, label = self.items[idx]
        wav = self._load_audio(path)           # [1, samples]
        spec = self.melspec(wav)              # [1, n_mels, time]
        spec_db = self.amplitude_to_db(spec)  # [1, n_mels, time]

        # Simple normalization per example
        spec_db = (spec_db - spec_db.mean()) / (spec_db.std() + 1e-8)

        return spec_db, label


# ----------------------------
# 3. MODEL
# ----------------------------

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, n_mels: int = N_MELS):
        super().__init__()
        # Input: [B, 1, n_mels, time]
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # global avg pool
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)      # [B, 64, 1, 1]
        x = x.view(x.size(0), -1) # [B, 64]
        x = self.classifier(x)    # [B, num_classes]
        return x


# ----------------------------
# 4. BUILD DATASET & SPLIT
# ----------------------------

full_dataset = AudioFolderDataset(DATA_DIR)
class_names = full_dataset.class_names
num_classes = len(class_names)

# Stratified train/test split per class
indices_by_class = defaultdict(list)
for idx, (_, label) in enumerate(full_dataset.items):
    indices_by_class[label].append(idx)

train_indices = []
test_indices = []
g = torch.Generator().manual_seed(42)

for label, idxs in indices_by_class.items():
    idxs_tensor = torch.tensor(idxs)
    perm = idxs_tensor[torch.randperm(len(idxs_tensor), generator=g)].tolist()

    # Number of test samples for this class
    if len(perm) <= 1:
        n_test = 0  # keep tiny classes only in train
    else:
        n_test = max(1, int(round(len(perm) * TEST_RATIO)))

    test_indices.extend(perm[:n_test])
    train_indices.extend(perm[n_test:])

print(f"Total samples: {len(full_dataset)}")
print(f"Train samples: {len(train_indices)}")
print(f"Test  samples: {len(test_indices)}")

train_dataset = Subset(full_dataset, train_indices)
test_dataset  = Subset(full_dataset, test_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# ----------------------------
# 5. TRAINING
# ----------------------------

model = SimpleCNN(num_classes=num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(DEVICE)   # [B, 1, n_mels, time]
        targets = targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch+1} | Step {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f}")

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    print(f"[Train] Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

def evaluate():
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    if len(test_dataset) == 0:
        print("[Test ] No test set (not enough data per class). Skipping eval.")
        return None, None

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    print(f"[Test ] Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc

for epoch in range(NUM_EPOCHS):
    train_one_epoch(epoch)
    evaluate()

print("Training complete.")

# ----------------------------
# 6. EXPORT TO ONNX
# ----------------------------

# Grab one batch from the *train* loader to define input shape
example_batch, _ = next(iter(train_loader))
example_input = example_batch[:1].to(DEVICE)  # [1, 1, n_mels, time]

onnx_path = os.path.join(ROOT_DIR, "drone_classifier.onnx")
print(f"Exporting model to ONNX: {onnx_path}")

model.eval()
torch.onnx.export(
    model,
    example_input,
    onnx_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={
        "input": {"batch_size": 0},
        "logits": {"batch_size": 0},
    },
)

print("ONNX export complete.")

# Print mapping of class index → folder name
print("\nClass index mapping:")
for idx, name in enumerate(class_names):
    print(f"  {idx}: {name}")
