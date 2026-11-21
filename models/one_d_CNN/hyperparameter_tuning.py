import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from itertools import product
import pandas as pd
import numpy as np
import time

# File location: /models/one_d_CNN/hyperparameter_tuning.py

import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

ROOT = Path(__file__).resolve().parents[2]

SPLIT_PATH = ROOT / "data" / "splits" / "participants.json"

with open(SPLIT_PATH, "r") as f:
    splits = json.load(f)

train_ids = splits["train_participants"]
test_ids = splits["test_participants"]
folds = splits["cv_folds"]

print(f"\nTrain participants: {train_ids}")
print(f"\nTest participants: {test_ids}")
print(f"\nFolds: {folds}")

# Now let's start checking out the data
DATA = ROOT / "data" / "raw" / "UCI HAR Dataset"

SIGNALS = DATA / "train" / "Inertial Signals"

print("\nMaking sure the Inertial Signals folder path is there.", SIGNALS.is_dir())

# LOADING TRAINING DATA
channel_names = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]

channels = []
print("\nLoading the channels in the order specified by channel_names.")
for name in channel_names:
    channel = np.loadtxt(
        SIGNALS / f"{name}_train.txt",
        dtype=np.float32,
    )
    channels.append(channel)

# Insert a channel axis between windows and time steps -> Shape: (window, channel, time step)
X = np.stack(channels, axis=1)
print(f"\nCombined shape: {X.shape}")

# LOADING TRAINING LABELS
y = np.loadtxt(DATA / "train" / "y_train.txt", dtype=np.int64)
y = y - 1  # Shift activity IDs to start from 0 instead of 1

# LOADING TRAINING SUBJECTS
subjects = np.loadtxt(DATA / "train" / "subject_train.txt", dtype=np.int64)

assert X.shape[0] == y.shape[0] == subjects.shape[0], "The number of windows, labels, and subjects should be the same."
assert set(subjects.tolist()) == set(train_ids), "The subjects loaded and the train ids from our setup don't match."

print(f"\nX.shape: {X.shape}, y.shape: {y.shape}, subjects.shape: {subjects.shape}")
print(f"Classes: {np.unique(y)}")
print(f"Participants: {np.unique(subjects)}")


# ==========================================
# 1D CNN MODEL DEFINITION
# ==========================================
class Har1DCNN(nn.Module):
    def __init__(self, in_channels=9, num_classes=6, kernel_size=3, num_filters=32, dropout=0.5):
        super(Har1DCNN, self).__init__()
        # Conv block 1
        self.conv1 = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=num_filters, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Conv block 2
        self.conv2 = nn.Conv1d(
            in_channels=num_filters, 
            out_channels=num_filters * 2, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2
        )
        
        # Global pooling and fully connected classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_filters * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape expected: (batch_size, channels, timesteps)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        out = self.fc(x)
        return out


# ==========================================
# HYPERPARAMETER SEARCH GRID (3 Parameters)
# ==========================================
param_grid = {
    "learning_rate": [0.01, 0.001, 0.0005, 0.0001],  # Added 0.01 and 0.0001
    "kernel_size": [3, 5, 7, 9],                     # Added 7 and 9
    "num_filters": [16, 32, 64, 128]                 # Added 16 and 128
}

print("Hyperparameter grid:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")

keys, values = zip(*param_grid.items())
hyperparameter_configs = [dict(zip(keys, v)) for v in product(*values)]

print(f"\nTotal hyperparameter configurations to evaluate: {len(hyperparameter_configs)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

best_accuracy = 0.0
best_config = None

# There are several hyperparameters to configure. But I chose to focus on learning rate, kernel size and number of filters.
# I have no idea how to explore more parameters without running into combinatorial explosion.

print("\nStarting hyperparameter tuning...")
start = time.time()

# 1. Initialize a list to track all results
results_log = []

# ==========================================
# OUTER LOOP: Hyperparameter Configurations
# ==========================================
for config_idx, config in enumerate(hyperparameter_configs):
    print(f"\n========================================")
    print(f"Evaluating Config {config_idx + 1}/{len(hyperparameter_configs)}: {config}")
    print(f"========================================")
    
    fold_accuracies = []

    # ==========================================
    # INNER LOOP: Cross-Validation Folds
    # ==========================================
    for fold_index, validation_ids in enumerate(folds):
        set_seed(splits["seed"] + fold_index)  # Here's the fix for the seed on this script. NOw for the train and evaluate script.
        training_ids = [p for p in train_ids if p not in validation_ids]

        train_mask = np.isin(subjects, training_ids)
        val_mask = np.isin(subjects, validation_ids)

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        model = Har1DCNN(
            in_channels=9,
            num_classes=6,
            kernel_size=config["kernel_size"],
            num_filters=config["num_filters"]
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

        num_epochs = 5
        for epoch in range(num_epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        val_accuracy = correct / total
        fold_accuracies.append(val_accuracy)
        print(f"  -> Fold {fold_index + 1} Val Accuracy: {val_accuracy:.4f}")

    mean_cv_accuracy = np.mean(fold_accuracies)
    print(f"Mean CV Accuracy for Config {config_idx + 1}: {mean_cv_accuracy:.4f}")

    if mean_cv_accuracy > best_accuracy:
        best_accuracy = mean_cv_accuracy
        best_config = config

    # 2. Log this configuration's results
    results_log.append({
        "config_id": config_idx + 1,
        "learning_rate": config["learning_rate"],
        "kernel_size": config["kernel_size"],
        "num_filters": config["num_filters"],
        **{f"fold_{i+1}_acc": acc for i, acc in enumerate(fold_accuracies)},
        "mean_cv_accuracy": mean_cv_accuracy
    })

end = time.time()
print(f"Hyperparameter tuning time: {end - start:.2f} seconds")

# 3. Save all results to a CSV file in experiments/one_d_cnn/
results_df = pd.DataFrame(results_log)
results_dir = ROOT / "experiments" / "one_d_cnn"
results_dir.mkdir(parents=True, exist_ok=True)
csv_output_path = results_dir / "tuning_results_64.csv"
results_df.to_csv(csv_output_path, index=False)
print(f"\nSaved all 64 configurations and results to: {csv_output_path}")

print(f"\n\nOptimization Complete!")
print(f"Best Hyperparameters: {best_config}")
print(f"Best Mean CV Accuracy: {best_accuracy:.4f}")
