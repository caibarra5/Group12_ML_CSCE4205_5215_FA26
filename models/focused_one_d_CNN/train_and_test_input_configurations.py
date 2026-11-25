import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.metrics import evaluate


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw" / "UCI HAR Dataset"
SPLIT_PATH = ROOT / "data" / "splits" / "participants.json"


# 1. Load the shared participant split.
with SPLIT_PATH.open("r") as f:
    splits = json.load(f)

train_ids = splits["train_participants"]
test_ids = splits["test_participants"]
seed = splits["seed"]

assert not set(train_ids) & set(test_ids), (
    "Training and test participants must not overlap."
)


# 2. Define the channel order.
CHANNEL_NAMES = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


# 3. Load signals, activity labels, and participant IDs.
def load_har_data(subset):
    signals_dir = DATA / subset / "Inertial Signals"

    channels = [
        np.loadtxt(
            signals_dir / f"{name}_{subset}.txt",
            dtype=np.float32,
        )
        for name in CHANNEL_NAMES
    ]

    # Shape: (windows, channels, time steps)
    X = np.stack(channels, axis=1)

    # Convert activity labels from 1–6 to 0–5.
    y = np.loadtxt(
        DATA / subset / f"y_{subset}.txt",
        dtype=np.int64,
    ) - 1

    subjects = np.loadtxt(
        DATA / subset / f"subject_{subset}.txt",
        dtype=np.int64,
    )

    assert X.shape[0] == y.shape[0] == subjects.shape[0], (
        "Window, label, and participant counts must match."
    )
    assert X.shape[1:] == (9, 128), (
        "Each window must contain nine channels and 128 time steps."
    )

    return X, y, subjects


# 4. Load the existing training and test sets.
X_train_full, y_train_full, subjects_train_full = load_har_data("train")
X_test, y_test, subjects_test = load_har_data("test")


# 5. Verify that they match the team's shared split.
assert set(np.unique(subjects_train_full)) == set(train_ids), (
    "Training participants do not match participants.json."
)
assert set(np.unique(subjects_test)) == set(test_ids), (
    "Test participants do not match participants.json."
)

print("Training participants:", train_ids)
print("Test participants:", test_ids)

print("\nTraining shapes:")
print("X:", X_train_full.shape)
print("y:", y_train_full.shape)
print("subjects:", subjects_train_full.shape)

print("\nTest shapes:")
print("X:", X_test.shape)
print("y:", y_test.shape)
print("subjects:", subjects_test.shape)