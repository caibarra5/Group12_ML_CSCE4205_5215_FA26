import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.metrics import evaluate

# Loading the data, defining the channel order, and verifying the splits.
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

# Now we will make the setup for the input configuraitons

# 6. Define the signal groups and input configurations.
SIGNAL_GROUPS = {
    "body_acceleration": [0, 1, 2],
    "angular_velocity": [3, 4, 5],
    "total_acceleration": [6, 7, 8],
}

# Each configuration lists the signal groups to remove.
INPUT_CONFIGURATIONS = {
    "all_inputs": [],
    "without_total_acceleration": ["total_acceleration"],
    "without_body_acceleration": ["body_acceleration"],
    "without_angular_velocity": ["angular_velocity"],
    "without_both_accelerations": [
        "body_acceleration",
        "total_acceleration",
    ],
}


def select_input_configuration(X, configuration):
    """Return selected channels and their names without changing X."""
    removed_groups = INPUT_CONFIGURATIONS[configuration]

    removed_indices = {
        index
        for group in removed_groups
        for index in SIGNAL_GROUPS[group]
    }

    kept_indices = [
        index
        for index in range(len(CHANNEL_NAMES))
        if index not in removed_indices
    ]

    selected_names = [CHANNEL_NAMES[index] for index in kept_indices]

    return X[:, kept_indices, :].copy(), selected_names


# Check the configurations. Training will be added later.
for configuration in INPUT_CONFIGURATIONS:
    X_train_config, selected_names = select_input_configuration(
        X_train_full, configuration
    )
    X_test_config, _ = select_input_configuration(
        X_test, configuration
    )

    print(f"\nConfiguration: {configuration}")
    print("Retained channels:", selected_names)
    print("Training shape:", X_train_config.shape)
    print("Test shape:", X_test_config.shape)


# Now we will define the architecture

# 7. Load the best hyperparameters from the baseline tuning results.
import pandas as pd

TUNING_PATH = (
    ROOT / "experiments" / "one_d_cnn" / "tuning_results_64.csv"
)

tuning_results = pd.read_csv(TUNING_PATH)

# Use the same selection criterion as the baseline script.
best_row = tuning_results.loc[
    tuning_results["mean_cv_accuracy"].idxmax()
]

best_config = {
    "learning_rate": float(best_row["learning_rate"]),
    "kernel_size": int(best_row["kernel_size"]),
    "num_filters": int(best_row["num_filters"]),
}

print("\nFixed hyperparameters for all input configurations:")
print(best_config)


# 8. Define the same CNN architecture as the baseline. The number of input channels will be set with X_train_config.shape[1]. 
# Output channels for each layer will not change.
class Har1DCNN(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes=6,
        kernel_size=3,
        num_filters=32,
        dropout=0.5,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(
            num_filters,
            num_filters * 2,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(num_filters * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# these settings are from the baseline, so all of the model instances should use them
BATCH_SIZE = 64
NUM_EPOCHS = 5
DROPOUT = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")