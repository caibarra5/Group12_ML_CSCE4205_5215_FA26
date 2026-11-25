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
# for configuration in INPUT_CONFIGURATIONS:
#     X_train_config, selected_names = select_input_configuration(
#         X_train_full, configuration
#     )
#     X_test_config, _ = select_input_configuration(
#         X_test, configuration
#     )

#     print(f"\nConfiguration: {configuration}")
#     print("Retained channels:", selected_names)
#     print("Training shape:", X_train_config.shape)
#     print("Test shape:", X_test_config.shape)


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

import random
import time

from sklearn.metrics import classification_report


FOCUSED_DIR = ROOT / "experiments" / "focused_study"
FOCUSED_DIR.mkdir(parents=True, exist_ok=True)

ACTIVITY_NAMES = [
    "walking",
    "walking_upstairs",
    "walking_downstairs",
    "sitting",
    "standing",
    "lying_down",
]
LABELS = list(range(6))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize_device():
    """Ensure GPU work finishes before reading a timer."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def activity_metrics(y_true, y_pred):
    """Return precision, recall, F1, and support for each activity."""
    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        target_names=ACTIVITY_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return {
        activity: report[activity]
        for activity in ACTIVITY_NAMES
    }


for configuration in INPUT_CONFIGURATIONS:
    print(f"\n{'=' * 60}")
    print(f"Configuration: {configuration}")
    print(f"{'=' * 60}")

    configuration_dir = FOCUSED_DIR / configuration
    configuration_dir.mkdir(parents=True, exist_ok=True)

    # 1. Select channels without modifying the original arrays.
    X_train_config, selected_names = select_input_configuration(
        X_train_full, configuration
    )
    X_test_config, _ = select_input_configuration(
        X_test, configuration
    )

    # 2. Standardize using training data only.
    # X_train_full and X_test must still be unstandardized here.
    channel_mean = X_train_config.mean(axis=(0, 2), keepdims=True)
    channel_std = X_train_config.std(axis=(0, 2), keepdims=True)
    channel_std = np.maximum(channel_std, 1e-8)

    X_train_config = (X_train_config - channel_mean) / channel_std
    X_test_config = (X_test_config - channel_mean) / channel_std

    # 3. Reset the seed before creating loaders and a fresh model.
    set_seed(seed)

    train_dataset = TensorDataset(
        torch.tensor(X_train_config, dtype=torch.float32),
        torch.tensor(y_train_full, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_config, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    # A separate generator keeps training-window order the same
    # across configurations, regardless of model parameter count.
    train_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=train_generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = Har1DCNN(
        in_channels=len(selected_names),
        kernel_size=best_config["kernel_size"],
        num_filters=best_config["num_filters"],
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=best_config["learning_rate"],
    )

    print("Retained channels:", selected_names)
    print("Training shape:", X_train_config.shape)
    print("Test shape:", X_test_config.shape)

    # 4. Train from scratch using the fixed epoch count.
    training_history = []

    synchronize_device()
    start_train = time.perf_counter()

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        total_windows = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            batch_size = batch_y.size(0)
            total_loss += loss.item() * batch_size
            total_windows += batch_size

        mean_loss = total_loss / total_windows
        training_history.append({
            "epoch": epoch + 1,
            "training_loss": mean_loss,
        })

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}: "
            f"training loss = {mean_loss:.4f}"
        )

    synchronize_device()
    train_seconds = time.perf_counter() - start_train

    # 5. Predict on the held-out test set.
    model.eval()
    prediction_batches = []

    synchronize_device()
    start_predict = time.perf_counter()

    with torch.no_grad():
        for batch_X, _ in test_loader:
            outputs = model(batch_X.to(device))
            predictions = outputs.argmax(dim=1)
            prediction_batches.append(predictions.cpu().numpy())

    synchronize_device()
    predict_seconds = time.perf_counter() - start_predict

    y_pred = np.concatenate(prediction_batches)

    # 6. Calculate shared metrics.
    final_metrics = evaluate(
        y_true=y_test,
        y_pred=y_pred,
        subjects=subjects_test,
        train_seconds=train_seconds,
        predict_seconds=predict_seconds,
        labels=LABELS,
    )

    # Include per-activity results for the hypothesis comparisons.
    final_metrics["per_activity"] = activity_metrics(y_test, y_pred)

    final_metrics["per_participant_activity"] = {}
    for participant in test_ids:
        mask = subjects_test == participant
        final_metrics["per_participant_activity"][str(participant)] = (
            activity_metrics(y_test[mask], y_pred[mask])
        )

    # 7. Save metrics and individual predictions.
    with (configuration_dir / "final_test_metrics.json").open("w") as f:
        json.dump(final_metrics, f, indent=4)
        f.write("\n")

    pd.DataFrame({
        "window_index": np.arange(len(y_test)),
        "participant_id": subjects_test,
        "y_true": y_test,
        "y_pred": y_pred,
    }).to_csv(
        configuration_dir / "test_predictions.csv",
        index=False,
    )

    pd.DataFrame(training_history).to_csv(
        configuration_dir / "training_history.csv",
        index=False,
    )

    # 8. Record settings and preprocessing for reproducibility.
    run_config = {
        "configuration": configuration,
        "removed_signal_groups": INPUT_CONFIGURATIONS[configuration],
        "retained_channels": selected_names,
        "in_channels": len(selected_names),
        "hyperparameters": best_config,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "dropout": DROPOUT,
        "seed": seed,
        "train_participants": train_ids,
        "test_participants": test_ids,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "class_names": ACTIVITY_NAMES,
        "standardization": "Per-channel training mean and standard deviation",
    }

    with (configuration_dir / "run_config.json").open("w") as f:
        json.dump(run_config, f, indent=4)
        f.write("\n")

    np.savez(
        configuration_dir / "standardization.npz",
        channel_mean=channel_mean,
        channel_std=channel_std,
    )

    # Save weights with the information needed to reconstruct the model.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_channels": len(selected_names),
            "best_config": best_config,
            "dropout": DROPOUT,
            "channel_names": selected_names,
            "channel_mean": torch.from_numpy(channel_mean),
            "channel_std": torch.from_numpy(channel_std),
        },
        configuration_dir / "model_checkpoint.pth",
    )

    print(f"Macro-F1: {final_metrics['macro_f1']:.4f}")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Saved results to: {configuration_dir}")