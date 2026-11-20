import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

from src.metrics import evaluate, timed


ROOT = Path(__file__).resolve().parents[2]

# 1. Load splits and best hyperparameters from CSV
SPLIT_PATH = ROOT / "data" / "splits" / "participants.json"
with open(SPLIT_PATH, "r") as f:
    splits = json.load(f)

train_ids = splits["train_participants"]
test_ids = splits["test_participants"]

# Load the 64-config results CSV generated from tuning script
RESULTS_DIR = ROOT / "experiments" / "one_d_cnn"
csv_path = RESULTS_DIR / "tuning_results_64.csv"
results_df = pd.read_csv(csv_path)

# configuration with the highest mean CV accuracy
best_row = results_df.loc[results_df["mean_cv_accuracy"].idxmax()]

best_config = {
    "learning_rate": float(best_row["learning_rate"]),
    "kernel_size": int(best_row["kernel_size"]),
    "num_filters": int(best_row["num_filters"])
}

print(f"\nLoaded Best Hyperparameters from CSV:")
print(best_config)
print(f"Achieved Mean CV Accuracy during tuning: {best_row['mean_cv_accuracy']:.4f}")

# 2. Load Full Training and Test Data
DATA = ROOT / "data" / "raw" / "UCI HAR Dataset"

def load_har_data(subset_type):
    signals_dir = DATA / subset_type / "Inertial Signals"
    channel_names = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z",
    ]
    channels = [np.loadtxt(signals_dir / f"{name}_{subset_type}.txt", dtype=np.float32) for name in channel_names]
    X = np.stack(channels, axis=1)
    y = np.loadtxt(DATA / subset_type / f"y_{subset_type}.txt", dtype=np.int64) - 1
    subjects = np.loadtxt(DATA / subset_type / f"subject_{subset_type}.txt", dtype=np.int64)
    return X, y, subjects

print("\nLoading Training and Test datasets...")
X_train_full, y_train_full, subjects_train_full = load_har_data("train")
X_test, y_test, subjects_test = load_har_data("test")

# Verifying participants match split expectations
assert set(np.unique(subjects_train_full)) == set(train_ids)
assert set(np.unique(subjects_test)) == set(test_ids)

# 3. Define the 1D CNN Model Architecture (same as the tuning script)
class Har1DCNN(nn.Module):
    def __init__(self, in_channels=9, num_classes=6, kernel_size=3, num_filters=32, dropout=0.5):
        super(Har1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, num_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size=kernel_size, padding=kernel_size // 2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_filters * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Prepare DataLoaders
train_dataset = TensorDataset(torch.tensor(X_train_full, dtype=torch.float32), torch.tensor(y_train_full, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 4. Train Final Model (timed)
model = Har1DCNN(
    in_channels=9,
    num_classes=6,
    kernel_size=best_config["kernel_size"],
    num_filters=best_config["num_filters"]
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=best_config["learning_rate"])
num_epochs = 5  # same epoch as in the CV search

print("\nTraining final model on full training set...")
import time
start_train = time.perf_counter()

for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(batch_X), batch_y)
        loss.backward()
        optimizer.step()

train_seconds = time.perf_counter() - start_train

# Save trained model weights
model_output_path = Path(__file__).resolve().parent / "best_1d_cnn_model.pth"
torch.save(model.state_dict(), model_output_path)
print(f"Saved final model weights to: {model_output_path}")

# 5. Evaluate on Test Set (timed prediction)
print("\nEvaluating model on the held-out test set...")
model.eval()
all_preds = []

start_pred = time.perf_counter()
with torch.no_grad():
    for batch_X, _ in test_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
predict_seconds = time.perf_counter() - start_pred

y_pred = np.array(all_preds)

# 6. Final metrics dictionary using the shared metrics module
final_metrics = evaluate(
    y_true=y_test,
    y_pred=y_pred,
    subjects=subjects_test,
    train_seconds=train_seconds,
    predict_seconds=predict_seconds,
    labels=[0, 1, 2, 3, 4, 5]
)

print("\n=== FINAL TEST EVALUATION RESULTS ===")
print(f"Macro-F1: {final_metrics['macro_f1']:.4f}")
print(f"Accuracy: {final_metrics['accuracy']:.4f}")
print(f"Training Time: {final_metrics['train_seconds']:.2f}s")
print(f"Prediction Time: {final_metrics['predict_seconds']:.2f}s")

# Save final metrics summary to JSON
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
metrics_output_path = RESULTS_DIR / "final_test_metrics.json"
with open(metrics_output_path, "w") as f:
    json.dump(final_metrics, f, indent=4)
print(f"Saved final metrics report to: {metrics_output_path}")
