"""Card 2a: logistic classifier for the UCI HAR data, written by hand.

The classifier is one-vs-rest: six yes/no questions, one per activity.
Each question has its own column of 561 weights.

What this file does, in order:
1. Load the shared participant split and the train/test data.
2. Check that the data files match the split file.
3. Turn the labels into a 0/1 table (one column per activity).
4. Train the weights with gradient descent (timed).
5. Predict the test rows (timed).
6. Score the predictions with the shared evaluate() from src/metrics.py.

NEXT: save the results to experiments/ so card 3 can compare methods.
"""
import json
import sys
from pathlib import Path

import numpy as np

# This file lives in /models/logistic_regression/, so the repo root is two folders up.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # lets Python find src/metrics.py

from src.metrics import evaluate, timed  # noqa: E402  (must come after the sys.path line)

# ---------------------------------------------------------------
# Settings and paths
# ---------------------------------------------------------------
LEARNING_RATE = 0.01  # how big each step downhill is (0.1 overshot, 0.01 worked)
N_STEPS = 200         # how many times the training loop repeats
N_CLASSES = 6         # six activities

SPLIT_FILE = ROOT / "data" / "splits" / "participants.json"
DATA = ROOT / "data" / "raw" / "UCI HAR Dataset"


# ---------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------
def load_har(name):
    """Load features (X), labels (y) and participant IDs for 'train' or 'test'."""
    folder = DATA / name
    X = np.loadtxt(folder / f"X_{name}.txt", dtype=np.float64)
    y = np.loadtxt(folder / f"y_{name}.txt", dtype=np.int64)
    subjects = np.loadtxt(folder / f"subject_{name}.txt", dtype=np.int64)
    return X, y, subjects


def one_hot(y, n_classes):
    """Turn labels 1-6 into a 0/1 table: one row per window, one column per activity.

    Label 3 becomes [0, 0, 1, 0, 0, 0] (column 2, since columns start at 0).
    """
    table = np.zeros((len(y), n_classes))
    for i, label in enumerate(y):
        table[i][label - 1] = 1
    return table


def sigmoid(x):
    """Squish any score into a probability between 0 and 1."""
    return 1 / (1 + np.exp(-x))


def logistic_loss(y_table, probs):
    """Average logistic loss over every cell of the table.

    Cells where the truth is 1 use -log(prob).
    Cells where the truth is 0 use -log(1 - prob).
    """
    return -np.mean(y_table * np.log(probs) + (1 - y_table) * np.log(1 - probs))


def train(X, Y_table):
    """Train the weights with gradient descent. Returns the trained weight table.

    X is the features, Y_table is the 0/1 table of true answers.
    The weight table has one row per feature and one column per activity.
    """
    # Start at zero: the model knows nothing yet.
    W = np.zeros((X.shape[1], N_CLASSES))

    for step in range(N_STEPS):
        probs = sigmoid(X @ W)               # predict: a probability per activity
        error = probs - Y_table              # how far off, per cell
        gradient = (X.T @ error) / len(X)    # which way is uphill, per weight
        W = W - LEARNING_RATE * gradient     # step the opposite way (downhill)

        if step % 20 == 0:                   # watch the loss fall
            print(f"step {step}: loss {logistic_loss(Y_table, probs):.4f}")

    return W


def predict(X, W):
    """Pick the activity (1-6) with the highest probability for each row."""
    probs = sigmoid(X @ W)
    # Columns run 0-5 and labels run 1-6, so add 1.
    return np.argmax(probs, axis=1) + 1


# ---------------------------------------------------------------
# 1. Load the shared participant split
# ---------------------------------------------------------------
with open(SPLIT_FILE, "r") as f:
    splits = json.load(f)

train_ids = splits["train_participants"]
test_ids = splits["test_participants"]

print(f"Train participants: {len(train_ids)}")
print(f"Test participants: {len(test_ids)}")

# ---------------------------------------------------------------
# 2. Load the data and check it matches the split
# ---------------------------------------------------------------
X_train, y_train, subjects_train = load_har("train")
X_test, y_test, subjects_test = load_har("test")

print(f"Train rows: {len(y_train)}, features: {X_train.shape[1]}")
print(f"Test rows: {len(y_test)}, features: {X_test.shape[1]}")

# Every participant in the data files should be exactly the ones in the split file.
print("Train participants match the split file:", set(np.unique(subjects_train)) == set(train_ids))
print("Test participants match the split file:", set(np.unique(subjects_test)) == set(test_ids))

# ---------------------------------------------------------------
# 3. Turn the train labels into the 0/1 table
# ---------------------------------------------------------------
y_train_onehot = one_hot(y_train, N_CLASSES)

# Each row should have exactly one 1, so every row adds up to 1.
print("Every row of the table adds to 1:", (y_train_onehot.sum(axis=1) == 1).all())

# ---------------------------------------------------------------
# 4. Train (timed)
# ---------------------------------------------------------------
W_table, train_seconds = timed(train, X_train, y_train_onehot)

# ---------------------------------------------------------------
# 5. Predict the test rows (timed)
# ---------------------------------------------------------------
y_pred, predict_seconds = timed(predict, X_test, W_table)

# ---------------------------------------------------------------
# 6. Score the predictions with the shared evaluate()
# ---------------------------------------------------------------
results = evaluate(
    y_test,            # the real test labels
    y_pred,            # our predictions
    subjects_test,     # participant ID of every test row (for per-person scores)
    train_seconds,
    predict_seconds,
    labels=[1, 2, 3, 4, 5, 6],  # all six activity labels
)
print("macro-F1:", results["macro_f1"])
print("accuracy:", results["accuracy"])
