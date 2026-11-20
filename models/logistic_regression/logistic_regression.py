"""Card 2a: logistic classifier for the UCI HAR data, written by hand.

The classifier is one-vs-rest: six yes/no questions, one per activity.
Each question has its own column of 561 weights.

What this file does, in order:
1. Load the shared participant split and the train/test data.
2. Check that the data files match the split file.
3. Turn the labels into a 0/1 table (one column per activity).
4. Train the weights with gradient descent and watch the loss fall.

NEXT: predict the test rows and call evaluate() from src/metrics.py.
"""
import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------
# Settings and paths
# ---------------------------------------------------------------
LEARNING_RATE = 0.01  # how big each step downhill is (0.1 overshot, 0.01 worked)
N_STEPS = 200         # how many times the training loop repeats
N_CLASSES = 6         # six activities

# This file lives in /models/logistic_regression/, so the repo root is two folders up.
ROOT = Path(__file__).resolve().parents[2]
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
# 4. Train the weights with gradient descent
# ---------------------------------------------------------------
# One column of 561 weights per activity. Start at zero (the model knows nothing yet).
W_table = np.zeros((X_train.shape[1], N_CLASSES))

for step in range(N_STEPS):
    # a) Predict: a score per activity, then a probability per activity.
    scores = X_train @ W_table
    probs = sigmoid(scores)

    # b) How wrong are we? (we only compute this to watch it fall)
    loss = logistic_loss(y_train_onehot, probs)

    # c) Which way is uphill? The error per cell, then the gradient per weight.
    error = probs - y_train_onehot
    gradient = (X_train.T @ error) / len(y_train)

    # d) Take a small step the opposite way (downhill).
    W_table = W_table - LEARNING_RATE * gradient

    if step % 20 == 0:
        print(f"step {step}: loss {loss:.4f}")

# ---------------------------------------------------------------
# 5. Predict the test rows
# ---------------------------------------------------------------
# a) Same steps as training, but on the TEST features and the trained weights.
scores_test = X_test @ W_table
probs_test = sigmoid(scores_test)

# b) For each row, pick the activity with the highest probability.
#    Columns run 0-5 and labels run 1-6, so add 1.
y_pred = np.argmax(probs_test, axis=1) + 1

print(y_pred.shape)
print(y_pred[:10], y_test[:10])

# c) Accuracy: the share of rows where our pick matches the true label.
accuracy = np.mean(y_pred == y_test)
print(f"Test accuracy: {accuracy:.4f}")