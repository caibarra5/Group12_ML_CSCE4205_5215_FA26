"""Card 2a: logistic classifier for the UCI HAR data, written by hand.

The classifier is one-vs-rest: six yes/no questions, one per activity.
Each question has its own column of 561 weights.

What this file does, in order:
1. Load the shared participant split and the train/test data.
2. Check that the data files match the split file.
2b. Tune the learning rate and steps with the cross-validation folds (the test rows are not used).
2c. Standardize the features.
3. Turn the labels into a 0/1 table (one column per activity).
4. Train the weights with gradient descent (timed).
5. Predict the test rows (timed).
6. Score the predictions with the shared evaluate() from src/metrics.py.
7. Save the scores to experiments/ so card 3 can compare methods.
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
# Settings tried during tuning (the cross-validation folds pick the winner)
LEARNING_RATES = [0.03, 0.1]        # how big each step downhill is
STEP_COUNTS = [1000, 2000, 3000]    # how many times the training loop repeats
N_CLASSES = 6                       # six activities

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
    # Keep the probabilities just inside 0 and 1, so log() never sees exactly 0 (that gave nan).
    probs = np.clip(probs, 1e-12, 1 - 1e-12)
    return -np.mean(y_table * np.log(probs) + (1 - y_table) * np.log(1 - probs))


def train(X, Y_table, learning_rate, n_steps, verbose=True):
    """Train the weights with gradient descent. Returns the trained weight table.

    X is the features, Y_table is the 0/1 table of true answers.
    learning_rate is how big each step is, n_steps is how many steps to take.
    verbose=False switches off the loss printout (used while tuning).
    The weight table has one row per feature and one column per activity.
    """
    # Start at zero: the model knows nothing yet.
    W = np.zeros((X.shape[1], N_CLASSES))

    for step in range(n_steps):
        probs = sigmoid(X @ W)               # predict: a probability per activity
        error = probs - Y_table              # how far off, per cell
        gradient = (X.T @ error) / len(X)    # which way is uphill, per weight
        W = W - learning_rate * gradient     # step the opposite way (downhill)

        if verbose and step % 20 == 0:       # watch the loss fall
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
# 2b. Tune the learning rate and steps with the cross-validation folds
# ---------------------------------------------------------------
# This runs BEFORE the standardizing in 2c, because each fold has to work out its own
# average and spread from its own training people only. The test people are not used here.
cv_folds = splits["cv_folds"]
combos = [(lr, steps) for lr in LEARNING_RATES for steps in STEP_COUNTS]
fold_accuracies = {combo: [] for combo in combos}  # one accuracy per fold, per combination

for fold_number, held_out_ids in enumerate(cv_folds, start=1):
    held_out = np.isin(subjects_train, held_out_ids)  # True = row belongs to the held-out people
    X_fold_train, y_fold_train = X_train[~held_out], y_train[~held_out]
    X_fold_val, y_fold_val = X_train[held_out], y_train[held_out]

    # Average and spread from this fold's training rows only
    fold_mean = X_fold_train.mean(axis=0)
    fold_std = np.maximum(X_fold_train.std(axis=0), 1e-8)
    X_fold_train = (X_fold_train - fold_mean) / fold_std
    X_fold_val = (X_fold_val - fold_mean) / fold_std
    Y_fold_table = one_hot(y_fold_train, N_CLASSES)

    for learning_rate, n_steps in combos:
        W = train(X_fold_train, Y_fold_table, learning_rate, n_steps, verbose=False)
        accuracy = np.mean(predict(X_fold_val, W) == y_fold_val)
        fold_accuracies[(learning_rate, n_steps)].append(accuracy)
    print(f"Finished fold {fold_number} of {len(cv_folds)}")

# Average each combination's accuracies over the folds, then keep the best one
print("Mean cross-validation accuracy for each combination:")
for combo in combos:
    print(f"  learning rate {combo[0]}, steps {combo[1]}: {np.mean(fold_accuracies[combo]):.4f}")
best_learning_rate, best_steps = max(combos, key=lambda c: np.mean(fold_accuracies[c]))
print(f"Best: learning rate {best_learning_rate}, steps {best_steps}")

# ---------------------------------------------------------------
# 2c. Standardize the features
# ---------------------------------------------------------------
# Find each feature's average and spread from the TRAIN rows only, then use
# those same numbers on both train and test so they are on the same scale.
feature_mean = X_train.mean(axis=0)
feature_std = X_train.std(axis=0)
feature_std = np.maximum(feature_std, 1e-8)  # a spread of 0 would divide by zero

X_train = (X_train - feature_mean) / feature_std
X_test = (X_test - feature_mean) / feature_std

# ---------------------------------------------------------------
# 3. Turn the train labels into the 0/1 table
# ---------------------------------------------------------------
y_train_onehot = one_hot(y_train, N_CLASSES)

# Each row should have exactly one 1, so every row adds up to 1.
print("Every row of the table adds to 1:", (y_train_onehot.sum(axis=1) == 1).all())

# ---------------------------------------------------------------
# 4. Train (timed)
# ---------------------------------------------------------------
W_table, train_seconds = timed(train, X_train, y_train_onehot, best_learning_rate, best_steps)

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

# ---------------------------------------------------------------
# 7. Save the results
# ---------------------------------------------------------------
# Write everything evaluate() returned into a JSON file, the same way the other
# methods do, so card 3 can compare the files directly.

results_dir = ROOT / "experiments" / "logistic"
results_dir.mkdir(parents=True, exist_ok=True)
out_path = results_dir / "final_test_metrics.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=4)
print("Saved results to", out_path)
