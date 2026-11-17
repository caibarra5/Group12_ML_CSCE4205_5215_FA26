import json
from pathlib import Path
import torch
import pandas as pd
import numpy as np

# File location: /models/one_d_CNN/train_1D_CNN_model.py

ROOT = Path(__file__).resolve().parents[2]

# Could be useful, but I prefer to run the python file as a module.
SPLIT_PATH = ROOT / "data" / "splits" / "participants.json"

with open("data/splits/participants.json", "r") as f:
    splits = json.load(f)

train_ids = splits["train_participants"]
test_ids = splits["test_participants"]
folds = splits["cv_folds"]

print(f"\nTrain participants: {train_ids}")
print(f"\nTest participants: {test_ids}")
print(f"\nFolds: {folds}")

# Now let's start checking out the data

DATA = Path("data/raw/UCI HAR Dataset")

# The derived features will not be used for the CNN.
# The CNN will use the raw data because it will learn the features itself. 

SIGNALS = DATA / "train" / "Inertial Signals"

print("\nMaking sure the Inertial Signals folder path is there.", SIGNALS.is_dir())

print("\nThe files in this Inertial Signals folder:")
for path in sorted(SIGNALS.glob("*.txt")):
    print(path.name)

body_acc_x = np.loadtxt(SIGNALS / "body_acc_x_train.txt", dtype = np.float32)

# print("\nShape:", body_acc_x.shape)
# print("First window, first 10 readings:", body_acc_x[0, :10])


# The order will determine which index corresponds to which channel
print("\n LOADING TRAINING DATA.")
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
    print(f"Loading channel: {name}")
    channel = np.loadtxt(
        SIGNALS / f"{name}_train.txt",
        dtype=np.float32,
    )
    channels.append(channel)

# Insert a channel axis between windows and time steps. 
print("\nInserting a channel axis between windows and time steps.")
X = np.stack(channels, axis=1)

# Now we know that the coordinates are (window, channel, time step)
print("\nFirst index is the window, the second the channel, and the third the time step.")
print("\nCombined shape:", X.shape)

print("\nLOADING TRAINING LABELS.")

# Each window corresponds to one activity label.

y = np.loadtxt(DATA/"train"/"y_train.txt", dtype=np.int64)

# Shift activity IDs to start from 0 instead of 1
y = y - 1

print("\nLOADING TRAINING SUBJECTS.")
subjects = np.loadtxt(DATA / "train" / "subject_train.txt", dtype = np.int64)

# Checking that each window has a label and a participant ID
assert X.shape[0] == y.shape[0] == subjects.shape[0], "The number of windows, labels, and subjects should be the same."


print("\nX.shape: ",X.shape, "y.shape: ", y.shape, "subjects.shape: ", subjects.shape)

print("\nClasses: ", np.unique(y))

print("\nParticipants: ", np.unique(subjects))

# Now we need to separate the training data into training and validation sets using the participants ID's

assert set(subjects.tolist()) == set(train_ids), "The subjects loaded and the train ids from our setup don't match. Check the participants.json file and the data loading code."

# print("\nFirst 40 train_ids values: ", train_ids[:40])
# print("\nFirst 40 subjects values: ", subjects[:40])

# Writing the CV loop first to start building the model hyperparameter tuning.
for fold_index, validation_ids in enumerate(folds):

    # These are this CV loop's training participants' ids.
    training_ids = [participant for participant in train_ids if participant not in validation_ids]

    # These are the boolean masks that separate the training and validation data for this CV loop. Each window is done by a participant.
    train_mask = np.isin(subjects, training_ids)
    val_mask = np.isin(subjects, validation_ids)

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_val = X[val_mask]
    y_val = y[val_mask]

    # Print just to check 
    print(f"\nFold {fold_index + 1}:")
    print(f"Training participants: {training_ids}")
    print(f"Validation participants: {validation_ids}")
    print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
    print(f"X_val.shape: {X_val.shape}, y_val.shape: {y_val.shape}")