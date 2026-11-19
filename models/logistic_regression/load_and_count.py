"""Card 2a, step 1: load the shared split file and count participants and rows.

Nothing here trains a model yet.
"""
import json
import numpy as np
from pathlib import Path

# File location: /models/logistic_regression/load_and_count.py

ROOT = Path(__file__).resolve().parents[2]

# path to the participant split data
path_to_split_file = ROOT / "data" / "splits" / "participants.json"

# open and load the file with json.load since its a json.file
with open(path_to_split_file, "r") as f:
    splits = json.load(f)

train_ids = splits["train_participants"]
test_ids = splits["test_participants"]

train_length = len(train_ids)
test_length = len(test_ids)

print(f"\nlength of Train participants: {train_length}")
print(f"\nlength of Test participants: {test_length}")

# TRAIN DATA LOADING
# path to subject_train.txt
DATA = ROOT / "data" / "raw" / "UCI HAR Dataset"
path_to_train_data = DATA / "train" / "subject_train.txt"

print(path_to_train_data.exists())


# load training data
train_subjects = np.loadtxt(path_to_train_data, dtype=np.int64)


print(train_subjects[:5])
print(len(train_subjects))



# TEST DATA LOADING
# path to subject_test.txt
path_to_test_data = DATA / "test" / "subject_test.txt"

print(path_to_test_data.exists())


# load test data
test_subjects = np.loadtxt(path_to_test_data, dtype=np.int64)


print(test_subjects[:5])
print(len(test_subjects))


# Check that participants match
# check that train participants match
print(set(np.unique(train_subjects)) == set(train_ids))


# check that test participants match
print(set(np.unique(test_subjects)) == set(test_ids))


# Load X and y
path_to_train_folder = DATA / "train"
X_train = np.loadtxt(path_to_train_folder / "X_train.txt", dtype=np.float64)
y_train = np.loadtxt(path_to_train_folder / "y_train.txt", dtype=np.int64)


path_to_test_folder = DATA / "test"
X_test = np.loadtxt(path_to_test_folder / "X_test.txt", dtype=np.float64)
y_test = np.loadtxt(path_to_test_folder / "y_test.txt", dtype=np.int64)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)