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

print(f"Train participants: {train_ids}")
print(f"Test participants: {test_ids}")
print(f"Folds: {folds}")

# Now let's start checking out the data

DATA = Path("data/raw/UCI HAR Dataset")

# The derived features will not be used for the CNN.
# The CNN will use the raw data because it will learn the features itself. 


SIGNALS = DATA / "train" / "Inertial Signals"

print("Making sure the Inertial Signals folder path is correct.", SIGNALS.is_dir())

print("\nThe files in this Inertial Signals folder:")
for path in sorted(SIGNALS.glob("*.txt")):
    print(path.name)

