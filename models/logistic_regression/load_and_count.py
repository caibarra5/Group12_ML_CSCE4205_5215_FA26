"""Card 2a, step 1: load the shared split file and count participants and rows.

Nothing here trains a model yet.
"""
import json
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