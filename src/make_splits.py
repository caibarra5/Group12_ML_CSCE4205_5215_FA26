"""Create the shared participant split used by every method.

Run once from the repo root:  python src/make_splits.py
It writes data/splits/participants.json. That file is committed, so all three
of us load the same list instead of re-creating it on our own machines.
"""
import json
import random
from pathlib import Path

SEED = 42
N_FOLDS = 7  # 21 training participants -> 7 folds of 3

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "UCI HAR Dataset"
OUT = ROOT / "data" / "splits" / "participants.json"


def read_ids(path):
    """Each line of a subject file is the participant ID for one row."""
    return sorted({int(line) for line in path.read_text().split()})


# 1. Use the dataset's own split: 21 training and 9 test participants.
train_ids = read_ids(RAW / "train" / "subject_train.txt")
test_ids = read_ids(RAW / "test" / "subject_test.txt")

# 2. Safety checks: nobody may appear on both sides.
assert len(train_ids) == 21 and len(test_ids) == 9
assert not set(train_ids) & set(test_ids)

# 3. Shuffle the 21 training IDs with the fixed seed, then cut into 7 folds.
shuffled = train_ids.copy()
random.Random(SEED).shuffle(shuffled)
size = len(shuffled) // N_FOLDS
folds = [sorted(shuffled[i * size:(i + 1) * size]) for i in range(N_FOLDS)]
assert sorted(p for fold in folds for p in fold) == train_ids

# 4. Save. The test participants are never used for tuning.
split = {
    "seed": SEED,
    "n_folds": N_FOLDS,
    "train_participants": train_ids,
    "test_participants": test_ids,
    "cv_folds": folds,
}
# newline="\n" keeps LF line endings on Windows too, so reruns match the committed file
with open(OUT, "w", newline="\n") as f:
    f.write(json.dumps(split, indent=2) + "\n")
print(f"Wrote {OUT.relative_to(ROOT)}")
for i, fold in enumerate(folds):
    print(f"  fold {i}: {fold}")
