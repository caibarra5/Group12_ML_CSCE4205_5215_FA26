
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np

ACTIVITY_NAMES = {
    1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING', 5: 'STANDING', 6: 'LAYING',
}
LABELS = list(ACTIVITY_NAMES)

@dataclass(frozen=True)
class HARData:
    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray


def load_har(root: Path, section: str) -> HARData:
    """Load engineered 561-feature representation only (not inertial signals)."""
    if section not in ('train', 'test'):
        raise ValueError('section must be train or test')
    root = Path(root)
    folder = root / section
    paths = [folder / f'{f}_{section}.txt' for f in ('X', 'y', 'subject')]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f'Missing dataset file: {p}')
    X = np.loadtxt(paths[0], dtype=np.float64, ndmin=2)
    y = np.loadtxt(paths[1], dtype=np.int64, ndmin=1).reshape(-1)
    subjects = np.loadtxt(paths[2], dtype=np.int64, ndmin=1).reshape(-1)
    if len(X) != len(y) or len(y) != len(subjects):
        raise ValueError(f'Row counts differ for {section}')
    if X.shape[1] == 0 or not np.isfinite(X).all():
        raise ValueError('Feature matrix is invalid')
    if not set(np.unique(y)).issubset(ACTIVITY_NAMES):
        raise ValueError('Unknown activity labels')
    return HARData(X=X, y=y, subjects=subjects)


def read_shared_splits(split_json: Path) -> dict:
    """Load David's committed file; never regenerate folds in model code."""
    with open(split_json, encoding='utf-8') as f:
        split = json.load(f)
    train = [int(x) for x in split['train_participants']]
    test = [int(x) for x in split['test_participants']]
    folds = [[int(x) for x in fold] for fold in split['cv_folds']]
    if len(train) != 21 or len(test) != 9 or len(folds) != 7:
        raise ValueError('Unexpected split definition (expected 21 train, 9 test, 7 folds)')
    if any(len(fold) != 3 for fold in folds):
        raise ValueError('Every validation fold must have three participants')
    if len(set(train)) != len(train) or len(set(test)) != len(test):
        raise ValueError('Repeated participant IDs in shared split')
    if set(train) & set(test):
        raise ValueError('Train/test participant leakage')
    flat = [id for fold in folds for id in fold]
    if len(flat) != 21 or set(flat) != set(train):
        raise ValueError('CV folds must partition training participants exactly')
    if split.get('seed') != 42 or split.get('n_folds') != 7:
        raise ValueError('Use the team-approved seed 42 and 7 folds')
    return split


def verify_participants(actual_subjects, expected_ids, section: str):
    found = set(np.asarray(actual_subjects, dtype=int).tolist())
    expected = set(int(x) for x in expected_ids)
    if found != expected:
        raise ValueError(
            f'{section} participants differ from committed split. '
            f'Missing={sorted(expected-found)}, extra={sorted(found-expected)}'
        )
