"""Shared metrics, so all three methods report the same numbers.

Primary metric: macro-F1, the plain average of the per-activity F1 scores.
Also here: accuracy, the confusion matrix, per-participant scores and timing.

Labels can be 1-6 (as in the UCI files) or 0-5, as long as y_true and
y_pred use the same ones.
"""
import time

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def overall_scores(y_true, y_pred):
    """Headline numbers for one set of predictions."""
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def confusion(y_true, y_pred, labels=None):
    """Confusion matrix as a list of lists. Row = true activity, column = predicted."""
    return confusion_matrix(y_true, y_pred, labels=labels).tolist()


def per_participant(y_true, y_pred, subjects):
    """One set of scores per participant.

    subjects holds the participant ID for every row, in the same order as
    y_true and y_pred.
    """
    y_true, y_pred, subjects = np.asarray(y_true), np.asarray(y_pred), np.asarray(subjects)
    assert len(y_true) == len(y_pred) == len(subjects)
    scores = {}
    for person in sorted(set(subjects.tolist())):
        rows = subjects == person  # True for this person's rows only
        scores[int(person)] = overall_scores(y_true[rows], y_pred[rows])
    return scores


def timed(fn, *args, **kwargs):
    """Run fn(*args, **kwargs). Returns (what fn returned, seconds it took)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def evaluate(y_true, y_pred, subjects, train_seconds=None, predict_seconds=None, labels=None):
    """Everything card 1b asks for, in one dictionary. Every method calls this."""
    return {
        **overall_scores(y_true, y_pred),
        "confusion_matrix": confusion(y_true, y_pred, labels),
        "per_participant": per_participant(y_true, y_pred, subjects),
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
    }
