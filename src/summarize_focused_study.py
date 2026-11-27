# Collect focused-study results into CSV tables. We can copy and paste them into our
# report.

# Run from the repository root: python -m src.summarize_focused_study


import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "focused_study"
OUTPUT_DIR = RESULTS_DIR / "tables"

CONFIGURATIONS = {
    "all_inputs": "All inputs (baseline)",
    "without_angular_velocity": "Without angular velocity",
    "without_both_accelerations": "Without both acceleration groups",
    "without_body_acceleration": "Without body acceleration",
    "without_total_acceleration": "Without total acceleration",
}

ACTIVITIES = [
    "walking",
    "walking_upstairs",
    "walking_downstairs",
    "sitting",
    "standing",
    "lying_down",
]

ACTIVITY_GROUPS = {
    "Walking activities": ACTIVITIES[:3],
    "Stationary postures": ACTIVITIES[3:],
}

METRICS = {
    "precision": "Precision",
    "recall": "Recall",
    "f1-score": "F1",
}

def save_table(table, filename):
    table.to_csv(
        OUTPUT_DIR / filename,
        index=False,
        float_format="%.6f",
    )
    print(f"Saved: {OUTPUT_DIR / filename}")


def group_scores(activity_results, activities):
    """Equal-weight average of the three activities' metrics."""
    return {
        label: float(np.mean([
            activity_results[activity][key]
            for activity in activities
        ]))
        for key, label in METRICS.items()
    }
