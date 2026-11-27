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
    csv_path = OUTPUT_DIR / filename
    markdown_path = csv_path.with_suffix(".md")

    table.to_csv(
        csv_path,
        index=False,
        float_format="%.6f",
    )

    markdown_path.write_text(
        table.to_markdown(index=False, floatfmt=".4f") + "\n",
        encoding="utf-8",
    )

    print(f"Saved: {csv_path}")
    print(f"Saved: {markdown_path}")


def group_scores(activity_results, activities):
    """Equal-weight average of the three activities' metrics."""
    return {
        label: float(np.mean([
            activity_results[activity][key]
            for activity in activities
        ]))
        for key, label in METRICS.items()
    }



def main():
    # Read only saved test metrics. 
    results = {}
    for configuration in CONFIGURATIONS:
        path = RESULTS_DIR / configuration / "final_test_metrics.json"

        if not path.is_file():
            if configuration == "all_inputs":
                raise FileNotFoundError(f"Baseline results are required: {path}")
            print(f"Skipping missing configuration: {configuration}")
            continue

        with path.open("r") as f:
            results[configuration] = json.load(f)

    baseline = results["all_inputs"]
    participant_ids = sorted(
        baseline["per_participant"], key=int
    )

    # Check that comparisons use the same participants and class counts.
    baseline_cm = np.asarray(baseline["confusion_matrix"])
    if baseline_cm.shape != (6, 6):
        raise ValueError("Expected a 6-by-6 baseline confusion matrix.")

    for configuration, result in results.items():
        cm = np.asarray(result["confusion_matrix"])

        if cm.shape != (6, 6):
            raise ValueError(f"Invalid confusion matrix: {configuration}")

        if not np.array_equal(
            cm.sum(axis=1), baseline_cm.sum(axis=1)
        ):
            raise ValueError(
                f"Test activity counts differ: {configuration}"
            )

        if set(result["per_participant"]) != set(participant_ids):
            raise ValueError(
                f"Test participants differ: {configuration}"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    overall_rows = []
    activity_rows = []
    group_rows = []
    participant_group_rows = []

    for configuration, result in results.items():
        name = CONFIGURATIONS[configuration]
        per_activity = result["per_activity"]

        overall_rows.append({
            "Configuration": name,
            "Macro-F1": result["macro_f1"],
            "Accuracy": result["accuracy"],
            "Macro-precision": np.mean([
                per_activity[a]["precision"] for a in ACTIVITIES
            ]),
            "Macro-recall": np.mean([
                per_activity[a]["recall"] for a in ACTIVITIES
            ]),
            "Macro-F1 decrease from baseline":
                baseline["macro_f1"] - result["macro_f1"],
            "Accuracy decrease from baseline":
                baseline["accuracy"] - result["accuracy"],
            "Training time (s)": result["train_seconds"],
            "Prediction time (s)": result["predict_seconds"],
        })

        for activity in ACTIVITIES:
            scores = per_activity[activity]
            baseline_scores = baseline["per_activity"][activity]

            row = {
                "Configuration": name,
                "Activity": activity.replace("_", " "),
                "Support": int(scores["support"]),
            }

            for key, label in METRICS.items():
                row[label] = scores[key]
                row[f"{label} decrease from baseline"] = (
                    baseline_scores[key] - scores[key]
                )

            activity_rows.append(row)

        for group, activities in ACTIVITY_GROUPS.items():
            scores = group_scores(per_activity, activities)
            baseline_scores = group_scores(
                baseline["per_activity"], activities
            )

            row = {
                "Configuration": name,
                "Activity group": group,
            }

            for label in METRICS.values():
                row[f"Macro-{label.lower()}"] = scores[label]
                row[f"{label} decrease from baseline"] = (
                    baseline_scores[label] - scores[label]
                )

            group_rows.append(row)

            for participant in participant_ids:
                participant_scores = group_scores(
                    result["per_participant_activity"][participant],
                    activities,
                )
                participant_baseline = group_scores(
                    baseline["per_participant_activity"][participant],
                    activities,
                )

                row = {
                    "Participant ID": int(participant),
                    "Configuration": name,
                    "Activity group": group,
                }

                for label in METRICS.values():
                    row[f"Macro-{label.lower()}"] = (
                        participant_scores[label]
                    )
                    row[f"{label} decrease from baseline"] = (
                        participant_baseline[label]
                        - participant_scores[label]
                    )

                participant_group_rows.append(row)

    save_table(pd.DataFrame(overall_rows), "overall_results.csv")

    activity_table = pd.DataFrame(activity_rows)
    save_table(activity_table, "per_activity_results.csv")

    # Compact table we can c&p into the report.
    configuration_names = [
        CONFIGURATIONS[c] for c in results
    ]
    activity_comparison = activity_table.pivot(
        index="Activity",
        columns="Configuration",
        values="F1",
    ).reindex(
        index=[a.replace("_", " ") for a in ACTIVITIES],
        columns=configuration_names,
    )
    save_table(
        activity_comparison.reset_index(),
        "activity_f1_comparison.csv",
    )

    save_table(
        pd.DataFrame(group_rows),
        "activity_group_comparison.csv",
    )

    participant_rows = []
    for participant in participant_ids:
        row = {"Participant ID": int(participant)}
        for configuration, result in results.items():
            row[CONFIGURATIONS[configuration]] = (
                result["per_participant"][participant]["macro_f1"]
            )
        participant_rows.append(row)

    save_table(
        pd.DataFrame(participant_rows),
        "participant_f1_comparison.csv",
    )
    save_table(
        pd.DataFrame(participant_group_rows),
        "participant_group_results.csv",
    )

    # Optional comparison from the broader hypothesis:
    # does removing both acceleration groups hurt more than
    # removing body acceleration alone? The code for these is already there and was ran.
    # So might as well include them in the csv files. Can decide to remove if we 
    # have the time and energy. Will need to write more in report tho.
    body_only = "without_body_acceleration"
    both = "without_both_accelerations"

    if body_only in results and both in results:
        comparison_rows = []

        for group, activities in ACTIVITY_GROUPS.items():
            body_scores = group_scores(
                results[body_only]["per_activity"], activities
            )
            both_scores = group_scores(
                results[both]["per_activity"], activities
            )

            for metric in METRICS.values():
                comparison_rows.append({
                    "Activity group": group,
                    "Metric": f"Macro-{metric.lower()}",
                    "Without body acceleration": body_scores[metric],
                    "Without both acceleration groups": both_scores[metric],
                    "Additional decrease when removing both":
                        body_scores[metric] - both_scores[metric],
                })

        save_table(
            pd.DataFrame(comparison_rows),
            "both_vs_body_only.csv",
        )


if __name__ == "__main__":
    main()