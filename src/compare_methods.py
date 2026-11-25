# Create comparison tables and figures from final held-out test results.

# Run from the repository root:    python -m src.compare_methods

# filename: compare_methods.py


import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save figures without opening windows.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
OUTPUT = EXPERIMENTS / "comparison"

METHODS = {
    "Logistic classifier": "logistic",
    "PCA with RBF-SVM": "pca_rbf_svm",
    "1D CNN": "one_d_cnn",
}

COLORS = {
    "Logistic classifier": "#0072B2",
    "PCA with RBF-SVM": "#E69F00",
    "1D CNN": "#009E73",
}

ACTIVITIES = [
    "Walking",
    "Upstairs",
    "Downstairs",
    "Sitting",
    "Standing",
    "Lying down",
]

SUMMARY_COLUMNS = [
    "Method",
    "Macro-F1 (primary)",
    "Accuracy",
    "Macro-precision",
    "Macro-recall",
    "Training time (s)",
    "Prediction time (s)",
]


def macro_precision_recall(cm):
    """Compute macro averages across all six classes."""
    correct = np.diag(cm)
    predicted_counts = cm.sum(axis=0)
    actual_counts = cm.sum(axis=1)

    precision = np.divide(
        correct,
        predicted_counts,
        out=np.zeros_like(correct, dtype=float),
        where=predicted_counts != 0,
    )
    recall = np.divide(
        correct,
        actual_counts,
        out=np.zeros_like(correct, dtype=float),
        where=actual_counts != 0,
    )
    return float(precision.mean()), float(recall.mean())


def save_table(df, filename):
    """Save full-precision CSV and report-ready Markdown."""
    df.to_csv(OUTPUT / f"{filename}.csv", index=False)

    def format_cell(value, column):
        if pd.isna(value):
            return "—"
        if column == "Participant ID":
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            decimals = 2 if "time (s)" in column else 4
            return f"{value:.{decimals}f}"
        return str(value)

    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]

    for row in df.itertuples(index=False, name=None):
        cells = [
            format_cell(value, column)
            for value, column in zip(row, df.columns)
        ]
        lines.append("| " + " | ".join(cells) + " |")

    text = "\n".join(lines) + "\n"
    (OUTPUT / f"{filename}.md").write_text(text, encoding="utf-8")
    print(text)


def save_figure(fig, filename):
    fig.savefig(OUTPUT / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    split_path = ROOT / "data" / "splits" / "participants.json"
    with split_path.open(encoding="utf-8") as f:
        test_ids = sorted(json.load(f)["test_participants"])

    results = {}
    summary_rows = []

    for method, folder in METHODS.items():
        path = EXPERIMENTS / folder / "final_test_metrics.json"

        if not path.is_file():
            print(f"Missing: {path.relative_to(ROOT)}")
            summary_rows.append({"Method": method})
            continue

        with path.open(encoding="utf-8") as f:
            metrics = json.load(f)

        cm = np.asarray(metrics["confusion_matrix"], dtype=float)

        if (
            cm.shape != (6, 6)
            or not np.isfinite(cm).all()
            or (cm < 0).any()
            or not np.equal(cm, np.floor(cm)).all()
            or cm.sum() == 0
        ):
            raise ValueError(f"{path}: expected a nonempty 6 × 6 count matrix.")

        participant_ids = {
            int(pid) for pid in metrics["per_participant"]
        }
        if participant_ids != set(test_ids):
            raise ValueError(
                f"{path}: participant IDs do not match the test split."
            )

        precision, recall = macro_precision_recall(cm)
        results[method] = {"metrics": metrics, "cm": cm}

        summary_rows.append({
            "Method": method,
            "Macro-F1 (primary)": metrics["macro_f1"],
            "Accuracy": metrics["accuracy"],
            "Macro-precision": precision,
            "Macro-recall": recall,
            "Training time (s)": metrics["train_seconds"],
            "Prediction time (s)": metrics["predict_seconds"],
        })

    # Methods evaluated on the same test windows should have identical
    # counts of actual examples in each class.
    if results:
        reference = next(iter(results.values()))["cm"].sum(axis=1)
        for method, result in results.items():
            if not np.array_equal(result["cm"].sum(axis=1), reference):
                raise ValueError(
                    f"{method}: actual class counts differ across methods."
                )

    # Table 1: overall held-out test results.
    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    save_table(summary, "overall_test_results")

    # Table 2: macro-F1 for each held-out participant.
    participant_table = pd.DataFrame({"Participant ID": test_ids})

    for method in METHODS:
        column = f"{method} macro-F1"
        if method in results:
            per_participant = results[method]["metrics"]["per_participant"]
            participant_table[column] = [
                per_participant[str(pid)]["macro_f1"]
                for pid in test_ids
            ]
        else:
            participant_table[column] = np.nan

    save_table(participant_table, "per_participant_macro_f1")

    if not results:
        print("No final test results found; saved empty table templates.")
        return

    available = list(results)
    available_summary = summary.set_index("Method").loc[available]

    # Figure 1: compare overall classification metrics.
    score_columns = SUMMARY_COLUMNS[1:5]
    x = np.arange(len(score_columns))
    width = 0.8 / len(available)

    fig, ax = plt.subplots(figsize=(10, 5))

    for index, method in enumerate(available):
        offset = (index - (len(available) - 1) / 2) * width
        values = available_summary.loc[method, score_columns].to_numpy(
            dtype=float
        )
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=method,
            color=COLORS[method],
        )
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Macro-F1", "Accuracy", "Macro-precision", "Macro-recall"]
    )
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Score")
    ax.set_title("Held-out test performance")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, "overall_test_scores.png")

    # Figure 2: participant comparison.
    # Points are not connected because participant IDs are categorical.
    x = np.arange(len(test_ids))
    fig, ax = plt.subplots(figsize=(10, 5))

    for index, method in enumerate(available):
        offset = (index - (len(available) - 1) / 2) * 0.18
        ax.scatter(
            x + offset,
            participant_table[f"{method} macro-F1"],
            label=method,
            color=COLORS[method],
            s=55,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(test_ids)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Test participant ID")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Held-out test performance by participant")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, "per_participant_macro_f1.png")

    # Figure 3: confusion matrices with a shared color scale.
    # Color represents percentage within the actual class.
    # Cell annotations show both counts and percentages.
    fig, axes = plt.subplots(
        1,
        len(available),
        figsize=(6 * len(available), 5.5),
        squeeze=False,
        layout="constrained",
    )

    for ax, method in zip(axes[0], available):
        cm = results[method]["cm"]
        row_totals = cm.sum(axis=1, keepdims=True)
        normalized = np.divide(
            cm,
            row_totals,
            out=np.zeros_like(cm),
            where=row_totals != 0,
        )

        image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)

        for row in range(6):
            for column in range(6):
                value = normalized[row, column]
                ax.text(
                    column,
                    row,
                    f"{int(cm[row, column])}\n{value:.1%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value > 0.5 else "black",
                )

        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels(ACTIVITIES, rotation=45, ha="right")
        ax.set_yticklabels(ACTIVITIES)
        ax.set_xlabel("Predicted activity")
        ax.set_ylabel("Actual activity")
        ax.set_title(method)

    fig.colorbar(
        image,
        ax=list(axes[0]),
        shrink=0.75,
        label="Proportion of actual class",
    )
    fig.suptitle("Held-out test confusion matrices")
    save_figure(fig, "test_confusion_matrices.png")

    print(f"Saved tables and figures to: {OUTPUT}")


if __name__ == "__main__":
    main()