# Plot focused-study results from the generated CSV tables.
# This only plots the baseline and the two removal configurations (remove gyro and remove both acc.'s), not all five.

# Run from the repository root: python -m src.plot_focused_study

# plots saved under experiments/focused_study/figures, including a pdf with all plots.


from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = ROOT / "experiments" / "focused_study"
TABLES_DIR = STUDY_DIR / "tables"
FIGURES_DIR = STUDY_DIR / "figures"

CONFIGURATIONS = [
    "All inputs (baseline)",
    "Without angular velocity",
    "Without both acceleration groups",
]

DISPLAY_NAMES = [
    "All inputs\n(baseline)",
    "Without angular\nvelocity",
    "Without both\nacceleration groups",
]

ACTIVITIES = [
    "walking",
    "walking upstairs",
    "walking downstairs",
    "sitting",
    "standing",
    "lying down",
]

GROUPS = [
    "Walking activities",
    "Stationary postures",
]

CONFIGURATION_COLORS = ["#334155", "#0072B2", "#D55E00"]
GROUP_COLORS = ["#0072B2", "#E69F00"]

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
})


def format_score_axis(ax):
    ax.set_xlim(0, 1.1)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Score")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)


def save_figure(fig, filename, pdf):
    path = FIGURES_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_overall(overall, pdf):
    metrics = [
        "Macro-F1",
        "Accuracy",
        "Macro-precision",
        "Macro-recall",
    ]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 8),
        layout="constrained",
    )

    for ax, metric in zip(axes.flat, metrics):
        bars = ax.barh(
            DISPLAY_NAMES,
            overall[metric].to_numpy(),
            color=CONFIGURATION_COLORS,
            height=0.6,
        )
        ax.invert_yaxis()
        ax.bar_label(bars, fmt="%.4f", padding=4)
        ax.set_title(metric, fontweight="bold")
        format_score_axis(ax)

    fig.suptitle(
        "Overall test performance",
        fontsize=18,
        fontweight="bold",
    )
    save_figure(fig, "overall_scores.png", pdf)


def plot_activity_f1(activity, pdf):
    values = activity.to_numpy(dtype=float)

    fig, ax = plt.subplots(
        figsize=(10, 6),
        layout="constrained",
    )

    heatmap = ax.imshow(
        values,
        vmin=0,
        vmax=1,
        cmap="YlGnBu",
        aspect="auto",
    )

    ax.set_xticks(range(len(CONFIGURATIONS)), DISPLAY_NAMES)
    ax.set_yticks(
        range(len(ACTIVITIES)),
        [activity.capitalize() for activity in ACTIVITIES],
    )

    # Separate walking activities from stationary postures.
    ax.axhline(2.5, color="white", linewidth=3)

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            ax.text(
                column,
                row,
                f"{value:.4f}",
                ha="center",
                va="center",
                color="white" if value > 0.65 else "black",
                fontsize=12,
            )

    fig.colorbar(heatmap, ax=ax, label="F1 score")
    ax.set_title(
        "F1 by activity",
        fontsize=17,
        fontweight="bold",
        pad=16,
    )
    save_figure(fig, "activity_f1.png", pdf)


def plot_activity_groups(groups, pdf):
    metrics = [
        ("Macro-precision", "Macro-precision"),
        ("Macro-recall", "Macro-recall"),
        ("Macro-f1", "Macro-F1"),
    ]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 5),
        layout="constrained",
    )

    positions = np.arange(len(CONFIGURATIONS))
    bar_height = 0.32

    for ax, (column, title) in zip(axes, metrics):
        for group_index, group in enumerate(GROUPS):
            values = [
                groups.loc[(configuration, group), column]
                for configuration in CONFIGURATIONS
            ]
            offset = (group_index - 0.5) * 0.36

            bars = ax.barh(
                positions + offset,
                values,
                height=bar_height,
                color=GROUP_COLORS[group_index],
                label=group,
            )
            ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)

        ax.set_yticks(
            positions,
            DISPLAY_NAMES if ax is axes[0] else [""] * 3,
        )
        ax.invert_yaxis()
        ax.set_title(title, fontweight="bold")
        format_score_axis(ax)

    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
    )

    fig.suptitle(
        "Walking activities versus stationary postures",
        fontsize=17,
        fontweight="bold",
    )
    save_figure(fig, "activity_group_scores.png", pdf)


def plot_f1_decreases(groups, pdf):
    # The baseline itself has a decrease of zero, so omit it here.
    configurations = CONFIGURATIONS[1:]
    positions = np.arange(len(configurations))

    fig, ax = plt.subplots(
        figsize=(10, 5),
        layout="constrained",
    )

    all_values = []

    for group_index, group in enumerate(GROUPS):
        # Convert differences on the 0–1 scale to percentage points.
        values = np.array([
            groups.loc[
                (configuration, group),
                "F1 decrease from baseline",
            ]
            for configuration in configurations
        ]) * 100

        all_values.extend(values)
        offset = (group_index - 0.5) * 0.36

        bars = ax.barh(
            positions + offset,
            values,
            height=0.32,
            color=GROUP_COLORS[group_index],
            label=group,
        )
        ax.bar_label(bars, fmt="%.2f", padding=4)

    ax.set_yticks(positions, DISPLAY_NAMES[1:])
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)

    ax.set_xlim(
        min(0, min(all_values)) - 2,
        max(0, max(all_values)) + 8,
    )
    ax.set_xlabel(
        "Baseline F1 minus configuration F1 (percentage points)\n"
        "Positive = worse; negative = better"
    )
    ax.set_title(
        "F1 decrease from the all-input baseline",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
    )
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)

    save_figure(fig, "f1_decreases.png", pdf)


def plot_execution_times(overall, pdf):
    metrics = [
        ("Training time (s)", "Training time"),
        ("Prediction time (s)", "Prediction time"),
    ]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 5),
        layout="constrained",
    )

    for ax, (column, title) in zip(axes, metrics):
        values = overall[column].to_numpy()

        bars = ax.barh(
            DISPLAY_NAMES,
            values,
            color=CONFIGURATION_COLORS,
            height=0.6,
        )
        ax.invert_yaxis()
        ax.bar_label(bars, fmt="%.4f", padding=4)
        ax.set_xlim(0, max(values) * 1.25)
        ax.set_xlabel("Seconds")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)

    save_figure(fig, "execution_times.png", pdf)


def main():
    overall = pd.read_csv(
        TABLES_DIR / "overall_results.csv"
    ).set_index("Configuration")

    activity = pd.read_csv(
        TABLES_DIR / "activity_f1_comparison.csv"
    ).set_index("Activity")

    groups = pd.read_csv(
        TABLES_DIR / "activity_group_comparison.csv"
    ).set_index(["Configuration", "Activity group"])

    # Select only the baseline and the two removal configurations.
    # .loc also raises an error if a required result is missing.
    overall = overall.loc[CONFIGURATIONS]
    activity = activity.loc[ACTIVITIES, CONFIGURATIONS]

    expected_groups = pd.MultiIndex.from_product(
        [CONFIGURATIONS, GROUPS],
        names=["Configuration", "Activity group"],
    )
    groups = groups.loc[expected_groups]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES_DIR / "focused_study_plots.pdf"

    with PdfPages(pdf_path) as pdf:
        plot_overall(overall, pdf)
        plot_activity_f1(activity, pdf)
        plot_activity_groups(groups, pdf)
        plot_f1_decreases(groups, pdf)
        plot_execution_times(overall, pdf)

    print(f"Saved combined PDF: {pdf_path}")


if __name__ == "__main__":
    main()