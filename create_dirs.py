from pathlib import Path

folders = [
    "data/raw",             # Original dataset; keep unchanged
    "data/processed",       # Cleaned data and extracted features
    "data/splits",          # Participant-separated train/validation/test IDs
    "notebooks",            # Exploration and preliminary experiments
    "src",                  # Reusable preprocessing, training, evaluation code
    "configs",              # Model settings and experiment parameters
    "experiments/comparison",     # Results for the three-method comparison
    "experiments/focused_study",  # Controlled experiments with one method
    "models",               # Saved trained models
    "reports/figures",       # Plots for the report and presentation
    "reports/tables",        # Performance metrics and comparison tables
    "reports/drafts",        # Working report drafts
    "presentations",         # Slides
    "docs",                 # Instructions, team notes, and references
]

for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)

print("Project directories created.")

for folder in folders:
    (Path(folder) / ".gitkeep").touch(exist_ok=True)

print(".gitkeep files created in all project directories.")