# Human Activity Recognition

Machine learning group project for **CSCE 4205 / CSCE 5215 — Fall 2026**.

## Overview
Classify six human activities from smartphone accelerometer and gyroscope data using the [UCI HAR dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones), which includes 10,299 samples from 30 participants and 561 engineered features.

## Objectives
- Train and compare at least three machine learning methods.
- Select one method for a focused, controlled experiment.
- Evaluate generalization to participants unseen during training.

## Evaluation
Use participant-separated data splits, with **macro-F1** as the primary metric. Also report accuracy, confusion matrices, performance by participant, and training/prediction times. Fit learned preprocessing on training data only and reserve the test set for final evaluation.

## Deliverables
- **Proposal:** October 30, 2026
- **Presentation:** November 30, 2026
- **Final report and code:** December 4, 2026

Final code will include environment requirements, split definitions, random seeds, reproduction instructions, and team contribution statements.

## Environment
Create a virtual environment in the repo root, activate it, then install the requirements:

```
python -m venv venv
venv\Scripts\activate        # Windows (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

Add a package to `requirements.txt` when your code starts importing it. The `venv/` folder is ignored by git.

## Data
The dataset is not stored in git. Download it from the [UCI HAR dataset page](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones), unzip it, and place the contents in `data/raw/UCI HAR Dataset/` so that `train/` and `test/` sit directly inside it.

## Shared participant split
Every method uses the same split, defined in `data/splits/participants.json` and created by `src/make_splits.py` (seed 42).
- **Test:** the dataset's 9 test participants. Used only for the final evaluation.
- **Training:** the 21 training participants, divided into 7 cross-validation folds of 3 participants each. Every fold holds out whole participants.

Load this file instead of re-creating the split.

