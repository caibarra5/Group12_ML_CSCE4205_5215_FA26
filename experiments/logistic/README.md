# Logistic classifier: tuning record

The logistic classifier is written by hand (one-vs-rest, trained with gradient descent). The code is in `models/logistic_regression/logistic_regression.py` and the test results are in `final_test_metrics.json` in this folder.

## How the settings were picked

The learning rate and the number of steps were picked with the seven cross-validation folds on the training participants, not with the test set. Each fold works out its own standardization from its own training people only. Settings were scored by mean cross-validation accuracy (not macro-F1).

- **Round 1:** learning rates 0.01, 0.03 and 0.1 with 200, 500 and 1000 steps. The best setting sat at the edge of the grid, so the grid was widened.
- **Round 2:** learning rates 0.03 and 0.1 with 1000, 2000 and 3000 steps.
- **Best setting:** learning rate 0.1 with 3000 steps, mean cross-validation accuracy 0.921.
- **Edge of the grid:** 3000 is the largest step count tried. The last 1000 steps added about 0.3 points, so the gains had nearly stopped.

Only the best setting's cross-validation score is recorded here. The scores for the other combinations are printed to the console when the script runs.

## Test results as the code changed

`final_test_metrics.json` was regenerated as the code changed, so older numbers show up in the git history.

| Version | Macro-F1 | Accuracy |
|---|---|---|
| First version: no standardization, learning rate 0.01, 200 steps | 0.837 | 0.848 |
| After adding standardization, before cross-validation tuning | 0.907 | 0.909 |
| Final: standardization plus tuned settings (learning rate 0.1, 3000 steps) | 0.953 | 0.952 |

The final version trains in about 15 seconds. The settings for the final version came from cross-validation only.

## To run

`python models/logistic_regression/logistic_regression.py` (needs the data in `data/raw`).
