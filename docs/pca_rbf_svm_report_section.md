## 3.2 PCA + RBF-SVM  

We trained a supervised RBF-kernel SVM on the UCI HAR dataset's 561 engineered activity-recognition features. Our preprocessing pipeline used training-only standardization followed by student-written PCA and `sklearn.svm.SVC(kernel="rbf", gamma="scale")`. The custom PCA code centered the data, computed the sample covariance matrix, obtained eigenvectors using `numpy.linalg.eigh`, sorted them by explained variance, and projected inputs onto the leading principal components. Validation and test inputs use the transformations learned **exclusively from the relevant training participants**.

The team fixed the official 21-participant training and 9-participant test split in `data/splits/participants.json` (seed 42). Within training, seven participant-grouped cross-validation folds hold out three people each; no participant appears on both sides of a fold. We considered PCA dimensions 40, 80, 160, and 240 and SVM penalty parameters `C` of 0.1, 1, 10, and 100, holding the RBF kernel and `gamma="scale"` fixed. We selected the configuration with the highest mean validation macro-F1, breaking ties toward fewer components and smaller C. We then refit on all 21 official training participants. As required by the group, metrics are computed with `src/metrics.py:evaluate()` and include macro-F1, accuracy, confusion matrix, per-person performance, and runtimes.

### Validation and Final Test Results

Using seven-fold participant-separated cross-validation, the best configuration was 240 PCA components with an SVM C value of 1.0. This produced a mean validation macro-F1 of 0.9229 and an out-of-fold macro-F1 of 0.9228.

The seven validation folds had a macro-F1 standard deviation of 0.026 and the out-of-fold accuracy was 0.9248. The 240 retained components explain 99.7% of the variance in the training data.

After selecting the hyperparameters, we trained the final PCA + RBF-SVM model using all 21 training participants and evaluated it once on the nine held-out test participants. The model achieved a test accuracy of **95.08%**, a macro-F1 score of **95.00%**, a macro-precision of 0.9515 and a macro-recall of 0.9492 (2,947 test windows). The test set was evaluated only after the hyperparameters were finalized, and no further tuning was performed using the test results.

**Error analysis.** The model misclassified 145 of the 2,947 test windows. About half of them (74) are SITTING and STANDING mixed up with each other: 45 SITTING windows were predicted as STANDING and 29 STANDING windows were predicted as SITTING. The next largest group is WALKING_DOWNSTAIRS predicted as WALKING_UPSTAIRS (29 windows). All 537 LAYING windows were classified correctly.

**Performance by participant.** Test macro-F1 ranges from 0.881 (participant 9) to 0.992 (participant 24). Four of the nine participants are below 0.95: participants 9 (0.881), 10 (0.907), 20 (0.903) and 18 (0.946). The test macro-F1 (0.950) is higher than the validation macro-F1 (0.923). With only nine test participants, this difference may reflect which people ended up in the test group, so we treat the test result as an estimate with real uncertainty and not as a guarantee for any new person.

**Training and prediction time.** The final model (standardization, PCA and SVM together) trained in 0.65 s and predicted all 2,947 test windows in 0.57 s (about 0.19 ms per window). These were measured on Windows 11 with an Intel CPU (16 logical cores) and Python 3.13.5, as recorded in `run_summary.json`. Timings on other machines will differ.

Use `cv_grid_summary.csv` for the tuning table and `validation_confusion.png` and `validation_per_participant.png` for discussion. Distinguish validation (used in model selection) from final held-out test results. The CNN uses sensor sequences while logistic and PCA-SVM use the engineered features; therefore, the three-method comparison evaluates the full pipelines and not just individual classifiers on identical inputs.

### Contribution statement

The PCA + RBF-SVM contribution includes `CustomPCA` with NumPy covariance/eigendecomposition/projection and the grouped cross-validation, RBF-SVM training, hyperparameter selection, shared-metrics integration, and reproducibility outputs in `src/pca_svm_custom_pca.py`, `src/pca_svm_har_data.py`, and `src/pca_rbf_svm.py`. Both standardization and PCA are fitted anew on the training participants of each validation fold, protecting the validation data from leakage. The official test set was held out until the group finished tuning and froze the experimental design, and it was then evaluated once with `--final-test`.

### Limitations

The selected PCA dimensionality of 240 components lies at the upper boundary of our tested grid. Although it achieved a higher mean validation macro-F1 than 160 components, the difference from the best 160-component setting was approximately 0.003 (0.9229 versus 0.9195), much smaller than the fold-to-fold standard deviation of about 0.026. Therefore, we cannot conclude that 240 components is globally optimal, and further dimensionality testing could be explored in future work. The test set also contains only nine participants, so the final test scores are uncertain too (see the per-participant results above).
