## 3.2 PCA + RBF-SVM  

We trained a supervised RBF-kernel SVM on the UCI HAR dataset's 561 engineered activity-recognition features. Our preprocessing pipeline used training-only standardization followed by student-written PCA and `sklearn.svm.SVC(kernel="rbf", gamma="scale")`. The custom PCA code centered the data, computed the sample covariance matrix, obtained eigenvectors using `numpy.linalg.eigh`, sorted them by explained variance, and projected inputs onto the leading principal components. Validation and test inputs use the transformations learned **exclusively from the relevant training participants**.

The team fixed the official 21-participant training and 9-participant test split in `data/splits/participants.json` (seed 42). Within training, seven participant-grouped cross-validation folds hold out three people each; no participant appears on both sides of a fold. We considered PCA dimensions 40, 80, 160, and 240 and SVM penalty parameters `C` of 0.1, 1, 10, and 100, holding the RBF kernel and `gamma="scale"` fixed. We selected the configuration with the highest mean validation macro-F1, breaking ties toward fewer components and smaller C. We then refit on all 21 official training participants. As required by the group, metrics are computed with `src/metrics.py:evaluate()` and include macro-F1, accuracy, confusion matrix, per-person performance, and runtimes.

### Validation and Final Test Results

Using seven-fold participant-separated cross-validation, the best configuration was 240 PCA components with an SVM C value of 1.0. This produced a mean validation macro-F1 of 0.9229 and an out-of-fold macro-F1 of 0.9228.

After selecting the hyperparameters, we trained the final PCA + RBF-SVM model using all 21 training participants and evaluated it on the nine held-out test participants. The model achieved a test accuracy of **95.08%** and a macro-F1 score of **95.00%**. These results suggest that the model generalized well to unseen participants. The test set was evaluated only after the hyperparameters were finalized, and no further tuning was performed using the test results.

Use `cv_grid_summary.csv` for the tuning table and `validation_confusion.png` and `validation_per_participant.png` for discussion. Distinguish validation (used in model selection) from final held-out test results. The CNN uses sensor sequences while logistic and PCA-SVM use the engineered features; therefore, the three-method comparison evaluates the full pipelines and not just individual classifiers on identical inputs.

### Contribution statement

The PCA + RBF-SVM contribution includes `CustomPCA` with NumPy covariance/eigendecomposition/projection and the grouped cross-validation, RBF-SVM training, hyperparameter selection, shared-metrics integration, and reproducibility outputs in `src/pca_svm_custom_pca.py`, `src/pca_svm_har_data.py`, and `src/pca_rbf_svm.py`. Both standardization and PCA are fitted anew on the training participants of each validation fold, protecting the validation data from leakage. The official test set is held out until the group finishes tuning and freezes the experimental design.

### Limitations

The selected PCA dimensionality of 240 components lies at the upper boundary of our tested grid. Although it achieved a higher mean validation macro-F1 than 160 components, the difference was approximately 0.005, smaller than the observed fold-to-fold variability. Therefore, we cannot conclude that 240 components is globally optimal, and further dimensionality testing could be explored in future work.
