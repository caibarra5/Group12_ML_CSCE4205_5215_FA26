## 3.2 PCA + RBF-SVM — add real results after running

We trained a supervised RBF-kernel SVM on the UCI HAR dataset's 561 engineered activity-recognition features. Our preprocessing pipeline used training-only standardization followed by student-written PCA and `sklearn.svm.SVC(kernel="rbf", gamma="scale")`. The custom PCA code centered the data, computed the sample covariance matrix, obtained eigenvectors using `numpy.linalg.eigh`, sorted them by explained variance, and projected inputs onto the leading principal components. Validation and test inputs use the transformations learned **exclusively from the relevant training participants**.

The team fixed the official 21-participant training and 9-participant test split in `data/splits/participants.json` (seed 42). Within training, seven participant-grouped cross-validation folds hold out three people each; no participant appears on both sides of a fold. We considered PCA dimensions 40, 80, 160, and 240 and SVM penalty parameters `C` of 0.1, 1, 10, and 100, holding the RBF kernel and `gamma="scale"` fixed. We selected the configuration with the highest mean validation macro-F1, breaking ties toward fewer components and smaller C. We then refit on all 21 official training participants. As required by the group, metrics are computed with `src/metrics.py:evaluate()` and include macro-F1, accuracy, confusion matrix, per-person performance, and runtimes.

**After running official data, replace these placeholders ONLY with saved results from `experiments/pca_rbf_svm/`**:

- Selected PCA components: **[read `run_summary.json`: selected_pca_components]**
- Selected SVM C: **[read `run_summary.json`: selected_C]**
- Mean seven-fold validation macro-F1: **[mean_validation_macro_f1]**
- Fold SD: **[validation_sd_macro_f1]**
- Out-of-fold validation accuracy: **[validation_metrics.json`: accuracy]**
- Final fit time: **[final_fit_seconds]** on **[CPU/system from environment]**
- Training-only retained PCA variance: **[final_training_explained_variance]**
- Held-out test macro-F1 and accuracy: **[only after all methods and experimental choices are locked; read `final_test_metrics.json`]**

Use `cv_grid_summary.csv` for the tuning table and `validation_confusion.png` and `validation_per_participant.png` for discussion. Distinguish validation (used in model selection) from final held-out test results. The CNN uses sensor sequences while logistic and PCA-SVM use the engineered features; therefore, the three-method comparison evaluates the full pipelines and not just individual classifiers on identical inputs.

### Contribution statement

The PCA + RBF-SVM contribution includes `CustomPCA` with NumPy covariance/eigendecomposition/projection and the grouped cross-validation, RBF-SVM training, hyperparameter selection, shared-metrics integration, and reproducibility outputs in `src/pca_svm_custom_pca.py`, `src/pca_svm_har_data.py`, and `src/pca_rbf_svm.py`. Both standardization and PCA are fitted anew on the training participants of each validation fold, protecting the validation data from leakage. The official test set is held out until the group finishes tuning and freezes the experimental design.
