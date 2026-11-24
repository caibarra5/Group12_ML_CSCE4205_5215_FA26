 

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class CustomPCA(TransformerMixin, BaseEstimator):
    """PCA via centering, covariance matrix, eigendecomposition, and projection.

    Compatible with sklearn Pipeline/GridSearchCV, including clone()/get_params().
    Parameters
    ----------
    n_components : int
        Fixed number of principal components to retain, chosen using training CV.
    """

    def __init__(self, n_components: int = 100):
        self.n_components = n_components

    @staticmethod
    def _check_X(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 1 or X.shape[1] < 1:
            raise ValueError("X must be a nonempty 2-D array")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinity")
        return X

    def fit(self, X, y=None):
        X = self._check_X(X)
        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError("PCA needs at least 2 training examples")
        if (isinstance(self.n_components, bool)
                or not isinstance(self.n_components, (int, np.integer))
                or not 1 <= self.n_components <= min(n_samples, n_features)):
            raise ValueError(
                "n_components must be an integer from 1 through min(n_samples, n_features)"
            )

        self.n_features_in_ = n_features
        self.mean_ = X.mean(axis=0)              # TRAINING observations only
        centered = X - self.mean_
        covariance = centered.T @ centered / (n_samples - 1)
        # eigh for the real symmetric covariance matrix; eigenvectors are columns.
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.clip(eigenvalues[indices], 0.0, None)
        eigenvectors = eigenvectors[:, indices]

        k = int(self.n_components)
        self.components_ = eigenvectors[:, :k].T  # shape (k, n_features)
        self.explained_variance_ = eigenvalues[:k]
        total_variance = eigenvalues.sum()
        self.explained_variance_ratio_ = (
            eigenvalues[:k] / total_variance
            if total_variance > 0 else np.zeros(k, dtype=np.float64)
        )
        self.singular_values_ = np.sqrt(eigenvalues[:k] * (n_samples - 1))
        return self

    def transform(self, X):
        check_is_fitted(self, ["mean_", "components_"])
        X = self._check_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} input features")
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, Z):
        check_is_fitted(self, ["mean_", "components_"])
        Z = self._check_X(Z)
        if Z.shape[1] != self.components_.shape[0]:
            raise ValueError("Projected dimension does not match PCA components")
        return Z @ self.components_ + self.mean_

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ["components_"])
        return np.asarray([f"PC{i + 1}" for i in range(self.components_.shape[0])])
