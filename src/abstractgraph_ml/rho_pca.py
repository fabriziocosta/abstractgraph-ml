import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


class RhoPCA(BaseEstimator, TransformerMixin):
    """
    rhoPCA transformer.

    Fits generalized eigenvectors solving:

        Sigma_target v = lambda (Sigma_background + reg I) v

    and transforms samples by projection onto the top generalized eigenvectors.

    Feature importance is computed as:

        I_j = sum_i lambda_i * v_{j,i}^2

    over the selected top-k directions.
    """

    def __init__(
        self,
        n_components=2,
        reg=1e-6,
        standardize=True,
        center=True,
        target_label=1,
        background_label=0,
        sort_descending=True,
    ):
        self.n_components = n_components
        self.reg = reg
        self.standardize = standardize
        self.center = center
        self.target_label = target_label
        self.background_label = background_label
        self.sort_descending = sort_descending

    def fit(self, X, y):
        """
        Fit rhoPCA.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Input feature matrix.

        y : array-like, shape (n_samples,)
            Binary labels. Samples with y == target_label form the target set.
            Samples with y == background_label form the background set.

        Returns
        -------
        self
        """
        X = self._to_dense(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D matrix.")

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have incompatible shapes.")

        Xt = X[y == self.target_label]
        Xb = X[y == self.background_label]

        if Xt.shape[0] < 2:
            raise ValueError("Target set must contain at least 2 samples.")

        if Xb.shape[0] < 2:
            raise ValueError("Background set must contain at least 2 samples.")

        if self.n_components > X.shape[1]:
            raise ValueError("n_components cannot exceed the number of features.")

        self.scaler_target_ = None
        self.scaler_background_ = None
        self.scaler_transform_ = None

        if self.standardize:
            self.scaler_target_ = StandardScaler(
                with_mean=self.center,
                with_std=True,
            )
            self.scaler_background_ = StandardScaler(
                with_mean=self.center,
                with_std=True,
            )

            Xt_scaled = self.scaler_target_.fit_transform(Xt)
            Xb_scaled = self.scaler_background_.fit_transform(Xb)

            # Used for transforming arbitrary future samples.
            self.scaler_transform_ = StandardScaler(
                with_mean=self.center,
                with_std=True,
            )
            X_scaled = self.scaler_transform_.fit_transform(X)
        else:
            Xt_scaled = self._center_if_needed(Xt)
            Xb_scaled = self._center_if_needed(Xb)
            X_scaled = self._center_if_needed(X)

        self.mean_ = X.mean(axis=0) if self.center else np.zeros(X.shape[1])

        Sigma_t = self._covariance(Xt_scaled)
        Sigma_b = self._covariance(Xb_scaled)

        n_features = X.shape[1]
        Sigma_b_reg = Sigma_b + self.reg * np.eye(n_features)

        eigvals, eigvecs = eigh(Sigma_t, Sigma_b_reg)

        if self.sort_descending:
            order = np.argsort(eigvals)[::-1]
        else:
            order = np.argsort(eigvals)

        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        self.eigenvalues_ = eigvals
        self.components_ = eigvecs[:, : self.n_components]
        self.selected_eigenvalues_ = eigvals[: self.n_components]

        self.feature_importances_ = self.compute_feature_importance()
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X):
        """
        Project X onto the learned rhoPCA directions.
        """
        check_is_fitted(self, ["components_", "n_features_in_"])

        X = self._to_dense(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

        if self.standardize:
            X = self.scaler_transform_.transform(X)
        else:
            X = X - self.mean_ if self.center else X

        return X @ self.components_

    def fit_transform(self, X, y):
        """
        Fit rhoPCA and return transformed X.
        """
        return self.fit(X, y).transform(X)

    def compute_feature_importance(self, normalize=True):
        """
        Compute feature importance using:

            I_j = sum_i lambda_i * v_{j,i}^2

        Parameters
        ----------
        normalize : bool
            If True, normalize importances to sum to 1.

        Returns
        -------
        importances : ndarray, shape (n_features,)
        """
        check_is_fitted(self, ["components_", "selected_eigenvalues_"])

        V = self.components_
        lambdas = self.selected_eigenvalues_

        importances = np.sum((V ** 2) * lambdas[np.newaxis, :], axis=1)

        # Numerical safety: remove tiny negative artifacts if eigenvalues are odd.
        importances = np.maximum(importances, 0.0)

        if normalize:
            total = importances.sum()
            if total > 0:
                importances = importances / total

        return importances

    def get_top_features(self, feature_names=None, top_k=20):
        """
        Return top-k features by rhoPCA importance.
        """
        check_is_fitted(self, ["feature_importances_"])

        importances = self.feature_importances_
        order = np.argsort(importances)[::-1][:top_k]

        if feature_names is None:
            feature_names = np.array([f"feature_{i}" for i in range(len(importances))])
        else:
            feature_names = np.asarray(feature_names)

        return [
            {
                "rank": rank + 1,
                "feature_index": int(idx),
                "feature_name": str(feature_names[idx]),
                "importance": float(importances[idx]),
            }
            for rank, idx in enumerate(order)
        ]

    @staticmethod
    def _covariance(X):
        return (X.T @ X) / (X.shape[0] - 1)

    @staticmethod
    def _to_dense(X):
        if sparse.issparse(X):
            return X.toarray()
        return np.asarray(X, dtype=float)

    def _center_if_needed(self, X):
        if self.center:
            return X - X.mean(axis=0)
        return X