"""
Graph estimator pipeline with optional feature selection and utilities.

Also exposes IsolationForestProba: a compact IsolationForest wrapper that
calibrates decision scores into probabilities via the empirical CDF of
training scores, providing a 2-column predict_proba compatible with
GraphEstimator and downstream selection.
"""

from copy import deepcopy
from typing import Any, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import issparse, vstack
from sklearn.base import BaseEstimator
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.utils.validation import check_is_fitted

from abstractgraph.vectorize import AbstractGraphTransformer

class IsolationForestProba:
    """IsolationForest with simple probability calibration.

    - Fits IsolationForest on features.
    - Calibrates decision_function scores (higher=inlier) to [0,1] using the
      empirical CDF of training scores.
    - Exposes ``classes_ = [0, 1]`` where 0=outlier and 1=inlier.

    This keeps the interface close to a classifier so that code relying on
    ``predict_proba`` and ``classes_`` continues to work in unsupervised mode.
    """

    def __init__(self, **iforest_kwargs) -> None:
        self.iforest_kwargs = dict(n_estimators=300, contamination="auto")
        self.iforest_kwargs.update(iforest_kwargs or {})
        self.model_: Optional[IsolationForest] = None
        self.sorted_scores_: Optional[np.ndarray] = None
        self.classes_ = np.array([0, 1])  # 0=outlier, 1=inlier

    def fit(self, X, y=None):
        self.model_ = IsolationForest(**self.iforest_kwargs)
        self.model_.fit(X)
        scores = self.model_.decision_function(X)  # higher => inlier
        self.sorted_scores_ = np.sort(scores)
        return self

    def _cdf(self, scores: np.ndarray) -> np.ndarray:
        ss = self.sorted_scores_
        if ss is None or ss.size == 0:
            return np.full_like(scores, 0.5, dtype=float)
        ranks = np.searchsorted(ss, scores, side="right")
        return ranks / float(ss.size)

    def predict(self, X):
        scores = self.model_.decision_function(X)
        return (scores >= 0.0).astype(int)

    def predict_proba(self, X):
        scores = self.model_.decision_function(X)
        p_in = self._cdf(scores)
        p_out = 1.0 - p_in
        return np.vstack([p_out, p_in]).T


class DropFirstTruncatedSVD:
    """Sparse-friendly default manifold that drops the leading component."""

    def __init__(self, n_components: int = 3, random_state: int = 0) -> None:
        self.n_components = int(n_components)
        self.random_state = random_state
        self.model_: Optional[TruncatedSVD] = None
        self.effective_n_components_: Optional[int] = None

    def fit(self, X, y=None):
        n_features = int(X.shape[1])
        if n_features < 2:
            self.model_ = None
            self.effective_n_components_ = 0
            return self
        effective_n_components = max(1, min(self.n_components, n_features))
        self.model_ = TruncatedSVD(
            n_components=effective_n_components,
            random_state=self.random_state,
        )
        self.model_.fit(X, y)
        self.effective_n_components_ = effective_n_components
        return self

    def transform(self, X):
        if self.model_ is None:
            return X.toarray() if issparse(X) else np.asarray(X)
        transformed = self.model_.transform(X)
        if transformed.shape[1] > 1:
            return transformed[:, 1:]
        return transformed


class GraphEstimator(BaseEstimator):
    """
    A lightweight pipeline for graph learning with optional feature selection.

    The class mirrors scikit‑learn's estimator API and chains together three
    stages:

    1) ``transformer``: an ``AbstractGraphTransformer`` that converts input
       graphs into a 2D feature matrix.
    2) ``manifold`` (optional): a dimensionality‑reduction (or any transformer
       with ``fit``/``transform``) applied to features. Defaults to a
       sparse-friendly ``TruncatedSVD`` wrapper that drops the first component.
    3) ``estimator`` (optional): a downstream predictor/classifier/regressor
       trained on the raw features (pre‑manifold).

    Feature selection (optional) can be enabled via ``n_selected_features`` and
    is driven by the downstream estimator's feature importances or coefficients
    (when available). Selected features are used to fit and apply the manifold,
    while the downstream estimator is always trained and evaluated on the raw
    features to avoid feedback loops.

    Parameters
    ----------
    transformer : AbstractGraphTransformer
        Fully specified transformer that converts graphs into feature matrices.
    estimator : Optional[sklearn.base.BaseEstimator], default None
        A scikit‑learn compatible estimator. If it exposes either
        ``feature_importances_`` (e.g., tree‑based models) or ``coef_`` (e.g.,
        linear models), those signals are used to rank features when
        ``n_selected_features`` is not ``None``. If absent, feature selection is
        skipped silently.
    manifold : Optional[Any], default sparse-friendly TruncatedSVD
        Any object providing ``fit``/``transform`` on feature matrices (e.g.,
        TruncatedSVD/UMAP/TSNE). When feature selection is active, the manifold
        is fit on the selected subset of features; otherwise on all raw
        features. The default manifold fits a sparse-friendly TruncatedSVD and
        returns components from the second onward.
    n_selected_features : Optional[int | float], default 0.1
        Controls optional feature selection based on the downstream estimator's
        importances/coefficients.
        - ``None``: disables feature selection (legacy behavior).
        - ``int``: select exactly that many top features.
        - ``float`` in ``(0, 1)``: select that fraction of the input features
          (rounded to at least 1) as determined by the transformer's output
          dimensionality.
        - ``float`` ``>= 1``: treated as an integer count (rounded).

        Notes on behavior:
        - Selection indices are computed after fitting the downstream estimator
          on the raw features. If the estimator does not expose an importance
          signal, selection is skipped and the full feature set is used.
        - Selection is applied before the manifold both at ``fit`` time
          (manifold.fit on selected features) and at ``transform`` time.
        - The downstream estimator is always trained and evaluated on the raw
          transformer features (no selection applied to ``predict``/``predict_proba``).

    Attributes
    ----------
    transformer_ : AbstractGraphTransformer
        Fitted copy of the provided transformer.
    estimator_ : Optional[BaseEstimator]
        Fitted copy of the downstream estimator (if provided).
    manifold_ : Optional[Any]
        Fitted copy of the manifold/transformer (if provided).
    selected_feature_indices_ : Optional[np.ndarray]
        Array of column indices selected by feature selection, or ``None`` when
        selection is disabled or importances are unavailable.

    Examples
    --------
    Train a model, fit TruncatedSVD on the top 20% most important features as
    given by a RandomForest, and visualize the 2D manifold embedding:

    >>> est = GraphEstimator(transformer=vec,
    ...                      estimator=RandomForestClassifier(random_state=0),
    ...                      manifold=TruncatedSVD(n_components=2, random_state=0),
    ...                      n_selected_features=0.2)
    >>> est.fit(graphs, y)
    >>> Z = est.transform(graphs)  # manifold on selected features
    >>> preds = est.predict(graphs)  # estimator uses raw features
    """

    def __init__(
        self,
        transformer: AbstractGraphTransformer,
        estimator: Optional[BaseEstimator] = None,
        manifold: Optional[Any] = None,
        n_selected_features: Optional[Union[int, float]] = None,
    ) -> None:
        """Initialize the graph estimator pipeline.

        Args:
            transformer: Graph-to-feature transformer.
            estimator: Downstream estimator to fit on raw features.
            manifold: Optional manifold/transformer applied after selection.
            n_selected_features: Optional feature selection specification.

        Returns:
            None.
        """
        self.transformer = transformer
        self.estimator = estimator
        self.manifold = manifold if manifold is not None else DropFirstTruncatedSVD()
        # Optional feature selection based on downstream estimator importance
        # None disables selection. If int -> select exactly that many features.
        # If float in (0,1) -> fraction of input feature count.
        # If float >= 1.0 -> treated as an integer count (rounded).
        self.n_selected_features = n_selected_features

    def fit(self, graphs, targets: Optional[Any] = None) -> "GraphEstimator":
        """Fit the transformer, estimator, and manifold.

        Supports both supervised and unsupervised modes:
        - Supervised: provide ``targets`` and an explicit downstream estimator.
        - Unsupervised: set ``targets=None``; if no estimator was provided,
          a default ``IsolationForestProba`` is used to expose predict_proba
          with outlier/inlier semantics (classes_ = [0, 1]).

        Args:
            graphs: Input graphs.
            targets: Optional target labels. When ``None``, unsupervised mode is used
                and an ``IsolationForestProba`` is employed if no estimator is set.

        Returns:
            Self for chaining.
        """
        self._initialize_fit_state()

        raw_features = self.transformer_.fit_transform(graphs, targets)
        self._fit_estimator(raw_features, targets)
        self._update_selection_and_manifold(raw_features, targets)
        return self._mark_fitted()

    def partial_fit(self, graphs, targets: Optional[Any] = None, **kwargs) -> "GraphEstimator":
        """Incrementally fit the downstream estimator when supported.

        This method keeps the fitted transformer state after the first batch so
        future updates can reuse ``transform(...)`` output directly. If the
        downstream estimator exposes ``partial_fit``, that is used directly.
        Otherwise, already-vectorized batches are cached and the estimator is
        refit on the concatenated feature matrix each time.

        Args:
            graphs: Input graphs.
            targets: Optional target labels or regression targets.
            **kwargs: Forwarded to the wrapped estimator's ``partial_fit``.

        Returns:
            Self for chaining.
        """
        fitted = self._is_graph_estimator_fitted()

        if not fitted:
            self._initialize_fit_state()
            raw_features = self.transformer_.fit_transform(graphs, targets)
        else:
            raw_features = self._transform_raw(graphs)

        estimator = self._ensure_estimator_for_fit(targets)
        raw_features = self._normalize_feature_matrix(raw_features)
        if hasattr(estimator, "partial_fit"):
            estimator.partial_fit(raw_features, targets, **kwargs)
        else:
            self._append_replay_batch(raw_features, targets)
            estimator.fit(self.replay_raw_features_, self.replay_targets_)

        manifold_targets = targets
        if not hasattr(estimator, "partial_fit"):
            raw_features = self.replay_raw_features_
            manifold_targets = self.replay_targets_

        self._update_selection_and_manifold(raw_features, manifold_targets, incremental=fitted)
        return self._mark_fitted()

    def _initialize_fit_state(self) -> None:
        # Copy components so successive fits do not share state.
        self.transformer_ = deepcopy(self.transformer)
        self.estimator_ = deepcopy(self.estimator) if self.estimator is not None else None
        self.manifold_ = deepcopy(self.manifold) if self.manifold is not None else None
        self.selected_feature_indices_: Optional[np.ndarray] = None
        self.replay_raw_features_ = None
        self.replay_targets_ = None

    def _is_graph_estimator_fitted(self) -> bool:
        try:
            check_is_fitted(self, "_is_fitted")
            return True
        except Exception:
            return False

    def _ensure_estimator_for_fit(self, targets: Optional[Any]):
        if self.estimator_ is None:
            if targets is None:
                self.estimator_ = IsolationForestProba()
            else:
                raise ValueError(
                    "estimator is None; provide an estimator or set targets=None for unsupervised fit."
                )
        return self.estimator_

    def _fit_estimator(self, raw_features, targets: Optional[Any]) -> None:
        estimator = self._ensure_estimator_for_fit(targets)
        estimator.fit(raw_features, targets)

    def _append_replay_batch(self, raw_features: np.ndarray, targets: Optional[Any]) -> None:
        feature_batch = self._normalize_feature_matrix(raw_features)
        if self.replay_raw_features_ is None:
            self.replay_raw_features_ = feature_batch.copy()
        elif issparse(self.replay_raw_features_) or issparse(feature_batch):
            self.replay_raw_features_ = vstack(
                [self._to_csr(self.replay_raw_features_), self._to_csr(feature_batch)]
            )
        else:
            self.replay_raw_features_ = np.concatenate(
                [self.replay_raw_features_, feature_batch],
                axis=0,
            )

        if targets is None:
            return

        target_batch = np.asarray(targets)
        if self.replay_targets_ is None:
            self.replay_targets_ = target_batch.copy()
        else:
            self.replay_targets_ = np.concatenate(
                [np.asarray(self.replay_targets_), target_batch],
                axis=0,
            )

    def _update_selection_and_manifold(
        self,
        raw_features,
        targets: Optional[Any],
        *,
        incremental: bool = False,
    ) -> None:
        raw_features = self._normalize_feature_matrix(raw_features)

        # Decide how many features to select (if any) and compute indices based on estimator importances.
        k = self._resolve_n_selected_features(raw_features.shape[1])
        if k is not None:
            idx = self._compute_feature_selection_indices(k)
            self.selected_feature_indices_ = idx if idx is not None else None

        if self.manifold_ is not None:
            feats_for_manifold = self._apply_feature_selection(raw_features)
            if incremental and hasattr(self.manifold_, "partial_fit"):
                self.manifold_.partial_fit(feats_for_manifold, targets)
            else:
                self.manifold_.fit(feats_for_manifold, targets)

    def _mark_fitted(self) -> "GraphEstimator":
        self._is_fitted = True
        return self

    def _normalize_feature_matrix(self, features):
        if issparse(features):
            return features.tocsr()
        return np.asarray(features)

    def _to_csr(self, features):
        if issparse(features):
            return features.tocsr()
        return self._normalize_feature_matrix(features)

    def _transform_raw(self, graphs):
        """Transform graphs to raw feature vectors.

        Args:
            graphs: Input graphs.

        Returns:
            Feature matrix from the transformer.
        """
        return self.transformer_.transform(graphs)

    def _resolve_n_selected_features(self, n_input_features: int) -> Optional[int]:
        """Return number of features to select, or None if selection is disabled.

        Args:
            n_input_features: Number of available features.

        Returns:
            Selected feature count or None.
        """
        nsel = self.n_selected_features
        if nsel is None:
            return None
        if isinstance(nsel, int):
            return max(1, min(n_input_features, int(nsel)))
        # Floats
        try:
            f = float(nsel)
        except Exception:
            return None
        if 0 < f < 1:
            k = int(max(1, round(f * n_input_features)))
            return min(k, n_input_features)
        if f >= 1:
            return max(1, min(n_input_features, int(round(f))))
        return None

    def _compute_feature_selection_indices(self, k: int) -> Optional[np.ndarray]:
        """Compute top-k feature indices from estimator importances/coefficients.

        Args:
            k: Number of features to select.

        Returns:
            Indices of selected features, or None if unavailable.
        """
        est = self.estimator_
        if est is None:
            est = getattr(self, "estimator", None)
        if est is None:
            return None
        importances = self._extract_feature_importances(est)
        if importances is None:
            return None
        k = int(max(1, min(k, importances.shape[0])))
        self.feature_ranking_ = np.argsort(importances)[::-1]
        return self.feature_ranking_[:k]

    def _apply_feature_selection(self, features: np.ndarray) -> np.ndarray:
        """Apply stored feature selection to a feature matrix.

        Args:
            features: Raw feature matrix.

        Returns:
            Feature matrix with selection applied when available.
        """
        idx = getattr(self, "selected_feature_indices_", None)
        if idx is None:
            return features
        try:
            return features[:, idx]
        except Exception:
            return features

    def _extract_feature_importances(self, estimator: Any) -> Optional[np.ndarray]:
        """Return a 1D importance vector or None if unavailable.

        Args:
            estimator: Fitted estimator to inspect.

        Returns:
            1D importance vector or None.
        """
        importances = None
        if hasattr(estimator, "feature_importances_"):
            try:
                imp = np.asarray(estimator.feature_importances_, dtype=float)
                if imp.ndim == 1:
                    importances = imp
            except Exception:
                importances = None
        if importances is None and hasattr(estimator, "coef_"):
            try:
                coef = np.asarray(estimator.coef_, dtype=float)
                if coef.ndim == 1:
                    importances = np.abs(coef)
                elif coef.ndim == 2:
                    importances = np.mean(np.abs(coef), axis=0)
            except Exception:
                importances = None
        return importances

    def get_feature_importances(
        self,
        graphs: Optional[Any] = None,
        targets: Optional[Any] = None,
        *,
        fit_if_needed: bool = True,
        normalize: bool = False,
    ) -> np.ndarray:
        """Return feature importances from the downstream estimator.

        Args:
            graphs: Optional input graphs; required when fitting is needed.
            targets: Optional targets; required when fitting is needed.
            fit_if_needed: If True, fit the graph estimator when not fitted.
            normalize: If True, normalize importances to sum to 1 when possible.

        Returns:
            1D numpy array of feature importances.
        """
        fitted = False
        try:
            check_is_fitted(self, "_is_fitted")
            fitted = True
        except Exception:
            fitted = False

        if not fitted:
            if not fit_if_needed:
                raise ValueError("GraphEstimator is not fitted.")
            if graphs is None or targets is None:
                raise ValueError("graphs and targets are required to fit.")
            self.fit(graphs, targets)

        est = self.estimator_ if hasattr(self, "estimator_") else self.estimator
        if est is None:
            raise AttributeError("Estimator is None; cannot compute importances.")

        importances = self._extract_feature_importances(est)
        if importances is None:
            raise AttributeError(
                "Estimator does not expose feature_importances_ or coef_."
            )
        importances = np.asarray(importances, dtype=float).ravel()
        if normalize:
            total = float(np.sum(np.abs(importances)))
            if total > 0:
                importances = importances / total
        return importances

    def get_ranked_feature_ids(
        self,
        graphs: Optional[Any] = None,
        targets: Optional[Any] = None,
        *,
        reserved: Sequence[int] = (0, 1),
        fit_if_needed: bool = True,
    ) -> Sequence[int]:
        """Return ranked feature ids in descending importance order.

        Args:
            graphs: Optional input graphs; required when fitting is needed.
            targets: Optional targets; required when fitting is needed.
            reserved: Feature indices to skip (e.g., bias/degree columns).
            fit_if_needed: If True, fit the graph estimator when not fitted.

        Returns:
            Ranked feature ids in descending importance order.
        """
        importances = self.get_feature_importances(
            graphs=graphs,
            targets=targets,
            fit_if_needed=fit_if_needed,
            normalize=False,
        )
        n_features = int(importances.shape[0])
        if n_features == 0:
            return []

        mask = np.ones(n_features, dtype=bool)
        for idx in reserved:
            if 0 <= int(idx) < n_features:
                mask[int(idx)] = False
        active_idx = np.where(mask)[0]
        if active_idx.size == 0:
            return []

        order = np.argsort(-importances[active_idx])
        return [int(active_idx[i]) for i in order]

    def transform(self, graphs):
        """Transform graphs via the transformer and optional manifold.

        Args:
            graphs: Input graphs.

        Returns:
            Transformed feature matrix.
        """
        raw_features = self._transform_raw(graphs)
        # Apply feature selection before manifold projection
        selected = self._apply_feature_selection(self._normalize_feature_matrix(raw_features))
        if self.manifold_ is not None:
            return self.manifold_.transform(selected)
        return selected

    def predict(self, graphs):
        """Predict labels using the downstream estimator.

        Args:
            graphs: Input graphs.

        Returns:
            Predicted labels.
        """
        if self.estimator_ is None:
            raise AttributeError("Estimator is None; provide an estimator to use predict.")
        raw_features = self._transform_raw(graphs)
        return self.estimator_.predict(raw_features)

    def predict_proba(self, graphs, log: bool = False):
        """Predict class probabilities using the downstream estimator.

        Args:
            graphs: Input graphs.
            log: If True, return log probabilities.

        Returns:
            Predicted probabilities (or log probabilities).
        """
        if self.estimator_ is None:
            raise AttributeError("Estimator is None; provide an estimator to use predict_proba.")
        raw_features = self._transform_raw(graphs)
        if hasattr(self.estimator_, "predict_proba"):
            probs = self.estimator_.predict_proba(raw_features)
        elif hasattr(self.estimator_, "decision_function"):
            probs = self._decision_function_to_proba(
                self.estimator_.decision_function(raw_features)
            )
        else:
            raise AttributeError(
                "Underlying estimator does not implement predict_proba or decision_function."
            )
        if log:
            return np.log(probs)
        return probs

    def _decision_function_to_proba(self, scores):
        """Convert classifier decision scores to probability-like outputs."""
        scores = np.asarray(scores, dtype=float)
        classes = getattr(self.estimator_, "classes_", None)
        n_classes = len(classes) if classes is not None else None

        if scores.ndim == 1 or (scores.ndim == 2 and scores.shape[1] == 1):
            if n_classes is not None and n_classes != 2:
                raise AttributeError(
                    "Underlying estimator decision_function is binary but classes_ "
                    f"contains {n_classes} classes."
                )
            positive_scores = scores.reshape(-1)
            p_positive = self._sigmoid(positive_scores)
            return np.vstack([1.0 - p_positive, p_positive]).T

        if scores.ndim != 2:
            raise AttributeError(
                "Underlying estimator decision_function must return a 1D or 2D array."
            )
        if n_classes is not None and scores.shape[1] != n_classes:
            raise AttributeError(
                "Underlying estimator decision_function output does not match classes_."
            )
        return self._softmax(scores)

    @staticmethod
    def _sigmoid(values):
        values = np.asarray(values, dtype=float)
        out = np.empty_like(values, dtype=float)
        positive = values >= 0
        out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
        exp_values = np.exp(values[~positive])
        out[~positive] = exp_values / (1.0 + exp_values)
        return out

    @staticmethod
    def _softmax(values):
        values = np.asarray(values, dtype=float)
        shifted = values - np.max(values, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def plot(
        self,
        graphs,
        scatter_kwargs: Optional[dict] = None,
        ax=None,
        viewport_to_quantile: Optional[float] = None,
    ):
        """
        Scatter plot of transformed features (uses manifold output if present).

        Args:
            graphs: Input graphs.
            scatter_kwargs: Passed directly to plt.scatter.
            ax: Optional matplotlib Axes for rendering.
            viewport_to_quantile: If set (0<q<1), limit axes to central quantile range.

        Returns:
            Matplotlib axis containing the plot.
        """
        features = self.transform(graphs)
        if features.shape[1] < 2:
            raise ValueError("plot requires at least 2 feature dimensions.")

        features = np.asarray(features)
        scatter_kwargs = scatter_kwargs or {}
        if ax is None:
            fig, ax = plt.subplots()
            show = True
        else:
            show = False
            fig = ax.figure

        ax.scatter(features[:, 0], features[:, 1], **scatter_kwargs)
        ax.set_xlabel("feature_0")
        ax.set_ylabel("feature_1")

        if viewport_to_quantile is not None:
            q = float(viewport_to_quantile)
            if not 0 < q < 1:
                raise ValueError("viewport_to_quantile must be in (0, 1).")
            q_low = (1 - q) / 2
            q_high = 1 - q_low
            x_low, x_high = np.quantile(features[:, 0], [q_low, q_high])
            y_low, y_high = np.quantile(features[:, 1], [q_low, q_high])
            ax.set_xlim(x_low, x_high)
            ax.set_ylim(y_low, y_high)

        if show:
            fig.tight_layout()
            plt.show()

        return ax
