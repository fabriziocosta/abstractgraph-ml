from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, issparse

from abstractgraph_ml.estimators import DropFirstTruncatedSVD, GraphEstimator


class _CountingTransformer:
    def __init__(self):
        self.fit_transform_calls = 0
        self.transform_calls = 0

    def fit_transform(self, graphs, targets=None):
        self.fit_transform_calls += 1
        return self.transform(graphs)

    def transform(self, graphs):
        self.transform_calls += 1
        return np.asarray([[float(graph)] for graph in graphs], dtype=float)


class _PartialFitRegressor:
    def __init__(self):
        self.partial_fit_calls = []

    def partial_fit(self, X, y, **kwargs):
        self.partial_fit_calls.append((np.asarray(X), np.asarray(y), dict(kwargs)))
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=float)


class _ReplayFitRegressor:
    def __init__(self):
        self.fit_calls = []

    def fit(self, X, y):
        self.fit_calls.append((X, np.asarray(y)))
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=float)


class _BinaryDecisionEstimator:
    classes_ = np.asarray([0, 1])

    def fit(self, X, y):
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, 0]


class _MulticlassDecisionEstimator:
    classes_ = np.asarray(["a", "b", "c"])

    def fit(self, X, y):
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        return np.column_stack([-X[:, 0], np.zeros(X.shape[0]), X[:, 0]])


class _SparseCountingTransformer:
    def __init__(self):
        self.fit_transform_calls = 0
        self.transform_calls = 0

    def fit_transform(self, graphs, targets=None):
        self.fit_transform_calls += 1
        return self.transform(graphs)

    def transform(self, graphs):
        self.transform_calls += 1
        rows = [[float(graph), float(graph) + 10.0] for graph in graphs]
        return csr_matrix(np.asarray(rows, dtype=float))


class _DenseThreeFeatureTransformer:
    def __init__(self):
        self.fit_transform_calls = 0
        self.transform_calls = 0

    def fit_transform(self, graphs, targets=None):
        self.fit_transform_calls += 1
        return self.transform(graphs)

    def transform(self, graphs):
        self.transform_calls += 1
        rows = [
            [float(graph), float(graph) ** 2, float(graph) + 5.0]
            for graph in graphs
        ]
        return np.asarray(rows, dtype=float)


def test_graph_estimator_partial_fit_reuses_transformer_state() -> None:
    transformer = _CountingTransformer()
    estimator = _PartialFitRegressor()
    graph_estimator = GraphEstimator(transformer=transformer, estimator=estimator, manifold=None)

    graph_estimator.partial_fit([1.0, 2.0], [0.1, 0.2])
    graph_estimator.partial_fit([3.0], [0.3])

    assert graph_estimator.transformer_.fit_transform_calls == 1
    assert graph_estimator.transformer_.transform_calls == 2
    assert len(graph_estimator.estimator_.partial_fit_calls) == 2
    np.testing.assert_allclose(
        graph_estimator.estimator_.partial_fit_calls[0][0],
        np.asarray([[1.0], [2.0]], dtype=float),
    )
    np.testing.assert_allclose(
        graph_estimator.estimator_.partial_fit_calls[1][0],
        np.asarray([[3.0]], dtype=float),
    )


def test_graph_estimator_partial_fit_replays_vectorized_batches_when_estimator_lacks_partial_fit() -> None:
    transformer = _CountingTransformer()
    estimator = _ReplayFitRegressor()
    graph_estimator = GraphEstimator(transformer=transformer, estimator=estimator, manifold=None)

    graph_estimator.partial_fit([1.0, 2.0], [0.1, 0.2])
    graph_estimator.partial_fit([3.0], [0.3])

    assert graph_estimator.transformer_.fit_transform_calls == 1
    assert graph_estimator.transformer_.transform_calls == 2
    assert len(graph_estimator.estimator_.fit_calls) == 2
    np.testing.assert_allclose(
        graph_estimator.estimator_.fit_calls[0][0],
        np.asarray([[1.0], [2.0]], dtype=float),
    )
    np.testing.assert_allclose(
        graph_estimator.estimator_.fit_calls[0][1],
        np.asarray([0.1, 0.2]),
    )
    np.testing.assert_allclose(
        graph_estimator.estimator_.fit_calls[1][0],
        np.asarray([[1.0], [2.0], [3.0]], dtype=float),
    )
    np.testing.assert_allclose(
        graph_estimator.estimator_.fit_calls[1][1],
        np.asarray([0.1, 0.2, 0.3]),
    )
    np.testing.assert_allclose(
        graph_estimator.replay_raw_features_,
        np.asarray([[1.0], [2.0], [3.0]], dtype=float),
    )
    np.testing.assert_allclose(
        graph_estimator.replay_targets_,
        np.asarray([0.1, 0.2, 0.3]),
    )


def test_graph_estimator_partial_fit_preserves_csr_replay_batches() -> None:
    transformer = _SparseCountingTransformer()
    estimator = _ReplayFitRegressor()
    graph_estimator = GraphEstimator(transformer=transformer, estimator=estimator, manifold=None)

    graph_estimator.partial_fit([1.0, 2.0], [0.1, 0.2])
    graph_estimator.partial_fit([3.0], [0.3])

    assert transformer.fit_transform_calls == 0
    assert graph_estimator.transformer_.fit_transform_calls == 1
    assert graph_estimator.transformer_.transform_calls == 2
    assert issparse(graph_estimator.replay_raw_features_)
    assert graph_estimator.replay_raw_features_.shape == (3, 2)
    np.testing.assert_allclose(
        graph_estimator.replay_raw_features_.toarray(),
        np.asarray([[1.0, 11.0], [2.0, 12.0], [3.0, 13.0]], dtype=float),
    )
    assert len(graph_estimator.estimator_.fit_calls) == 2
    assert issparse(graph_estimator.estimator_.fit_calls[0][0])
    assert issparse(graph_estimator.estimator_.fit_calls[1][0])
    np.testing.assert_allclose(
        graph_estimator.estimator_.fit_calls[1][0].toarray(),
        np.asarray([[1.0, 11.0], [2.0, 12.0], [3.0, 13.0]], dtype=float),
    )


def test_graph_estimator_default_manifold_uses_truncated_svd_and_drops_first_component() -> None:
    transformer = _DenseThreeFeatureTransformer()
    estimator = _ReplayFitRegressor()
    graph_estimator = GraphEstimator(transformer=transformer, estimator=estimator)

    graph_estimator.fit([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])

    assert isinstance(graph_estimator.manifold_, DropFirstTruncatedSVD)
    transformed = graph_estimator.transform([1.0, 2.0, 3.0])
    assert transformed.shape == (3, 2)


def test_graph_estimator_predict_proba_uses_binary_decision_function_fallback() -> None:
    graph_estimator = GraphEstimator(
        transformer=_CountingTransformer(),
        estimator=_BinaryDecisionEstimator(),
        manifold=None,
    )
    graph_estimator.fit([-1.0, 1.0], [0, 1])

    probs = graph_estimator.predict_proba([-2.0, 0.0, 2.0])

    assert probs.shape == (3, 2)
    np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(3))
    np.testing.assert_allclose(probs[1], np.asarray([0.5, 0.5]))
    assert probs[0, 1] < 0.5
    assert probs[2, 1] > 0.5


def test_graph_estimator_predict_proba_uses_multiclass_decision_function_fallback() -> None:
    graph_estimator = GraphEstimator(
        transformer=_CountingTransformer(),
        estimator=_MulticlassDecisionEstimator(),
        manifold=None,
    )
    graph_estimator.fit([-1.0, 0.0, 1.0], ["a", "b", "c"])

    probs = graph_estimator.predict_proba([2.0])

    assert probs.shape == (1, 3)
    np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(1))
    assert probs[0, 2] > probs[0, 1] > probs[0, 0]
