from __future__ import annotations

import numpy as np

from abstractgraph_ml.estimators import GraphEstimator


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
        self.fit_calls.append((np.asarray(X), np.asarray(y)))
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=float)


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
