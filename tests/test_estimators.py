from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from scipy.sparse import csr_matrix, issparse
from sklearn.exceptions import NotFittedError

from abstractgraph_ml.estimators import (
    DropFirstTruncatedSVD,
    GraphEstimator,
    GraphLabelRepairEstimator,
)


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


class _SupervisedFirstTwoColumnsPreprocessor:
    def __init__(self):
        self.fit_targets_ = None
        self.fit_shape_ = None

    def fit(self, X, y=None):
        self.fit_shape_ = X.shape
        self.fit_targets_ = None if y is None else np.asarray(y).copy()
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, :2]


class _RecordingPredictor:
    def __init__(self):
        self.fit_X_ = None
        self.fit_y_ = None
        self.predict_X_ = None

    def fit(self, X, y):
        self.fit_X_ = np.asarray(X, dtype=float)
        self.fit_y_ = np.asarray(y)
        return self

    def predict(self, X):
        self.predict_X_ = np.asarray(X, dtype=float)
        return np.zeros(self.predict_X_.shape[0], dtype=float)


class _MaskedElementTransformer:
    def fit_transform(self, graphs, targets=None):
        return self.transform(graphs)

    def transform(self, graphs):
        rows = []
        for graph in graphs:
            query_nodes = [
                node
                for node, data in graph.nodes(data=True)
                if data.get("label") == "?"
            ]
            query_edges = []
            if graph.is_multigraph():
                for u, v, key, data in graph.edges(keys=True, data=True):
                    if data.get("label") == "?":
                        query_edges.append((u, v, key))
            else:
                for u, v, data in graph.edges(data=True):
                    if data.get("label") == "?":
                        query_edges.append((u, v, -1))

            if query_nodes:
                rows.append([1.0, float(query_nodes[0]), 0.0, 0.0])
            elif query_edges:
                u, v, key = query_edges[0]
                rows.append([2.0, float(u), float(v), float(key)])
            else:
                rows.append([0.0, 0.0, 0.0, 0.0])
        return np.asarray(rows, dtype=float)


class _LookupClassifier:
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.fit_y_ = np.asarray(y, dtype=object)
        self.classes_ = np.unique(self.fit_y_)
        self.mapping_ = {tuple(row): label for row, label in zip(X, self.fit_y_)}
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.asarray([self.mapping_[tuple(row)] for row in X], dtype=object)


def _make_graph_label_repair_estimator():
    graph_estimator = GraphEstimator(
        transformer=_MaskedElementTransformer(),
        estimator=_LookupClassifier(),
        manifold=None,
    )
    return GraphLabelRepairEstimator(graph_estimator=graph_estimator)


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


def test_graph_estimator_default_postprocessor_uses_truncated_svd_and_drops_first_component() -> None:
    transformer = _DenseThreeFeatureTransformer()
    estimator = _ReplayFitRegressor()
    graph_estimator = GraphEstimator(transformer=transformer, estimator=estimator)

    graph_estimator.fit([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])

    assert isinstance(graph_estimator.postprocessor_, DropFirstTruncatedSVD)
    assert isinstance(graph_estimator.manifold_, DropFirstTruncatedSVD)
    transformed = graph_estimator.transform([1.0, 2.0, 3.0])
    assert transformed.shape == (3, 2)


def test_graph_estimator_manifold_alias_sets_postprocessor() -> None:
    transformer = _DenseThreeFeatureTransformer()
    estimator = _ReplayFitRegressor()
    graph_estimator = GraphEstimator(
        transformer=transformer,
        estimator=estimator,
        manifold=DropFirstTruncatedSVD(n_components=2),
    )

    graph_estimator.fit([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])

    assert isinstance(graph_estimator.postprocessor_, DropFirstTruncatedSVD)
    assert graph_estimator.manifold_ is graph_estimator.postprocessor_


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


def test_graph_estimator_preprocessor_fits_with_targets_before_estimator() -> None:
    preprocessor = _SupervisedFirstTwoColumnsPreprocessor()
    estimator = _RecordingPredictor()
    graph_estimator = GraphEstimator(
        transformer=_DenseThreeFeatureTransformer(),
        estimator=estimator,
        preprocessor=preprocessor,
        manifold=None,
    )

    graph_estimator.fit([1.0, 2.0, 3.0], [0, 1, 1])
    preds = graph_estimator.predict([4.0])

    assert preds.shape == (1,)
    assert graph_estimator.preprocessor_.fit_shape_ == (3, 3)
    np.testing.assert_array_equal(
        graph_estimator.preprocessor_.fit_targets_,
        np.asarray([0, 1, 1]),
    )
    np.testing.assert_allclose(
        graph_estimator.estimator_.fit_X_,
        np.asarray([[1.0, 1.0], [2.0, 4.0], [3.0, 9.0]], dtype=float),
    )
    np.testing.assert_allclose(
        graph_estimator.estimator_.predict_X_,
        np.asarray([[4.0, 16.0]], dtype=float),
    )


def test_graph_estimator_partial_fit_refits_preprocessor_on_replay() -> None:
    preprocessor = _SupervisedFirstTwoColumnsPreprocessor()
    estimator = _ReplayFitRegressor()
    graph_estimator = GraphEstimator(
        transformer=_DenseThreeFeatureTransformer(),
        estimator=estimator,
        preprocessor=preprocessor,
        manifold=None,
    )

    graph_estimator.partial_fit([1.0, 2.0], [0.1, 0.2])
    graph_estimator.partial_fit([3.0], [0.3])

    assert graph_estimator.preprocessor_.fit_shape_ == (3, 3)
    np.testing.assert_allclose(
        graph_estimator.estimator_.fit_calls[-1][0],
        np.asarray([[1.0, 1.0], [2.0, 4.0], [3.0, 9.0]], dtype=float),
    )


def test_graph_label_repair_estimator_repairs_all_labels_without_mutating_input() -> None:
    train_graph = nx.Graph(name="train")
    train_graph.add_node(0, label="A", color="red")
    train_graph.add_node(1, label="B", color="blue")
    train_graph.add_edge(0, 1, label="x", weight=2.0)

    repair_estimator = _make_graph_label_repair_estimator()
    repair_estimator.fit([train_graph])

    test_graph = nx.Graph(name="test")
    test_graph.add_node(0, label="wrong-a", color="red")
    test_graph.add_node(1, label="wrong-b", color="blue")
    test_graph.add_edge(0, 1, label="wrong-x", weight=2.0)

    repaired = repair_estimator.transform([test_graph])[0]

    assert repaired.nodes[0]["label"] == "A"
    assert repaired.nodes[1]["label"] == "B"
    assert repaired.edges[0, 1]["label"] == "x"
    assert repaired.nodes[0]["color"] == "red"
    assert repaired.edges[0, 1]["weight"] == 2.0
    assert test_graph.nodes[0]["label"] == "wrong-a"
    assert test_graph.nodes[1]["label"] == "wrong-b"
    assert test_graph.edges[0, 1]["label"] == "wrong-x"


def test_graph_label_repair_estimator_skips_missing_training_labels() -> None:
    train_graph = nx.Graph()
    train_graph.add_node(0, label="A")
    train_graph.add_node(1)

    repair_estimator = _make_graph_label_repair_estimator()
    repair_estimator.fit([train_graph])

    assert repair_estimator.node_graph_estimator_ is not None
    assert repair_estimator.edge_graph_estimator_ is None
    np.testing.assert_array_equal(
        repair_estimator.node_graph_estimator_.estimator_.fit_y_,
        np.asarray(["A"], dtype=object),
    )


def test_graph_label_repair_estimator_transform_before_fit_raises() -> None:
    repair_estimator = _make_graph_label_repair_estimator()

    with pytest.raises(NotFittedError):
        repair_estimator.transform([nx.Graph()])


def test_graph_label_repair_estimator_raises_when_needed_classifier_was_not_fitted() -> None:
    train_graph = nx.Graph()
    train_graph.add_node(0, label="A")

    repair_estimator = _make_graph_label_repair_estimator()
    repair_estimator.fit([train_graph])

    test_graph = nx.Graph()
    test_graph.add_node(0, label="wrong-a")
    test_graph.add_node(1, label="wrong-b")
    test_graph.add_edge(0, 1, label="wrong-x")

    with pytest.raises(ValueError, match="No edge label repair classifier was fitted"):
        repair_estimator.transform([test_graph])


def test_graph_label_repair_estimator_repairs_multigraph_edges_by_key() -> None:
    train_graph = nx.MultiGraph()
    train_graph.add_node(0, label="A")
    train_graph.add_node(1, label="B")
    train_graph.add_edge(0, 1, key=0, label="x")
    train_graph.add_edge(0, 1, key=1, label="y")

    repair_estimator = _make_graph_label_repair_estimator()
    repair_estimator.fit([train_graph])

    test_graph = nx.MultiGraph()
    test_graph.add_node(0, label="wrong-a")
    test_graph.add_node(1, label="wrong-b")
    test_graph.add_edge(0, 1, key=0, label="wrong-x")
    test_graph.add_edge(0, 1, key=1, label="wrong-y")

    repaired = repair_estimator.transform([test_graph])[0]

    assert repaired.nodes[0]["label"] == "A"
    assert repaired.nodes[1]["label"] == "B"
    assert repaired.edges[0, 1, 0]["label"] == "x"
    assert repaired.edges[0, 1, 1]["label"] == "y"
