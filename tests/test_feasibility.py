from __future__ import annotations

import networkx as nx
import numpy as np

from abstractgraph_ml.feasibility import FeasibilityEstimator, FeasibilityEstimatorFeatureCannotExist


class _RecordingEstimator:
    def __init__(self, predictions):
        self.predictions = predictions
        self.calls = []

    def fit(self, graphs):
        return self

    def predict(self, graphs):
        self.calls.append(len(graphs))
        return self.predictions[: len(graphs)]


class _ViolationEstimator:
    def __init__(self, violations):
        self._violations = np.asarray(violations)

    def fit(self, graphs):
        return self

    def number_of_violations(self, graphs):
        return self._violations[: len(graphs)]


class _ViolatingEdgeSetEstimator:
    def __init__(self, edge_sets):
        self._edge_sets = edge_sets

    def fit(self, graphs):
        return self

    def violating_edge_sets(self, graphs):
        return self._edge_sets[: len(graphs)]


def _make_graphs(n):
    graphs = []
    for idx in range(n):
        graph = nx.Graph()
        graph.add_node(0, label=str(idx))
        graphs.append(graph)
    return graphs


def _make_labeled_path_graph(length):
    graph = nx.Graph()
    for node in range(length):
        graph.add_node(node, label="n")
    for node in range(length - 1):
        graph.add_edge(node, node + 1, label="e")
    return graph


def _singleton_and_edge_decomposition(abstract_graph):
    out = abstract_graph.copy()
    if out.base_graph.number_of_nodes() >= 4:
        prefix_nodes = sorted(out.base_graph.nodes())[:3]
        out.create_interpretation_node_with_subgraph_from_nodes(prefix_nodes)
    for node in out.base_graph.nodes():
        out.create_interpretation_node_with_subgraph_from_nodes([node])
    for edge in out.base_graph.edges():
        out.create_interpretation_node_with_subgraph_from_edges([edge])
    return out


def test_feasibility_predict_short_circuits_on_surviving_graphs():
    graphs = _make_graphs(4)
    first = _RecordingEstimator([True, False, True, False])
    second = _RecordingEstimator([False, True])
    third = _RecordingEstimator([True])
    estimator = FeasibilityEstimator([first, second, third])

    preds = estimator.predict(graphs)

    assert preds.tolist() == [False, False, True, False]
    assert first.calls == [4]
    assert second.calls == [2]
    assert third.calls == [1]


def test_feasibility_predict_stops_when_no_graphs_survive():
    graphs = _make_graphs(3)
    first = _RecordingEstimator([False, False, False])
    second = _RecordingEstimator([True, True, True])
    estimator = FeasibilityEstimator([first, second])

    preds = estimator.predict(graphs)

    assert preds.tolist() == [False, False, False]
    assert first.calls == [3]
    assert second.calls == []


def test_feasibility_violations_returns_per_estimator_matrix():
    graphs = _make_graphs(3)
    first = _ViolationEstimator([0, 1, 2])
    second = _ViolationEstimator([3, 4, 5])
    estimator = FeasibilityEstimator([first, second])

    preds = estimator.violations(graphs)

    assert preds.shape == (3, 2)
    assert preds.tolist() == [[0, 3], [1, 4], [2, 5]]


def test_feasibility_number_of_violations_sums_violation_matrix():
    graphs = _make_graphs(3)
    first = _ViolationEstimator([0, 1, 2])
    second = _ViolationEstimator([3, 4, 5])
    estimator = FeasibilityEstimator([first, second])

    preds = estimator.number_of_violations(graphs)

    assert preds.tolist() == [3, 5, 7]


def test_feature_cannot_exist_violating_edge_sets_returns_one_list_per_graph():
    train_graphs = [_make_labeled_path_graph(2)]
    test_graphs = [_make_labeled_path_graph(2), _make_labeled_path_graph(3)]
    estimator = FeasibilityEstimatorFeatureCannotExist(
        decomposition_function=_singleton_and_edge_decomposition,
        parallel=False,
    ).fit(train_graphs)

    violating = estimator.violating_edge_sets(test_graphs)

    assert len(violating) == 2
    assert violating[0] == []
    assert violating[1] == [frozenset({(0, 1), (1, 2)})]


def test_feature_cannot_exist_violating_edge_sets_preserves_multiple_violating_nodes():
    train_graph = _make_labeled_path_graph(2)
    test_graph = _make_labeled_path_graph(4)
    estimator = FeasibilityEstimatorFeatureCannotExist(
        decomposition_function=_singleton_and_edge_decomposition,
        parallel=False,
    ).fit([train_graph])

    violating = estimator.violating_edge_sets([test_graph])

    assert violating == [[
        frozenset({(0, 1), (1, 2), (2, 3)}),
        frozenset({(0, 1), (1, 2)}),
    ]]


def test_feature_cannot_exist_violating_edge_sets_canonicalizes_edge_direction():
    train_graph = nx.Graph()
    train_graph.add_node(1, label="a")
    train_graph.add_node(2, label="a")
    train_graph.add_edge(1, 2, label="x")

    test_graph = nx.Graph()
    test_graph.add_node(5, label="a")
    test_graph.add_node(2, label="a")
    test_graph.add_node(1, label="a")
    test_graph.add_edge(5, 2, label="x")
    test_graph.add_edge(2, 1, label="x")

    estimator = FeasibilityEstimatorFeatureCannotExist(
        decomposition_function=_singleton_and_edge_decomposition,
        parallel=False,
    ).fit([train_graph])

    violating = estimator.violating_edge_sets([test_graph])

    assert violating == [[frozenset({(1, 2), (2, 5)})]]


def test_feasibility_violating_edge_sets_concatenates_and_skips_missing_methods():
    graphs = _make_graphs(2)
    first = _ViolatingEdgeSetEstimator([[frozenset({(0, 1)})], []])
    second = _ViolationEstimator([0, 0])
    third = _ViolatingEdgeSetEstimator([[frozenset({(2, 3)})], [frozenset()]])
    estimator = FeasibilityEstimator([first, second, third])

    violating = estimator.violating_edge_sets(graphs)

    assert violating == [
        [frozenset({(0, 1)}), frozenset({(2, 3)})],
        [frozenset()],
    ]
