from __future__ import annotations

import networkx as nx
import numpy as np

from abstractgraph_ml.feasibility import FeasibilityEstimator


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


def _make_graphs(n):
    graphs = []
    for idx in range(n):
        graph = nx.Graph()
        graph.add_node(0, label=str(idx))
        graphs.append(graph)
    return graphs


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
