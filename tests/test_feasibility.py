from __future__ import annotations

import networkx as nx

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
