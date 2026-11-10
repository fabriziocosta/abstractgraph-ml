from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from abstractgraph.graphs import AbstractGraph
from abstractgraph_ml import NeuralGraphEstimator
from abstractgraph_ml.neural import _graph_to_input


def identity_decomposition(abstract_graph: AbstractGraph) -> AbstractGraph:
    return abstract_graph


def _graph(n_nodes: int, *, vector_width: int = 1, reverse_insertion: bool = False) -> nx.Graph:
    graph = nx.Graph()
    nodes = list(range(n_nodes))
    if reverse_insertion:
        nodes.reverse()
    for node in nodes:
        graph.add_node(node, label=f"node-{node % 2}", attribute=[float(node % 3)] * vector_width)
    edges = [(node, node + 1) for node in range(n_nodes - 1)]
    if reverse_insertion:
        edges.reverse()
    for source, target in edges:
        graph.add_edge(source, target, label="link", attribute=[0.5])
    return graph


def _estimator(**kwargs) -> NeuralGraphEstimator:
    params = dict(
        decomposition_function=identity_decomposition,
        nbits=5,
        local_d_model=8,
        local_nhead=2,
        local_num_layers=1,
        local_dim_feedforward=16,
        global_d_model=8,
        global_nhead=2,
        global_num_layers=1,
        global_dim_feedforward=16,
        dropout=0.0,
        epochs=1,
        batch_size=2,
        seed=7,
        verbose=False,
    )
    params.update(kwargs)
    return NeuralGraphEstimator(**params)


def test_preprocessing_builds_node_and_edge_element_tokens() -> None:
    graph = _graph(3)
    node_input = _graph_to_input(graph, identity_decomposition, 5, "node")
    edge_input = _graph_to_input(graph, identity_decomposition, 5, "edge")

    assert len(node_input["interpretations"]) == 1
    assert len(node_input["interpretations"][0]["element_labels"]) == 3
    assert len(edge_input["interpretations"][0]["element_labels"]) == 2
    assert len(edge_input["interpretations"][0]["endpoint_labels"]) == 2


@pytest.mark.parametrize("element_mode", ["node", "edge"])
def test_fit_predict_transform_and_permutation_invariance(element_mode: str) -> None:
    graphs = [_graph(2), _graph(3), _graph(4), _graph(5)]
    model = _estimator(element_mode=element_mode, mode="classification")
    model.fit(graphs, ["short", "short", "long", "long"])

    probabilities = model.predict_proba([graphs[0], graphs[-1]])
    embeddings = model.transform([graphs[0], graphs[-1]])
    assert probabilities.shape == (2, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2), atol=1e-6)
    assert model.predict([graphs[0], graphs[-1]]).shape == (2,)
    assert embeddings.shape == (2, 8)

    if element_mode == "edge":
        # Reversing insertion order changes iteration order but not the edge set.
        reversed_graph = _graph(4, reverse_insertion=True)
        np.testing.assert_allclose(model.transform([graphs[2]]), model.transform([reversed_graph]), atol=1e-5)


def test_regression_empty_graph_and_attribute_width_validation() -> None:
    graphs = [_graph(1), _graph(2), _graph(3)]
    model = _estimator(mode="regression", output_dim=1)
    model.fit(graphs, [0.1, 0.2, 0.3])
    prediction = model.predict([graphs[0]])
    assert prediction.shape == (1, 1)
    assert model.transform([nx.Graph()]).shape == (1, 8)
    assert model.predict([]).shape == (0, 1)

    with pytest.raises(ValueError, match="attribute width changed"):
        model.transform([_graph(2, vector_width=2)])


def test_get_set_params_and_reset() -> None:
    model = _estimator()
    assert model.get_params()["element_mode"] == "node"
    model.fit([_graph(2), _graph(3)], [0, 1])
    assert model.model_ is not None
    model.set_params(element_mode="edge")
    assert model.model_ is None
    assert model.element_mode == "edge"


def test_missing_labels_and_attributes_use_reserved_or_zero_features() -> None:
    graph = nx.Graph()
    graph.add_node("a")
    graph.add_node("b", label="known", attribute=[1.0])
    graph.add_edge("a", "b")

    node_input = _graph_to_input(graph, identity_decomposition, 5, "node")
    assert node_input["interpretations"][0]["element_labels"][0] == 1
    assert node_input["interpretations"][0]["element_attributes"][0].size == 0

    edge_input = _graph_to_input(graph, identity_decomposition, 5, "edge")
    assert edge_input["interpretations"][0]["element_attributes"][0].size == 0

