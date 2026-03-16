from __future__ import annotations

import networkx as nx

import abstractgraph.operators as ops
from abstractgraph.graphs import AbstractGraph
from abstractgraph_ml.importance import annotate_graph_node_saliency


class _DummyTransformer:
    def __init__(self):
        self.decomposition_function = ops.node()
        self.nbits = 6


class _DummyEstimator:
    def __init__(self):
        self.transformer = _DummyTransformer()

    def get_ranked_feature_ids(self, fit_if_needed: bool = False):
        return [2, 3, 4]


def test_importance_uses_canonical_graph_and_mapped_subgraph_names() -> None:
    graph = nx.path_graph(3)
    for node in graph.nodes:
        graph.nodes[node]["label"] = str(node)
        graph.nodes[node]["attribute"] = [1.0]

    annotated, node_score, edge_score = annotate_graph_node_saliency(graph, _DummyEstimator())
    assert annotated.number_of_nodes() == graph.number_of_nodes()
    assert set(node_score) == set(graph.nodes())
    assert len(edge_score) == graph.number_of_edges()
