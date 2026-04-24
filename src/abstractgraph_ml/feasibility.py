#!/usr/bin/env python
"""Provides scikit interface."""

from typing import FrozenSet, Iterable, List, Tuple

import numpy as np
import networkx as nx
from abstractgraph.graphs import get_mapped_subgraph, graph_to_abstract_graph
from abstractgraph.vectorize import AbstractGraphTransformer

Node = int
Edge = Tuple[Node, Node]

def label_attribute_is_present(graph):
    """Check whether all nodes and edges carry a ``label`` attribute.

    Args:
        graph: NetworkX graph to inspect.

    Returns:
        bool: ``True`` when every node and edge has a ``label`` attribute.
    """
    for u in graph.nodes():
        if 'label' not in graph.nodes[u]:
            return False
    for u,v in graph.edges():
        if 'label' not in graph.edges[u,v]:
            return False
    return True

def filter_graphs_without_node_and_edge_label_attribute(graphs):
    """Keep only graphs that have node and edge labels.

    Args:
        graphs: Iterable of NetworkX graphs.

    Returns:
        list: Graphs for which :func:`label_attribute_is_present` is ``True``.
    """
    return [graph for graph in graphs if label_attribute_is_present(graph)]

class FeasibilityEstimatorFromBooleanFunction(object):
    """Wrap a graph-level boolean function as a feasibility estimator.

    Args:
        boolean_function: Callable taking a graph and returning a boolean-like
            feasibility decision.
    """
    def __init__(self, boolean_function):
        self.boolean_function = boolean_function

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def fit(self, graphs):
        """Return the estimator unchanged.

        Args:
            graphs: Training graphs.

        Returns:
            FeasibilityEstimatorFromBooleanFunction: The fitted estimator.
        """
        return self

    def number_of_violations(self, graphs):
        """Return per-graph violation counts.

        Args:
            graphs: Graphs to evaluate.

        Returns:
            np.ndarray: Integer array with one entry per graph.
        """
        preds = np.array([self.boolean_function(graph) for graph in graphs]).astype(int)
        return preds

    def predict(self, graphs):
        """Predict feasibility for each graph.

        Args:
            graphs: Graphs to evaluate.

        Returns:
            np.ndarray: Boolean feasibility predictions.
        """
        preds = np.array([self.boolean_function(graph) for graph in graphs])
        return preds


def FeasibilityEstimatorHasNodeAndEdgeLabelAttribute():
    """Build a feasibility estimator enforcing node and edge labels.

    Returns:
        FeasibilityEstimatorFromBooleanFunction: Label-presence estimator.
    """
    return FeasibilityEstimatorFromBooleanFunction(boolean_function=label_attribute_is_present)


def FeasibilityEstimatorHasNoSelfLoops():
    """Build a feasibility estimator forbidding self-loops.

    Returns:
        FeasibilityEstimatorFromBooleanFunction: Self-loop feasibility
        estimator.
    """
    return FeasibilityEstimatorFromBooleanFunction(boolean_function=lambda graph:nx.number_of_selfloops(graph)==0)


def FeasibilityEstimatorNumberOfNodesInRange(min_size=1, max_size=None):
    """Build a node-count feasibility estimator with fixed bounds.

    Args:
        min_size: Minimum allowed number of nodes.
        max_size: Optional maximum allowed number of nodes.

    Returns:
        FeasibilityEstimatorFromBooleanFunction: Node-count feasibility
        estimator.
    """
    def is_n_nodes_in_range(graph):
        number_of_nodes = nx.number_of_nodes(graph)
        if max_size is None: return number_of_nodes >= min_size
        return  min_size <= number_of_nodes <= max_size

    return FeasibilityEstimatorFromBooleanFunction(boolean_function=is_n_nodes_in_range)


def FeasibilityEstimatorNumberOfEdgesInRange(min_size=1, max_size=None):
    """Build an edge-count feasibility estimator with fixed bounds.

    Args:
        min_size: Minimum allowed number of edges.
        max_size: Optional maximum allowed number of edges.

    Returns:
        FeasibilityEstimatorFromBooleanFunction: Edge-count feasibility
        estimator.
    """
    def is_n_edges_in_range(graph):
        number_of_edges = nx.number_of_edges(graph)
        if max_size is None: return number_of_edges >= min_size
        return  min_size <= number_of_edges <= max_size

    return FeasibilityEstimatorFromBooleanFunction(boolean_function=is_n_edges_in_range)


def FeasibilityEstimatorNumberOfNodesInObservedRange(quantile=None):
    """Build a node-count feasibility estimator that learns bounds at fit time.

    Args:
        quantile: Optional symmetric trimming quantile. If provided, learned
            bounds are ``[q, 1-q]`` quantiles of observed node counts.

    Returns:
        WithinRangeFeasibilityEstimatorFromNumericalFunction: Fitted bounds are
        learned from training graphs in ``fit``.
    """
    return WithinRangeFeasibilityEstimatorFromNumericalFunction(
        numerical_function=lambda graph: nx.number_of_nodes(graph),
        quantile=quantile,
    )


def FeasibilityEstimatorNumberOfEdgesInObservedRange(quantile=None):
    """Build an edge-count feasibility estimator that learns bounds at fit time.

    Args:
        quantile: Optional symmetric trimming quantile. If provided, learned
            bounds are ``[q, 1-q]`` quantiles of observed edge counts.

    Returns:
        WithinRangeFeasibilityEstimatorFromNumericalFunction: Fitted bounds are
        learned from training graphs in ``fit``.
    """
    return WithinRangeFeasibilityEstimatorFromNumericalFunction(
        numerical_function=lambda graph: nx.number_of_edges(graph),
        quantile=quantile,
    )

class WithinRangeFeasibilityEstimatorFromNumericalFunction(object):
    """Learn an admissible numerical range from training graphs.

    Args:
        numerical_function: Callable mapping a graph to a numeric summary.
        quantile: Optional symmetric trimming quantile used to learn bounds.
    """
    def __init__(self, numerical_function, quantile=None):
        self.numerical_function = numerical_function
        self.quantile = quantile

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def fit(self, graphs):
        """Learn lower and upper bounds from training graphs.

        Args:
            graphs: Training graphs.

        Returns:
            WithinRangeFeasibilityEstimatorFromNumericalFunction: The fitted
            estimator.
        """
        vals = np.array([self.numerical_function(graph) for graph in graphs])
        if self.quantile is None:
            self.min = np.min(vals)
            self.max = np.max(vals)
        else:
            self.min = np.quantile(vals, self.quantile)
            self.max = np.quantile(vals, 1-self.quantile)
        return self

    def is_feasible(self, graph):
        """Check whether a graph stays within the learned range.

        Args:
            graph: Graph to evaluate.

        Returns:
            bool: ``True`` when the graph value is within the learned bounds.
        """
        val = self.numerical_function(graph)
        test = val >= self.min and val <= self.max 
        return test

    def size_of_violation(self, graph):
        """Measure how far a graph lies outside the learned range.

        Args:
            graph: Graph to evaluate.

        Returns:
            float: Distance from the nearest valid boundary, or ``0`` when the
            graph is feasible.
        """
        val = self.numerical_function(graph)
        if val <= self.min: 
            return self.min - val 
        elif val >= self.max: 
            return val - self.max
        else:
            return 0

    def number_of_violations(self, graphs):
        """Return per-graph violation magnitudes.

        Args:
            graphs: Graphs to evaluate.

        Returns:
            np.ndarray: Integer violation magnitudes.
        """
        preds = np.array([self.size_of_violation(graph) for graph in graphs]).astype(int)
        return preds

    def predict(self, graphs):
        """Predict feasibility for each graph.

        Args:
            graphs: Graphs to evaluate.

        Returns:
            np.ndarray: Boolean feasibility predictions.
        """
        preds = np.array([self.is_feasible(graph) for graph in graphs])
        return preds


class FeasibilityEstimatorIsConnected(object):
    """Require a specific number of connected components.

    Args:
        number_connected_components: Required number of connected components.
    """
    def __init__(self, number_connected_components=1):
        self.number_connected_components = number_connected_components

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def fit(self, graphs):
        """Return the estimator unchanged.

        Args:
            graphs: Training graphs.

        Returns:
            FeasibilityEstimatorIsConnected: The fitted estimator.
        """
        return self

    def number_of_violations(self, graphs):
        """Return excess connected components for each graph.

        Args:
            graphs: Graphs to evaluate.

        Returns:
            np.ndarray: Integer violation counts.
        """
        preds = np.array([nx.number_connected_components(graph)-1 for graph in graphs])
        return preds

    def predict(self, graphs):
        """Predict connectivity feasibility for each graph.

        Args:
            graphs: Graphs to evaluate.

        Returns:
            np.ndarray: Boolean feasibility predictions.
        """
        preds = np.array([nx.number_connected_components(graph)==self.number_connected_components for graph in graphs])
        return preds


class FeasibilityEstimatorFeatureMustExist(object):
    """Require features that are always present in the training set.

    Args:
        decomposition_function: Optional graph decomposition applied before
            vectorization.
        nbits: Hash width used by the vectorizer.
        parallel: Whether vectorization should use parallel workers by
            default.
        n_jobs: Optional explicit worker count overriding ``parallel``.
    """
    def __init__(self, decomposition_function=None, nbits=14, parallel=True, n_jobs=None):
        self.decomposition_function = decomposition_function
        self.nbits = nbits
        self.must_exist_features_vec = None
        self.parallel = parallel
        self.n_jobs = n_jobs

    def transform(self, graphs):
        """Vectorize graphs into a sparse feature matrix.

        Args:
            graphs: Graphs to transform.

        Returns:
            scipy.sparse.csr_matrix: Sparse feature matrix with one row per
            graph.
        """
        # Allow explicit worker control while preserving the old parallel flag.
        n_jobs = self.n_jobs if self.n_jobs is not None else (-1 if self.parallel else 1)

        # Default to identity decomposition if none provided
        decomp = self.decomposition_function or (lambda ag: ag)

        transformer = AbstractGraphTransformer(
            nbits=self.nbits,
            decomposition_function=decomp,
            return_dense=False,   # keep CSR (original code expects it)
            n_jobs=n_jobs,
        )
        return transformer.transform(graphs)

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def fit(self, graphs):
        """Learn which features must always exist.

        Args:
            graphs: Training graphs.

        Returns:
            FeasibilityEstimatorFeatureMustExist: The fitted estimator.
        """
        data_mtx = self.transform(graphs).astype(bool)
        n_instances, n_features = data_mtx.shape
        # find all features that are always present
        exist_feats = data_mtx.sum(axis=0).A.flatten().astype(bool)
        self.must_exist_features_vec = np.logical_or(np.zeros(n_features).astype(bool), (exist_feats == n_instances))
        return self

    def number_of_violations(self, graphs):
        """Count missing mandatory features for each graph.

        Args:
            graphs: Graphs to evaluate.

        Returns:
            np.ndarray: Integer violation counts.
        """
        assert self.must_exist_features_vec is not None, 'FeasibilityEstimatorFeatureMustExist is not fit'
        data_mtx = self.transform(graphs)
        preds = data_mtx.dot(self.must_exist_features_vec.astype(int))
        return preds

    def predict(self, graphs):
        """Predict whether each graph contains all mandatory features.

        Args:
            graphs: Graphs to evaluate.

        Returns:
            np.ndarray: Boolean feasibility predictions.
        """
        assert self.must_exist_features_vec is not None, 'FeasibilityEstimatorFeatureMustExist is not fit'
        data_mtx = self.transform(graphs).astype(bool)
        preds = data_mtx.dot(self.must_exist_features_vec)
        return preds


class FeasibilityEstimatorFeatureCannotExist(object):
    """Forbid features that never appear in the training set.

    Args:
        decomposition_function: Optional graph decomposition applied before
            vectorization.
        nbits: Hash width used by the vectorizer.
        parallel: Whether vectorization should use parallel workers by
            default.
        backend: Joblib backend passed to the vectorizer.
        n_jobs: Optional explicit worker count overriding ``parallel``.
    """
    def __init__(self, decomposition_function=None, nbits=14, parallel=True, backend="threading", n_jobs=None):
        self.decomposition_function = decomposition_function
        self.nbits = nbits
        self.cannot_exist_features_vec = None
        self.parallel = parallel
        self.backend = backend
        self.n_jobs = n_jobs

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def transform(self, graphs):
        """Vectorize graphs into a sparse feature matrix.

        Args:
            graphs: Graphs to transform.

        Returns:
            scipy.sparse.csr_matrix: Sparse feature matrix with one row per
            graph.
        """
        # Allow explicit worker control while preserving the old parallel flag.
        n_jobs = self.n_jobs if self.n_jobs is not None else (-1 if self.parallel else 1)

        # Default to identity decomposition if none provided
        decomp = self.decomposition_function or (lambda qg: qg)

        transformer = AbstractGraphTransformer(
            nbits=self.nbits,
            decomposition_function=decomp,
            return_dense=False,   # keep CSR (original code expects it)
            n_jobs=n_jobs,
            backend=self.backend,
        )
        return transformer.transform(graphs)

    def _decomposition(self):
        """Return the decomposition function used by both transform and diagnostics."""
        return self.decomposition_function or (lambda qg: qg)

    def _abstract_graphs(self, graphs):
        """Convert graphs into labeled AbstractGraphs matching transform semantics."""
        graphs = list(graphs)
        decomp = self._decomposition()
        return [
            graph_to_abstract_graph(graph=graph, decomposition_function=decomp, nbits=self.nbits)
            for graph in graphs
        ]

    def _is_forbidden_label(self, label):
        """Check whether an interpretation-node label maps to a forbidden bucket."""
        assert self.cannot_exist_features_vec is not None, 'FeasibilityEstimatorFeatureCannotExist is not fit'
        if label is None:
            return False
        try:
            label_int = int(label)
        except (TypeError, ValueError):
            return False
        if label_int < 0 or label_int >= len(self.cannot_exist_features_vec):
            return False
        return bool(self.cannot_exist_features_vec[label_int])

    def _mapped_subgraph_edge_set(self, mapped_subgraph) -> FrozenSet[Edge]:
        """Return a stable edge-set view for a mapped base subgraph."""
        if mapped_subgraph is None:
            return frozenset()
        if mapped_subgraph.is_directed():
            return frozenset((u, v) for u, v in mapped_subgraph.edges())
        return frozenset((min(u, v), max(u, v)) for u, v in mapped_subgraph.edges())

    def _mapped_subgraph_node_set(self, mapped_subgraph) -> FrozenSet[Node]:
        """Return the base-graph node ids covered by a mapped subgraph."""
        if mapped_subgraph is None:
            return frozenset()
        return frozenset(mapped_subgraph.nodes())

    def fit(self, graphs):
        """Learn which features must never exist.

        Args:
            graphs: Training graphs.

        Returns:
            FeasibilityEstimatorFeatureCannotExist: The fitted estimator.
        """
        data_mtx = self.transform(graphs).astype(bool)
        # find all missing features
        exist_feats = data_mtx.astype(bool).sum(axis=0).A.flatten().astype(bool)
        self.cannot_exist_features_vec = np.logical_not(exist_feats)
        self.seen_feature_labels = set(np.flatnonzero(exist_feats))
        return self

    def number_of_violations(self, graphs):
        """Count forbidden features present in each graph.

        Args:
            graphs: Graphs to evaluate.

        Returns:
            np.ndarray: Integer violation counts.
        """
        assert self.cannot_exist_features_vec is not None, 'FeasibilityEstimatorFeatureCannotExist is not fit'
        data_mtx = self.transform(graphs)
        preds = data_mtx.dot(self.cannot_exist_features_vec.astype(int))
        return preds

    def predict(self, graphs):
        """Predict whether each graph avoids forbidden features.

        Args:
            graphs: Graphs to evaluate.

        Returns:
            np.ndarray: Boolean feasibility predictions.
        """
        assert self.cannot_exist_features_vec is not None, 'FeasibilityEstimatorFeatureCannotExist is not fit'
        data_mtx = self.transform(graphs).astype(bool)
        cannot_exist = data_mtx.dot(self.cannot_exist_features_vec)
        preds = np.logical_not(cannot_exist)
        return preds

    def violating_edge_sets(self, graphs: Iterable) -> List[List[FrozenSet[Edge]]]:
        """Return violating mapped-subgraph edge sets for each graph."""
        assert self.cannot_exist_features_vec is not None, 'FeasibilityEstimatorFeatureCannotExist is not fit'
        graphs = list(graphs)
        violating_sets: List[List[FrozenSet[Edge]]] = []
        for abstract_graph in self._abstract_graphs(graphs):
            graph_violations: List[FrozenSet[Edge]] = []
            for _, data in abstract_graph.interpretation_graph.nodes(data=True):
                if not self._is_forbidden_label(data.get("label")):
                    continue
                mapped_subgraph = get_mapped_subgraph(data)
                graph_violations.append(self._mapped_subgraph_edge_set(mapped_subgraph))
            violating_sets.append(graph_violations)
        return violating_sets

    def violating_node_labels_sets(self, graphs: Iterable) -> List[List[FrozenSet[Node]]]:
        """Return violating mapped-subgraph node-id sets for each graph."""
        assert self.cannot_exist_features_vec is not None, 'FeasibilityEstimatorFeatureCannotExist is not fit'
        graphs = list(graphs)
        violating_sets: List[List[FrozenSet[Node]]] = []
        for abstract_graph in self._abstract_graphs(graphs):
            graph_violations: List[FrozenSet[Node]] = []
            for _, data in abstract_graph.interpretation_graph.nodes(data=True):
                if not self._is_forbidden_label(data.get("label")):
                    continue
                mapped_subgraph = get_mapped_subgraph(data)
                graph_violations.append(self._mapped_subgraph_node_set(mapped_subgraph))
            violating_sets.append(graph_violations)
        return violating_sets


class FeasibilityEstimator(object):
    """Composite graph feasibility estimator.

    Sub-estimators are evaluated in order. During prediction, each estimator
    sees only the graphs that survived all previous checks, so later and more
    expensive estimators can short-circuit when earlier constraints fail.

    Args:
        feasibility_estimators: Sequence of sub-estimators to apply.
        parallel: Whether child estimators should default to parallel mode.
    """
    
    def __init__(self, feasibility_estimators, parallel=True):
        self.feasibility_estimators = feasibility_estimators
        self.set_parallel(parallel)        

    def set_parallel(self, parallel):
        """Set the parallel flag on all sub-estimators.

        Args:
            parallel: Parallel flag propagated to child estimators.
        """
        feasibility_estimators = []
        for feasibility_estimator in self.feasibility_estimators:
            feasibility_estimator.parallel = parallel
            feasibility_estimators.append(feasibility_estimator)
        self.feasibility_estimators = feasibility_estimators

    def __repr__(self):
        infos = ['feasibility_estimator_%d:%s'%(i,feasibility_estimator) for i,feasibility_estimator in enumerate(self.feasibility_estimators)]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def fit(self, graphs):
        """Fit all sub-estimators on the valid labeled subset.

        Args:
            graphs: Training graphs.

        Returns:
            FeasibilityEstimator: The fitted estimator.
        """
        graphs = filter_graphs_without_node_and_edge_label_attribute(graphs)
        self.feasibility_estimators = [feasibility_estimator.fit(graphs) for feasibility_estimator in self.feasibility_estimators]
        return self

    def _predict_estimator_on_indices(self, feasibility_estimator, graphs, indices):
        """Evaluate one estimator on a graph subset.

        Args:
            feasibility_estimator: Sub-estimator to evaluate.
            graphs: Full graph collection.
            indices: Indices selecting the subset to score.

        Returns:
            np.ndarray: Predictions for the selected graphs.
        """
        if hasattr(feasibility_estimator, 'predict_masked'):
            return np.asarray(feasibility_estimator.predict_masked(graphs, indices))
        selected_graphs = [graphs[idx] for idx in indices]
        return np.asarray(feasibility_estimator.predict(selected_graphs))

    def predict_masked(self, graphs, indices=None):
        """Predict feasibility for a subset of graphs and return a full mask.

        Args:
            graphs: Full graph collection.
            indices: Optional graph indices to evaluate. If omitted, all graphs
                are evaluated.

        Returns:
            np.ndarray: Boolean feasibility mask with length ``len(graphs)``.
            Entries outside ``indices`` remain ``False``.
        """
        if indices is None:
            indices = np.arange(len(graphs))
        else:
            indices = np.asarray(indices, dtype=int)

        preds = np.zeros(len(graphs), dtype=bool)
        if len(indices) == 0:
            return preds

        surviving_indices = indices.copy()
        preds[surviving_indices] = True
        for feasibility_estimator in self.feasibility_estimators:
            if len(surviving_indices) == 0:
                break
            estimator_preds = self._predict_estimator_on_indices(
                feasibility_estimator, graphs, surviving_indices
            ).astype(bool, copy=False)
            failed_indices = surviving_indices[np.logical_not(estimator_preds)]
            preds[failed_indices] = False
            surviving_indices = surviving_indices[estimator_preds]
        return preds

    def predict(self, graphs):
        """Predict feasibility for all graphs.

        Args:
            graphs: Graphs to evaluate.

        Returns:
            np.ndarray: Boolean feasibility mask.
        """
        return self.predict_masked(graphs)

    def violations(self, graphs):
        """Return per-estimator violation counts for each graph.

        Args:
            graphs: Graph collection to evaluate.

        Returns:
            np.ndarray: Matrix with shape
            ``(len(graphs), len(self.feasibility_estimators))`` where each
            column contains the violation magnitudes reported by one
            sub-estimator.
        """
        n_graphs = len(graphs)
        n_estimators = len(self.feasibility_estimators)
        if n_estimators == 0:
            return np.zeros((n_graphs, 0), dtype=int)
        preds = [
            np.asarray(feasibility_estimator.number_of_violations(graphs)).reshape(-1, 1)
            for feasibility_estimator in self.feasibility_estimators
        ]
        return np.hstack(preds)

    def number_of_violations(self, graphs):
        """Return total violation count per graph.

        Args:
            graphs: Graphs to evaluate.

        Returns:
            np.ndarray: Row-wise sums of :meth:`violations`.
        """
        preds = self.violations(graphs)
        return np.sum(preds, axis=1)

    def violating_edge_sets(self, graphs: Iterable) -> List[List[FrozenSet[Edge]]]:
        """Concatenate violating edge sets from child estimators that expose them."""
        graphs = list(graphs)
        violating_sets: List[List[FrozenSet[Edge]]] = [[] for _ in range(len(graphs))]
        for feasibility_estimator in self.feasibility_estimators:
            if not hasattr(feasibility_estimator, "violating_edge_sets"):
                continue
            estimator_violations = feasibility_estimator.violating_edge_sets(graphs)
            for graph_idx, graph_violations in enumerate(estimator_violations):
                violating_sets[graph_idx].extend(graph_violations)
        return violating_sets

    def violating_node_labels_sets(self, graphs: Iterable) -> List[List[FrozenSet[Node]]]:
        """Concatenate violating node-id sets from child estimators that expose them."""
        graphs = list(graphs)
        violating_sets: List[List[FrozenSet[Node]]] = [[] for _ in range(len(graphs))]
        for feasibility_estimator in self.feasibility_estimators:
            if not hasattr(feasibility_estimator, "violating_node_labels_sets"):
                continue
            estimator_violations = feasibility_estimator.violating_node_labels_sets(graphs)
            for graph_idx, graph_violations in enumerate(estimator_violations):
                violating_sets[graph_idx].extend(graph_violations)
        return violating_sets

    def filter(self, graphs, targets=None):
        """Keep only feasible graphs, optionally preserving targets.

        Args:
            graphs: Graphs to filter.
            targets: Optional target values aligned with ``graphs``.

        Returns:
            list | tuple[list, list]: Feasible graphs alone, or feasible graphs
            with aligned targets when ``targets`` is provided.
        """
        graphs = filter_graphs_without_node_and_edge_label_attribute(graphs)
        is_feasible = self.predict(graphs)
        selected_graphs = [graphs[idx] for idx in range(len(graphs)) if is_feasible[idx]==True]
        if targets is not None: 
            selected_targets = [targets[idx] for idx in range(len(targets)) if is_feasible[idx]==True]
            return selected_graphs, selected_targets
        return selected_graphs


def ConcreteFeasibilityEstimator(min_size=1, max_size=None):
    """Build the standard hand-coded feasibility estimator.

    Args:
        min_size: Minimum allowed number of nodes.
        max_size: Optional maximum allowed number of nodes.

    Returns:
        FeasibilityEstimator: Composite estimator with structural constraints.
    """
    feasibility_estimators = [FeasibilityEstimatorHasNodeAndEdgeLabelAttribute(),
                              FeasibilityEstimatorHasNoSelfLoops(),
                              FeasibilityEstimatorNumberOfNodesInRange(min_size=min_size, max_size=max_size),
                              FeasibilityEstimatorNumberOfEdgesInRange(min_size=1, max_size=None),
                              FeasibilityEstimatorIsConnected(number_connected_components=1)]
    return FeasibilityEstimator(feasibility_estimators=feasibility_estimators)


def ConcreteFeasibilityEstimatorObservedSize(node_quantile=None, edge_quantile=None):
    """Concrete feasibility estimator with size bounds learned during ``fit``.

    This avoids precomputing dataset min/max sizes before constructing the
    estimator.

    Args:
        node_quantile: Optional symmetric trimming quantile for node-count
            bounds.
        edge_quantile: Optional symmetric trimming quantile for edge-count
            bounds.

    Returns:
        FeasibilityEstimator: Composite estimator that learns node/edge size
        bounds from training graphs on ``fit``.
    """
    feasibility_estimators = [
        FeasibilityEstimatorHasNodeAndEdgeLabelAttribute(),
        FeasibilityEstimatorHasNoSelfLoops(),
        FeasibilityEstimatorNumberOfNodesInObservedRange(quantile=node_quantile),
        FeasibilityEstimatorNumberOfEdgesInObservedRange(quantile=edge_quantile),
        FeasibilityEstimatorIsConnected(number_connected_components=1),
    ]
    return FeasibilityEstimator(feasibility_estimators=feasibility_estimators)
