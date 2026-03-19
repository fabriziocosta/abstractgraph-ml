#!/usr/bin/env python
"""Provides scikit interface."""

import numpy as np
import networkx as nx
from abstractgraph.vectorize import AbstractGraphTransformer

def label_attribute_is_present(graph):
    for u in graph.nodes():
        if 'label' not in graph.nodes[u]:
            return False
    for u,v in graph.edges():
        if 'label' not in graph.edges[u,v]:
            return False
    return True

def filter_graphs_without_node_and_edge_label_attribute(graphs):
    return [graph for graph in graphs if label_attribute_is_present(graph)]

class FeasibilityEstimatorFromBooleanFunction(object):
    def __init__(self, boolean_function):
        self.boolean_function = boolean_function

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def fit(self, graphs):
        return self

    def number_of_violations(self, graphs):
        preds = np.array([self.boolean_function(graph) for graph in graphs]).astype(int)
        return preds

    def predict(self, graphs):
        preds = np.array([self.boolean_function(graph) for graph in graphs])
        return preds


def FeasibilityEstimatorHasNodeAndEdgeLabelAttribute():
    return FeasibilityEstimatorFromBooleanFunction(boolean_function=label_attribute_is_present)


def FeasibilityEstimatorHasNoSelfLoops():
    return FeasibilityEstimatorFromBooleanFunction(boolean_function=lambda graph:nx.number_of_selfloops(graph)==0)


def FeasibilityEstimatorNumberOfNodesInRange(min_size=1, max_size=None):
    def is_n_nodes_in_range(graph):
        number_of_nodes = nx.number_of_nodes(graph)
        if max_size is None: return number_of_nodes >= min_size
        return  min_size <= number_of_nodes <= max_size

    return FeasibilityEstimatorFromBooleanFunction(boolean_function=is_n_nodes_in_range)


def FeasibilityEstimatorNumberOfEdgesInRange(min_size=1, max_size=None):
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
    def __init__(self, numerical_function, quantile=None):
        self.numerical_function = numerical_function
        self.quantile = quantile

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def fit(self, graphs):
        vals = np.array([self.numerical_function(graph) for graph in graphs])
        if self.quantile is None:
            self.min = np.min(vals)
            self.max = np.max(vals)
        else:
            self.min = np.quantile(vals, self.quantile)
            self.max = np.quantile(vals, 1-self.quantile)
        return self

    def is_feasible(self, graph):
        val = self.numerical_function(graph)
        test = val >= self.min and val <= self.max 
        return test

    def size_of_violation(self, graph):
        val = self.numerical_function(graph)
        if val <= self.min: 
            return self.min - val 
        elif val >= self.max: 
            return val - self.max
        else:
            return 0

    def number_of_violations(self, graphs):
        preds = np.array([self.size_of_violation(graph) for graph in graphs]).astype(int)
        return preds

    def predict(self, graphs):
        preds = np.array([self.is_feasible(graph) for graph in graphs])
        return preds


class FeasibilityEstimatorIsConnected(object):
    def __init__(self, number_connected_components=1):
        self.number_connected_components = number_connected_components

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def fit(self, graphs):
        return self

    def number_of_violations(self, graphs):
        preds = np.array([nx.number_connected_components(graph)-1 for graph in graphs])
        return preds

    def predict(self, graphs):
        preds = np.array([nx.number_connected_components(graph)==self.number_connected_components for graph in graphs])
        return preds


class FeasibilityEstimatorFeatureMustExist(object):
    def __init__(self, decomposition_function=None, nbits=14, parallel=True, n_jobs=None):
        self.decomposition_function = decomposition_function
        self.nbits = nbits
        self.must_exist_features_vec = None
        self.parallel = parallel
        self.n_jobs = n_jobs

    def transform(self, graphs):
        """
        Convert a list of NetworkX graphs to a CSR feature matrix using the
        new QuotientGraph pipeline.  Remains sparse so downstream .dot() and
        .A calls behave exactly like before.
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
        data_mtx = self.transform(graphs).astype(bool)
        n_instances, n_features = data_mtx.shape
        # find all features that are always present
        exist_feats = data_mtx.sum(axis=0).A.flatten().astype(bool)
        self.must_exist_features_vec = np.logical_or(np.zeros(n_features).astype(bool), (exist_feats == n_instances))
        return self

    def number_of_violations(self, graphs):
        assert self.must_exist_features_vec is not None, 'FeasibilityEstimatorFeatureMustExist is not fit'
        data_mtx = self.transform(graphs)
        preds = data_mtx.dot(self.must_exist_features_vec.astype(int))
        return preds

    def predict(self, graphs):
        assert self.must_exist_features_vec is not None, 'FeasibilityEstimatorFeatureMustExist is not fit'
        data_mtx = self.transform(graphs).astype(bool)
        preds = data_mtx.dot(self.must_exist_features_vec)
        return preds


class FeasibilityEstimatorFeatureCannotExist(object):
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
        """
        Convert a list of NetworkX graphs to a CSR feature matrix using the
        new QuotientGraph pipeline.  Remains sparse so downstream .dot() and
        .A calls behave exactly like before.
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

    def fit(self, graphs):
        data_mtx = self.transform(graphs).astype(bool)
        # find all missing features
        exist_feats = data_mtx.astype(bool).sum(axis=0).A.flatten().astype(bool)
        self.cannot_exist_features_vec = np.logical_not(exist_feats)
        return self

    def number_of_violations(self, graphs):
        assert self.cannot_exist_features_vec is not None, 'FeasibilityEstimatorFeatureCannotExist is not fit'
        data_mtx = self.transform(graphs)
        preds = data_mtx.dot(self.cannot_exist_features_vec.astype(int))
        return preds

    def predict(self, graphs):
        assert self.cannot_exist_features_vec is not None, 'FeasibilityEstimatorFeatureCannotExist is not fit'
        data_mtx = self.transform(graphs).astype(bool)
        cannot_exist = data_mtx.dot(self.cannot_exist_features_vec)
        preds = np.logical_not(cannot_exist)
        return preds


class FeasibilityEstimator(object):
    """Composite graph feasibility estimator.

    Sub-estimators are evaluated in order. During prediction, each estimator
    sees only the graphs that survived all previous checks, so later and more
    expensive estimators can short-circuit when earlier constraints fail.
    """
    
    def __init__(self, feasibility_estimators, parallel=True):
        self.feasibility_estimators = feasibility_estimators
        self.set_parallel(parallel)        

    def set_parallel(self, parallel):
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
        graphs = filter_graphs_without_node_and_edge_label_attribute(graphs)
        self.feasibility_estimators = [feasibility_estimator.fit(graphs) for feasibility_estimator in self.feasibility_estimators]
        return self

    def _predict_estimator_on_indices(self, feasibility_estimator, graphs, indices):
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
        return self.predict_masked(graphs)

    def number_of_violations(self, graphs):
        preds = [feasibility_estimator.number_of_violations(graphs).reshape(-1,1) for feasibility_estimator in self.feasibility_estimators]
        preds = np.hstack(preds)
        preds = np.sum(preds, axis=1)
        return preds

    def filter(self, graphs, targets=None):
        graphs = filter_graphs_without_node_and_edge_label_attribute(graphs)
        is_feasible = self.predict(graphs)
        selected_graphs = [graphs[idx] for idx in range(len(graphs)) if is_feasible[idx]==True]
        if targets is not None: 
            selected_targets = [targets[idx] for idx in range(len(targets)) if is_feasible[idx]==True]
            return selected_graphs, selected_targets
        return selected_graphs


def ConcreteFeasibilityEstimator(min_size=1, max_size=None):
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
