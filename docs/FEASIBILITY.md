# Feasibility

`abstractgraph_ml.feasibility` provides a constraint layer for graph datasets
and graph generation workflows.

The main idea is simple:

- predictors answer whether a graph is desirable or likely
- feasibility estimators answer whether a graph is admissible

In practice, the module is useful in three places:

- dataset sanitation before training
- filtering or ranking generated candidates
- enforcing coarse structural priors learned from a training set

## Core Interface

Most feasibility objects expose a small common interface:

- `fit(graphs)`
- `predict(graphs)`
- `predict_masked(graphs, indices=None)` on the composite estimator
- `violations(graphs)` on the composite estimator
- `number_of_violations(graphs)`
- `filter(graphs, targets=None)` on the composite estimator

This makes them easy to use as a final validation layer around other
estimators or generators.

## Constraint Types

The module currently mixes two kinds of constraints.

### Structural constraints

These are graph-level validity checks such as:

- node and edge labels must be present
- no self-loops
- node count within a fixed range
- edge count within a fixed range
- connectivity constraints

Relevant constructors:

- `FeasibilityEstimatorHasNodeAndEdgeLabelAttribute()`
- `FeasibilityEstimatorHasNoSelfLoops()`
- `FeasibilityEstimatorNumberOfNodesInRange(...)`
- `FeasibilityEstimatorNumberOfEdgesInRange(...)`
- `FeasibilityEstimatorIsConnected(...)`

### Learned or observed constraints

These learn admissibility from training graphs:

- observed node-count range
- observed edge-count range
- features that must always exist
- features that must never exist

Relevant constructors:

- `FeasibilityEstimatorNumberOfNodesInObservedRange(...)`
- `FeasibilityEstimatorNumberOfEdgesInObservedRange(...)`
- `FeasibilityEstimatorFeatureMustExist(...)`
- `FeasibilityEstimatorFeatureCannotExist(...)`

The feature-based estimators use
`AbstractGraphTransformer` from `abstractgraph.vectorize` to turn graphs into
sparse structural feature vectors before learning which features are always
present or always absent.

## Composite Estimator

`FeasibilityEstimator(...)` combines multiple constraints into one object.

Its behavior is:

- `predict(graphs)` returns `True` only when all sub-estimators accept the
  graph
- sub-estimators are evaluated in order, from first to last
- after each step, only the surviving graphs are passed to the next
  sub-estimator
- evaluation stops early when no graphs remain feasible
- `predict_masked(graphs, indices=None)` runs the same logic on a selected
  subset of graph indices and returns a full-length boolean mask
- `violations(graphs)` returns one column per sub-estimator, so you can see
  which constraints each graph violates and by how much
  the matrix shape is `(n_graphs, n_estimators)`
  columns follow the order of `feasibility_estimators`
- `number_of_violations(graphs)` sums violation magnitudes across the
  sub-estimators
- `filter(graphs, targets=None)` keeps only feasible graphs, optionally keeping
  targets aligned

This composite form is the main entry point for generation workflows.
Ordering therefore matters: put cheap structural checks first and expensive
feature-based checks later so the composite can reject invalid graphs as early
as possible.

## Concrete Presets

Two convenience constructors package the most common checks.

`ConcreteFeasibilityEstimator(...)`

- requires node and edge labels
- forbids self-loops
- constrains node count
- constrains edge count
- requires connectedness

`ConcreteFeasibilityEstimatorObservedSize(...)`

- uses the same structural checks
- learns node and edge size bounds from training graphs during `fit`

## Role In Generation

Feasibility already matters most in `abstractgraph-generative`.

There it acts as:

- a post-generation filter
- a rejection mechanism inside candidate search
- a way to enforce dataset-derived structural validity

This is stronger than a generic utility function. It is the ecosystem's
current notion of admissibility over graph space.

## Minimal Example

```python
from abstractgraph_ml.feasibility import (
    ConcreteFeasibilityEstimatorObservedSize,
)

feasibility = ConcreteFeasibilityEstimatorObservedSize(
    node_quantile=0.05,
    edge_quantile=0.05,
)

feasibility.fit(train_graphs)

is_feasible = feasibility.predict(candidate_graphs)
subset_mask = feasibility.predict_masked(candidate_graphs, indices=[0, 3, 5])
filtered_graphs = feasibility.filter(candidate_graphs)
violation_matrix = feasibility.violations(candidate_graphs)
violation_sizes = feasibility.number_of_violations(candidate_graphs)
```

Here `violation_matrix[i, j]` is the violation count contributed by the
`j`-th sub-estimator on the `i`-th graph, and `violation_sizes[i]` is the sum
across that row.

## Current Limitation

The module is already useful, but it is still fairly coarse:

- some estimators act like hard filters rather than rich diagnostics
- feature-based feasibility depends on hashed vectorization, so it is not yet a
  full structural explanation layer

That makes feasibility a strong practical component today, and a good candidate
for deeper constraint-reporting later.
