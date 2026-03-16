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
- `number_of_violations(graphs)` sums violation magnitudes across the
  sub-estimators
- `filter(graphs, targets=None)` keeps only feasible graphs, optionally keeping
  targets aligned

This composite form is the main entry point for generation workflows.

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
filtered_graphs = feasibility.filter(candidate_graphs)
violation_sizes = feasibility.number_of_violations(candidate_graphs)
```

## Current Limitation

The module is already useful, but it is still fairly coarse:

- some estimators act like hard filters rather than rich diagnostics
- `number_of_violations(...)` is scalar and does not explain which constraint
  failed
- feature-based feasibility depends on hashed vectorization, so it is not yet a
  full structural explanation layer

That makes feasibility a strong practical component today, and a good candidate
for deeper constraint-reporting later.
