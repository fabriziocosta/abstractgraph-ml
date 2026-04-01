# abstractgraph-ml

`abstractgraph-ml` provides the estimator and analysis layer built on top of
`abstractgraph`.

It is the package to use when you want to turn Abstract Graph decompositions
into predictive models, evaluate them, and inspect feature relevance.

At a high level, `abstractgraph` defines the representation and vectorization
machinery, while `abstractgraph-ml` adds model-facing components that operate
on those structural features.

## Ecosystem

This repo is one part of a three-repo stack:

- `abstractgraph`
  Path: `/home/fabrizio/work/abstractgraph`
- `abstractgraph-ml`
  Path: `/home/fabrizio/work/abstractgraph-ml`
- `abstractgraph-generative`
  Path: `/home/fabrizio/work/abstractgraph-generative`

See [ECOSYSTEM.md](ECOSYSTEM.md) for the dependency graph and install order.

## Main Components

### Estimators

[estimators.py](src/abstractgraph_ml/estimators.py) provides the main
scikit-style estimator layer.

This is the entry point when you want to:

- apply an `abstractgraph` decomposition and vectorization pipeline
- train a classical ML model on graph-level features
- expose a familiar `fit` / `predict` style interface

It includes:

- `GraphEstimator`
  the main wrapper that combines decomposition, vectorization, and a downstream
  estimator
- `IsolationForestProba`
  a small utility estimator for anomaly-style workflows

### Neural Models

[neural.py](src/abstractgraph_ml/neural.py) contains neural estimators and
input adapters.

This layer matters when you want to move beyond a plain fixed feature matrix
while still using the `AbstractGraph` feature pipeline as the structural input
stage.

It includes:

- `NeuralGraphEstimator`
  the main neural estimator wrapper
- `InputAdapterLinear`
  a simple adapter for flat graph feature inputs
- `InputAdapterFactorized`
  an adapter for factorized or structured neural input layouts

### Feasibility

[feasibility.py](src/abstractgraph_ml/feasibility.py) defines the ecosystem's
constraint layer for graph admissibility.

This module is useful for:

- filtering invalid graphs before training
- learning coarse admissibility from observed datasets
- rejecting or scoring generated candidates in downstream generative workflows

It contains:

- simple structural feasibility checks
- observed-range estimators learned during `fit`
- feature-based feasibility estimators built on top of vectorization
- motif-level diagnostics for estimators that can report violating edge sets
  and violating node-id sets
- `FeasibilityEstimator`
  a composite constraint object that combines multiple checks and evaluates
  them in order on the surviving graphs only

See [docs/FEASIBILITY.md](docs/FEASIBILITY.md) for a fuller overview.

### Importance

[importance.py](src/abstractgraph_ml/importance.py) provides the interpretive
layer that maps model relevance back onto graph structure.

Use it when you want to answer questions such as:

- which nodes or edges matter most for a prediction
- which structural features dominate a fitted model
- how a ranked feature can be visualized as recurring subgraph structure

Key helper:

- `display_topk_feature_subgraphs(...)`
  vectorizes a graph set once, finds graphs that contain the estimator's top
  ranked hashed features, recovers unique mapped subgraphs for those feature
  ids, and renders them grouped by `feature_id`

### Top-k Selection

[topk.py](src/abstractgraph_ml/topk.py) provides feature and operator ranking
utilities.

This is the module to use when you want to:

- reduce a large structural feature space
- keep only the most informative hashed features
- drive operator selection or simplified downstream pipelines

## Typical Workflow

1. Define an operator program in `abstractgraph`.
2. Build a graph transformer or estimator around that decomposition.
3. Train a classical or neural model in `abstractgraph-ml`.
4. Optionally apply feasibility filtering to constrain the admissible graph set.
5. Inspect salient features or subgraphs with the importance utilities.
6. Use top-k selection if you need a reduced structural feature space.

## Documentation

- [docs/README.md](docs/README.md)
- [docs/FEASIBILITY.md](docs/FEASIBILITY.md)
- [ECOSYSTEM.md](ECOSYSTEM.md)

## Notebooks

- `notebooks/examples/` contains copied ML-facing notebooks updated to the
  split package names.
- [notebooks/README.md](notebooks/README.md) describes the notebook layout and
  bootstrap behavior.
- Example notebooks now bootstrap their imports and working directory
  automatically for the standard ecosystem layout, so they can be launched
  from the repo root, the notebook directory, or the workspace root.

## Dependency

- `abstractgraph`

## Local validation

```bash
python -m pip install -e ../abstractgraph --no-deps
python -m pip install -e . --no-deps
python scripts/smoke_test.py
```
