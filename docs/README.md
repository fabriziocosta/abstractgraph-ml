# abstractgraph-ml docs

`abstractgraph-ml` contains the model-facing layer built on top of
`abstractgraph`.

## Scope

This repo owns:
- classical estimator wrappers
- neural estimators
- feasibility estimators
- importance/saliency utilities
- top-k feature ranking and selection helpers

## Module map

- `abstractgraph_ml.estimators`
  `GraphEstimator`, `IsolationForestProba`
- `abstractgraph_ml.neural`
  neural graph estimators and input adapters
- `abstractgraph_ml.feasibility`
  structural and feature-based feasibility filters
- `abstractgraph_ml.importance`
  saliency and feature-to-graph visualization helpers
- `abstractgraph_ml.topk`
  top-k operator/feature selection workflows

## Related docs

- [GRAPH_ESTIMATOR.md](GRAPH_ESTIMATOR.md)
- [FEASIBILITY.md](FEASIBILITY.md)

## Dependencies

- `abstractgraph`

## Ecosystem

Sibling repositories:

- `abstractgraph`
  Path: `/home/fabrizio/work/abstractgraph`
- `abstractgraph-ml`
  Path: `/home/fabrizio/work/abstractgraph-ml`
- `abstractgraph-generative`
  Path: `/home/fabrizio/work/abstractgraph-generative`

See [../ECOSYSTEM.md](../ECOSYSTEM.md) for install order and dependency
direction.

## Typical workflow

1. Define a decomposition in `abstractgraph.operators`.
2. Build an `AbstractGraphTransformer` or `AbstractGraphNodeTransformer`.
3. Train a `GraphEstimator` or `NeuralGraphEstimator`.
4. Inspect results with `importance` or `topk`.

## Notebooks

- `notebooks/examples/` contains estimator-facing examples copied from the
  original monorepo and updated to the split package names.
