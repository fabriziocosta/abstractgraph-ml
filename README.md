# abstractgraph-ml

`abstractgraph-ml` provides the estimator and analysis layer built on top of
`abstractgraph`.

It is the package to use when you want to turn Abstract Graph decompositions
into predictive models, evaluate them, and inspect feature relevance.

## Ecosystem

This repo is one part of a three-repo stack:

- `abstractgraph`
  Path: `/home/fabrizio/work/abstractgraph`
- `abstractgraph-ml`
  Path: `/home/fabrizio/work/abstractgraph-ml`
- `abstractgraph-generative`
  Path: `/home/fabrizio/work/abstractgraph-generative`

See [ECOSYSTEM.md](ECOSYSTEM.md) for the dependency graph and install order.

## Package layout

- `src/abstractgraph_ml/estimators.py`
  scikit-style estimator wrapper and `IsolationForestProba`
- `src/abstractgraph_ml/neural.py`
  neural graph estimators and input adapters
- `src/abstractgraph_ml/feasibility.py`
  structural and feature feasibility checks
- `src/abstractgraph_ml/importance.py`
  saliency and subgraph-importance visualization
- `src/abstractgraph_ml/topk.py`
  top-k operator/feature selection utilities

## Documentation

- [docs/README.md](docs/README.md)
- [docs/FEASIBILITY.md](docs/FEASIBILITY.md)
- [ECOSYSTEM.md](ECOSYSTEM.md)

## Notebooks

- `notebooks/examples/` contains copied ML-facing notebooks updated to the
  split package names.
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
