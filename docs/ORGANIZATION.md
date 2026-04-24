# abstractgraph-ml Organization

This document covers code organization, local setup, validation, and supporting
documentation for `abstractgraph-ml`.

For the semantic role of this repository, see [../README.md](../README.md).

## Package Layout

- `src/abstractgraph_ml/estimators.py`
  Scikit-style estimator wrappers around graph decomposition and vectorization
  pipelines.
- `src/abstractgraph_ml/neural.py`
  Neural estimators and input adapters.
- `src/abstractgraph_ml/feasibility.py`
  Graph admissibility checks and feasibility estimators.
- `src/abstractgraph_ml/importance.py`
  Utilities that map model relevance back to graph structure.
- `src/abstractgraph_ml/topk.py`
  Feature and operator ranking utilities.

## Documentation

- [README.md](README.md)
- [GRAPH_ESTIMATOR.md](GRAPH_ESTIMATOR.md)
- [FEASIBILITY.md](FEASIBILITY.md)

## Notebooks

- `notebooks/examples/` contains ML-facing notebooks updated to the split
  package names.
- [../notebooks/README.md](../notebooks/README.md) describes the notebook
  layout and bootstrap behavior.
- Example notebooks bootstrap imports and working directory automatically for
  the standard ecosystem layout.

## Dependency

- `abstractgraph`

## Local Validation

```bash
python -m pip install -e ../abstractgraph --no-deps
python -m pip install -e . --no-deps
python scripts/smoke_test.py
```
