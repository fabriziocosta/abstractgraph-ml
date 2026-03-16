# ML notebooks

This folder contains notebooks centered on estimators, importance analysis,
benchmarking, and model-selection workflows built on top of `abstractgraph`.

Layout:
- `examples/` for estimator-facing workflows
- `research/` reserved for ML-specific exploratory notebooks

Recommended example entry points:
- `examples/example_graph_estimator_supervised_and_unsupervised_pubchem.ipynb`
  for a compact supervised vs unsupervised `GraphEstimator` walkthrough on
  PubChem molecule graphs loaded through `abstractgraph-graphicalizer`

Bootstrap behavior:
- Example notebooks use `notebooks/_bootstrap.py` to locate the repo root.
- They prepend available sibling `src/` directories to `sys.path`.
- They normalize the working directory to the repo root so relative paths are
  consistent across Jupyter launch locations.
