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
- `examples/example_graph_label_repair_zinc.ipynb`
  for a ZINC molecular graph label-repair workflow that corrupts held-out node
  and edge labels, restores them with `GraphLabelRepairEstimator`, and reports
  node, edge, and total precision/recall/F1
- `examples/example_graph_estimator_rhopca_preprocessor_comparison.ipynb`
  for a fixed `neighborhood(radius=(0, 2))` graph transformer benchmark comparing
  raw features against SVD and supervised `RhoPCA` preprocessing across linear,
  random forest, and RBF SVM estimators

Bootstrap behavior:
- Example notebooks use `notebooks/_bootstrap.py` to locate the repo root.
- They prepend available sibling `src/` directories to `sys.path`.
- They normalize the working directory to the repo root so relative paths are
  consistent across Jupyter launch locations.
