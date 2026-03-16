# GraphEstimator

`GraphEstimator` is the main classical ML pipeline in `abstractgraph-ml`.

It wraps the structural feature machinery from `abstractgraph` and exposes a
scikit-style estimator interface on top of it.

## Main Idea

`GraphEstimator` separates graph learning into three stages:

1. transform graphs into structural feature vectors
2. optionally reduce or re-embed those features through a manifold step
3. fit a downstream estimator on the raw structural features

This gives you a practical bridge from operator programs and vectorization to
standard ML models.

The implementation lives in
[estimators.py](/home/fabrizio/sync/Projects/AbstractGraphEcosystem/abstractgraph-ml/src/abstractgraph_ml/estimators.py).

## Pipeline Stages

### Transformer

The `transformer` is usually an `AbstractGraphTransformer` from `abstractgraph`.

Its job is to:

- apply a decomposition function to each graph
- vectorize the resulting abstract graph
- produce a 2D feature matrix suitable for ML

This is the structural entry point of the pipeline.

### Estimator

The `estimator` is any scikit-compatible downstream estimator.

Typical choices are:

- random forests
- linear models
- SVM-style estimators
- anomaly or outlier estimators

The estimator is always trained on the raw transformer output.

That detail matters: even when feature selection or manifold learning is used,
the predictive estimator still learns directly from the original structural
feature matrix.

### Manifold

The `manifold` is an optional transformer applied after feature extraction.

It is intended for:

- dimensionality reduction
- visualization
- alternate embeddings of the structural feature space

The default is `PCA()`.

This step is used by `transform(...)`, not as the main predictive surface.

## Feature Selection

`GraphEstimator` supports optional feature selection through
`n_selected_features`.

The ranking signal is taken from the downstream estimator when available:

- `feature_importances_`
- `coef_`

If the estimator exposes neither, feature selection is skipped.

Selection behavior:

- `None`: no selection
- integer: exact number of top features
- float in `(0, 1)`: fraction of input features
- float `>= 1`: interpreted as a feature count

Feature selection is mainly used before the manifold stage.

## Supervised And Unsupervised Modes

`GraphEstimator.fit(graphs, targets)` supports both:

- supervised mode when `targets` are provided
- unsupervised mode when `targets=None`

If no estimator is passed and `targets=None`, the class falls back to
`IsolationForestProba`.

`IsolationForestProba` is a small wrapper around `IsolationForest` that:

- fits an isolation forest on the graph features
- calibrates decision scores into probabilities using the empirical CDF of the
  training scores
- exposes `predict_proba(...)` and `classes_ = [0, 1]`

This keeps unsupervised workflows compatible with downstream code expecting a
classifier-like interface.

## Typical Usage

```python
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from abstractgraph.vectorize import AbstractGraphTransformer
from abstractgraph_ml.estimators import GraphEstimator

transformer = AbstractGraphTransformer(
    decomposition_function=decomposition_function,
    nbits=14,
)

estimator = GraphEstimator(
    transformer=transformer,
    estimator=RandomForestClassifier(random_state=0),
    manifold=PCA(n_components=2),
    n_selected_features=0.2,
)

estimator.fit(train_graphs, train_targets)
preds = estimator.predict(test_graphs)
probs = estimator.predict_proba(test_graphs)
embedding = estimator.transform(test_graphs)
```

## When To Use It

Use `GraphEstimator` when you want:

- a standard estimator API over structural graph features
- a simple path from decomposition functions to classical ML
- optional feature selection driven by model importances
- a manifold view of the learned feature space

It is the default estimator-facing abstraction in `abstractgraph-ml`.

## Related Components

- [FEASIBILITY.md](FEASIBILITY.md)
  for admissibility and structural validity checks
- `NeuralGraphEstimator`
  when you want a neural estimator path instead of a classical feature model
