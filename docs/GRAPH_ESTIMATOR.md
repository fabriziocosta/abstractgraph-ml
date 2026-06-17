# GraphEstimator

`GraphEstimator` is the main classical ML pipeline in `abstractgraph-ml`.

It wraps the structural feature machinery from `abstractgraph` and exposes a
scikit-style estimator interface on top of it.

## Main Idea

`GraphEstimator` separates graph learning into four stages:

1. transform graphs into structural feature vectors
2. optionally preprocess those features before prediction
3. fit a downstream estimator on the preprocessed structural features
4. optionally reduce or re-embed estimator features through a postprocessor

This gives you a practical bridge from operator programs and vectorization to
standard ML models.

The implementation lives in
[estimators.py](../src/abstractgraph_ml/estimators.py).

## Pipeline Stages

### Transformer

The `transformer` is usually an `AbstractGraphTransformer` from `abstractgraph`.

Its job is to:

- apply a decomposition function to each graph
- vectorize the resulting abstract graph
- produce a 2D feature matrix suitable for ML

This is the structural entry point of the pipeline.

### Preprocessor

The `preprocessor` is an optional scikit-compatible transformer applied after
graph vectorization and before the downstream estimator.

Use it when the predictive model should operate on a transformed feature space,
for example:

- `PCA`
- `TruncatedSVD`
- `StandardScaler`
- supervised transforms such as `RhoPCA`

The preprocessor is fit on the graph transformer's feature matrix. If
`targets` are passed to `GraphEstimator.fit(...)`, they are also passed to the
preprocessor, so supervised transformers can learn from labels.

`predict(...)` and `predict_proba(...)` use the same fitted preprocessor before
calling the downstream estimator.

### Estimator

The `estimator` is any scikit-compatible downstream estimator.

Typical choices are:

- random forests
- linear models
- SVM-style estimators
- anomaly or outlier estimators

The estimator is trained on the preprocessor output when `preprocessor` is
provided. Otherwise, it is trained directly on the raw transformer output.

That detail matters: `preprocessor` is part of the predictive path, while
`postprocessor` is not.

### Postprocessor

The `postprocessor` is an optional transformer applied after feature extraction
and preprocessing.

It is intended for:

- dimensionality reduction
- visualization
- alternate embeddings of the structural feature space

The default is a sparse-friendly truncated SVD helper,
`DropFirstTruncatedSVD()`.

This step is used by `transform(...)`, not as the main predictive surface.

`manifold` is still accepted as a backward-compatible alias for
`postprocessor`, but new code should use `postprocessor`.

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

Feature selection is computed from the downstream estimator and is mainly used
before the postprocessor stage. When a preprocessor is present, selected feature
indices refer to the preprocessed feature space.

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
    postprocessor=PCA(n_components=2),
    n_selected_features=0.2,
)

estimator.fit(train_graphs, train_targets)
preds = estimator.predict(test_graphs)
probs = estimator.predict_proba(test_graphs)
embedding = estimator.transform(test_graphs)
```

### Predict On Reduced Dimensions

Use `preprocessor` when the downstream estimator should train and predict on a
reduced representation.

```python
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from abstractgraph_ml.estimators import GraphEstimator

estimator = GraphEstimator(
    transformer=transformer,
    estimator=LogisticRegression(max_iter=1000),
    preprocessor=PCA(n_components=32, random_state=0),
    postprocessor=PCA(n_components=2, random_state=0),
)

estimator.fit(train_graphs, train_targets)
preds = estimator.predict(test_graphs)
```

For supervised preprocessing, pass a transformer whose `fit` or `fit_transform`
accepts `y`:

```python
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from abstractgraph_ml.estimators import GraphEstimator
from abstractgraph_ml.rho_pca import RhoPCA

estimator = GraphEstimator(
    transformer=transformer,
    estimator=LogisticRegression(max_iter=1000),
    preprocessor=RhoPCA(n_components=16, target_label=1, background_label=0),
    postprocessor=PCA(n_components=2, random_state=0),
)

estimator.fit(train_graphs, train_targets)
```

## When To Use It

Use `GraphEstimator` when you want:

- a standard estimator API over structural graph features
- a simple path from decomposition functions to classical ML
- a predictive preprocessing stage before a linear or nonlinear estimator
- optional feature selection driven by model importances
- a postprocessed view of the learned feature space

It is the default estimator-facing abstraction in `abstractgraph-ml`.

## Related Components

- [FEASIBILITY.md](FEASIBILITY.md)
  for admissibility and structural validity checks
- `NeuralGraphEstimator`
  when you want a neural estimator path instead of a classical feature model
