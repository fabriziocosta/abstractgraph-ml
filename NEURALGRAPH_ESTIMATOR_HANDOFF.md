# NeuralGraphEstimator tuning handoff

Updated: 2026-11-11

## Aim

Find and validate the strongest configuration of the package's `NeuralGraphEstimator` for the PubChem graph classification task. Tune settings using training/inner validation data, compare candidates on identical outer folds, and use the random-forest graph kernel as a benchmark. Accuracy is the primary selection metric and ROC AUC is a secondary metric. The target is to match or exceed the kernel on fresh held-out data. Do not blend kernel predictions into the neural estimator.

## Task and data

- Dataset: PubChem assay **AID 492952**, 2,294 graphs.
- Target: active compounds are class `1` (positive); inactive compounds are class `0`.
- Original example split: stratified 80/20 split, `random_state=0` (1,835 train / 459 test).
- Kernel reference on that split: 300-tree `RandomForestClassifier`, using 2,048 graph-hash features (`nbits=11`), accuracy **0.773**, ROC AUC **0.835**.
- Same-fold references vary with the outer CV seed; use the matched RF row for each run rather than comparing CV results directly to the single-split score.

The loader sets the target explicitly in `abstractgraph_graphicalizer.chem.pubchem.PubChemAssayLoader`: active is 1 and inactive is 0.

## The class we are tuning

`NeuralGraphEstimator` is implemented in `src/abstractgraph_ml/neural.py`. Its default architecture is a hierarchical Transformer over AbstractGraph decompositions. It embeds mapped-subgraph elements, encodes each local subgraph, then applies a graph-level Transformer across the subgraph representations. The estimator also supports `architecture="gin"` for direct relation-aware message passing. Hierarchical mode supports node or edge tokens through `element_mode`.

The example notebook currently instantiates the hierarchical estimator in **node mode** with:

- `nbits=11`
- local Transformer: width 32, 4 heads, 1 layer, feed-forward width 64
- global Transformer: width 32, 4 heads, 1 layer, feed-forward width 64
- dropout 0.3; embedding dropout 0.3
- up to 200 epochs, batch size 64, validation split 0.1, early-stopping patience 30
- learning rate 1e-3, weight decay 0.01
- ReduceLROnPlateau: patience 8, factor 0.5, minimum learning rate 1e-5
- MPS device; seed and validation seed 4213; verbose progress enabled

This is a compact starting configuration, not yet a proven best configuration under nested CV.

## Results for NeuralGraphEstimator so far

| Evaluation | Neural configuration | Neural accuracy / ROC AUC | Matched kernel accuracy / ROC AUC |
| --- | --- | ---: | ---: |
| Original stratified holdout | Hierarchical Transformer, node mode, best single run | 0.732 / 0.771 | 0.773 / 0.835 |
| Original stratified holdout | Hierarchical Transformer, edge mode, width 32 | 0.725 / 0.766 | 0.773 / 0.835 |
| Five-fold OOF, node mode | Width 32, fixed 20 epochs | 0.704 / 0.760 | See same-fold RF row in notebook |
| Fresh five-fold OOF, edge mode | Width 32, local/global one layer, batch 64, early stopping | 0.700 / 0.733 | 0.744 / 0.812 |
| Original holdout | Relation-aware GIN, three-model ensemble | 0.684 / 0.748 | 0.773 / 0.835 |
| Original holdout | Relation-aware GIN, width 64, three layers, dropout 0.4, 88 epochs | 0.691 / 0.730 | 0.773 / 0.835 |

Other hierarchical Transformer trials on the original split included width 64, alternate hash widths (`nbits=10` and 14), latent mixup, class weighting, mapped atom/bond context, and multi-assay pretraining. The notebook records these individually; none established a repeatable improvement over the kernel. Single-holdout results are exploratory because that split has been consulted across experiments.

## Separate residual-MLP work

A 16-layer residual MLP on precomputed graph-hash vectors was explored in a different experiment. It is **not** `NeuralGraphEstimator` and is not part of the target configuration search. Its 128-epoch run and normalization/initialization/regularization/ensemble/hash-width trials remained below the kernel. A separate mixup sweep was interrupted after three outer folds when the scope was clarified; it is incomplete and must not be reported as a result. Do not use those residual-MLP settings as the selected `NeuralGraphEstimator` configuration.

## Current state

- The notebook's original `NeuralGraphEstimator` config above is the starting point.
- The direct graph-model comparisons available so far favor node mode over edge mode on the original holdout, but both node and edge modes trail the kernel in their available OOF comparisons.
- There is no validated best hyperparameter configuration for `NeuralGraphEstimator` yet. A controlled nested search is underway across five configurations: compact hierarchical node, stronger-regularized node, wider/deeper node, compact edge, and three-layer GIN. It uses three inner folds, five outer folds, inner-selected thresholds, early stopping, and a matched RF on the same outer folds.
- Completed fold 1: inner selection chose compact edge mode (best validation epochs 10, 39, and 33 across the inner folds). Outer score was 0.6776 accuracy / 0.7266 ROC AUC; matched RF scored 0.7320 / 0.7832. Fold 2 is running. Treat this as interim evidence only.
- The unrelated residual-MLP mixup run was interrupted after three outer folds and is incomplete. The Jupyter kernel is now occupied by the `NeuralGraphEstimator` search above.
- User instruction: skip TabPFN 2.6. No kernel/neural prediction blend.

## Next work

1. Finish the controlled search currently running. Based on its results, refine `NeuralGraphEstimator` hyperparameters (including width/layer counts, dropout, learning rate, weight decay, batch size, and early stopping) inside inner folds.
2. Use inner folds for every configuration and threshold decision. Compare on the same stratified outer folds against the matched 300-tree RF. Track accuracy, ROC AUC, selected epoch, and train/validation gap.
3. Keep the outer predictions untouched while selecting configurations. Re-evaluate the strongest candidate on fresh/repeated outer folds before calling it the best or claiming it beats the kernel.
4. Update this handoff and the training notebook with results. Commit and sync both the `abstractgraph-ml` submodule and the ecosystem repository.
