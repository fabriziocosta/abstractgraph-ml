"""
Utilities for building Top-K decomposition operators and visualizing their ROC-AUC
behavior on downstream classification tasks.

The helpers in this module move the heavy lifting out of the demo notebooks so
that Top-K experiments can be scripted or re-used elsewhere.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

from abstractgraph import operators as ag_ops
from abstractgraph.xml import operator_to_xml_string, register_from_module
from abstractgraph.operators import (
    forward_compose,
    neighborhood,
    select_top_by_feature_ranking,
)
from abstractgraph.vectorize import AbstractGraphTransformer
from abstractgraph_ml.estimators import GraphEstimator

TopKDecomposition = Any  # Operator callables live in operator.py and are duck-typed.


def _clone_estimator_instance(
    estimator: Any, estimator_kwargs: Mapping[str, Any]
) -> Any:
    if isinstance(estimator, BaseEstimator):
        est = clone(estimator)
    else:
        est = deepcopy(estimator)
    if estimator_kwargs and hasattr(est, "set_params"):
        est.set_params(**estimator_kwargs)
    return est


def _instantiate_estimator(estimator: Any, estimator_kwargs: Mapping[str, Any]) -> Any:
    if isinstance(estimator, BaseEstimator):
        return _clone_estimator_instance(estimator, estimator_kwargs)
    if callable(estimator):
        if estimator_kwargs:
            return estimator(**estimator_kwargs)
        return estimator()
    return _clone_estimator_instance(estimator, estimator_kwargs)


def _build_graph_estimator(
    base_estimator: GraphEstimator,
    transformer: AbstractGraphTransformer,
    estimator_kwargs: Mapping[str, Any],
) -> GraphEstimator:
    base_model = base_estimator.estimator
    if base_model is None:
        raise ValueError("graph_estimator.estimator must be set to evaluate Top-K.")
    model = _clone_estimator_instance(base_model, estimator_kwargs)
    return GraphEstimator(
        transformer=transformer,
        estimator=model,
        manifold=base_estimator.manifold,
        n_selected_features=base_estimator.n_selected_features,
    )


def make_topk_df(
    graphs: Sequence[Any],
    targets: Sequence[Any],
    *,
    graph_estimator: Optional[GraphEstimator] = None,
    nbits: Optional[int] = None,
    decomposition_function: Optional[TopKDecomposition] = None,
    top_ks: Sequence[int] = (5, 10, 30, 100, 300),
    estimator: Optional[Callable[..., Any]] = None,
    estimator_kwargs: Optional[MutableMapping[str, Any]] = None,
    n_splits: int = 5,
    return_dense: Optional[bool] = None,
    use_permutation: bool = False,
    random_state: int = 42,
) -> Tuple[List[TopKDecomposition], List[int], Dict[str, Any]]:
    """
    Build Top-K operators with stability selection over feature importances.

    Returns the Top-K decomposition functions, the ranked feature ids, and
    diagnostics about the ranking process.

    When ``graph_estimator`` is provided, its transformer supplies ``nbits`` and
    ``decomposition_function`` (unless overridden), and its estimator is used to
    rank features.
    """

    if graph_estimator is not None:
        base_transformer = graph_estimator.transformer
        if decomposition_function is None:
            decomposition_function = base_transformer.decomposition_function
        if nbits is None:
            nbits = base_transformer.nbits
        if return_dense is None:
            return_dense = base_transformer.return_dense
        n_jobs = getattr(base_transformer, "n_jobs", -1)
    else:
        if decomposition_function is None:
            decomposition_function = neighborhood()
        if nbits is None:
            nbits = 14
        if return_dense is None:
            return_dense = True
        n_jobs = -1

    if estimator_kwargs is None:
        estimator_kwargs = {}
    estimator_kwargs = dict(estimator_kwargs)
    apply_defaults = graph_estimator is None or estimator is not None
    if apply_defaults:
        estimator_kwargs.setdefault("random_state", random_state)
        estimator_kwargs.setdefault("n_jobs", -1)
        if len(set(targets)) == 2:
            estimator_kwargs.setdefault("class_weight", "balanced")

    if graph_estimator is not None and estimator is None:
        if graph_estimator.estimator is None:
            raise ValueError("graph_estimator.estimator must be set to rank features.")
        build_estimator = lambda: _clone_estimator_instance(
            graph_estimator.estimator, estimator_kwargs
        )
    else:
        if estimator is None:
            estimator = ExtraTreesClassifier
        build_estimator = lambda: _instantiate_estimator(estimator, estimator_kwargs)

    vec = AbstractGraphTransformer(
        nbits=nbits,
        decomposition_function=decomposition_function,
        return_dense=return_dense,
        n_jobs=n_jobs,
    )
    X = vec.fit_transform(graphs)
    y = np.asarray(targets)

    if return_dense:
        col_sums = X.sum(axis=0)
    else:
        col_sums = np.array(X.sum(axis=0)).ravel()
    n_features = int(col_sums.shape[0])
    reserved = {0, 1}
    active_mask = (col_sums > 0) & ~np.isin(np.arange(n_features), list(reserved))
    active_idx = np.where(active_mask)[0]
    X_active = X[:, active_idx]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_imps = []
    for tr, te in skf.split(X_active, y):
        clf = build_estimator()
        clf.fit(X_active[tr], y[tr])
        imp = getattr(clf, "feature_importances_", None)
        if imp is None or use_permutation:
            r = permutation_importance(
                clf,
                X_active[te],
                y[te],
                n_repeats=5,
                random_state=random_state,
                n_jobs=-1,
            )
            imp = r.importances_mean
        fold_imps.append(imp)
    fold_imps = np.vstack(fold_imps)
    mean_imp_active = fold_imps.mean(axis=0)

    ranks = np.argsort(-fold_imps, axis=1)
    pos = np.zeros_like(fold_imps, dtype=int)
    for f, r in enumerate(ranks):
        pos[f, r] = np.arange(r.size)
    median_rank_active = np.median(pos, axis=0)

    if return_dense:
        pres_counts_active = (X_active > 0).astype(int).sum(axis=0)
    else:
        pres_counts_active = np.array((X_active > 0).astype(int).sum(axis=0)).ravel()
    n_samples = X_active.shape[0]
    freq_weight_active = np.sqrt(pres_counts_active / max(1, n_samples))

    scores_active = mean_imp_active * freq_weight_active

    valid_active = mean_imp_active > 0
    order_active = np.lexsort(
        (median_rank_active[valid_active], -scores_active[valid_active])
    )
    ranked_feature_ids = [
        int(active_idx[vi]) for vi in np.where(valid_active)[0][order_active]
    ]

    register_from_module(ag_ops)
    max_available = len(ranked_feature_ids)
    topk_dfs: List[TopKDecomposition] = []
    topk_xml: List[str] = []
    for k in top_ks:
        kk = min(int(k), max_available)
        df_top = forward_compose(
            decomposition_function,
            select_top_by_feature_ranking(
                ranked_features=ranked_feature_ids, max_num=kk
            ),
        )
        topk_dfs.append(df_top)
        topk_xml.append(operator_to_xml_string(df_top))

    diagnostics = {
        "active_idx": active_idx,
        "mean_importance_active": mean_imp_active,
        "median_rank_active": median_rank_active,
        "freq_weight_active": freq_weight_active,
        "ranked_feature_ids": ranked_feature_ids,
        "topk_xml": topk_xml,
        "nbits": nbits,
        "n_splits": n_splits,
    }
    return topk_dfs, ranked_feature_ids, diagnostics


def compute_topk_roc_results(
    graphs: Sequence[Any],
    targets: Sequence[Any],
    *,
    topk_dfs: Sequence[TopKDecomposition],
    top_ks: Sequence[int],
    graph_estimator: Optional[GraphEstimator] = None,
    nbits: Optional[int] = None,
    estimator_factory: Optional[Callable[..., Any]] = None,
    performance_fn: Callable[..., Any],
    estimator_kwargs: Optional[Mapping[str, Any]] = None,
    performance_kwargs: Optional[Mapping[str, Any]] = None,
    vectorizer_kwargs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate ROC-AUC distributions for each Top-K decomposition.

    Parameters
    ----------
    graph_estimator
        Optional initialized :class:`GraphEstimator`. When provided, its
        transformer configuration supplies ``nbits`` and the estimator used
        inside ``performance_fn``.
    estimator_factory
        Callable receiving the fitted :class:`AbstractGraphTransformer` and
        returning an estimator compatible with ``performance_fn``. Ignored
        when ``graph_estimator`` is provided.
    performance_fn
        Callable with signature similar to ``predictive_performance_estimate``.
    """

    if len(topk_dfs) != len(top_ks):
        raise ValueError("topk_dfs and top_ks must have the same length")

    estimator_kwargs = dict(estimator_kwargs or {})
    performance_kwargs = dict(performance_kwargs or {})
    if graph_estimator is not None:
        base_transformer = graph_estimator.transformer
        if nbits is None:
            nbits = base_transformer.nbits
        vectorizer_defaults = {
            "return_dense": base_transformer.return_dense,
            "n_jobs": base_transformer.n_jobs,
        }
    else:
        if estimator_factory is None:
            raise ValueError("estimator_factory is required when graph_estimator is None.")
        if nbits is None:
            raise ValueError("nbits is required when graph_estimator is None.")
        vectorizer_defaults = {"return_dense": True, "n_jobs": -1}
    vec_kwargs = {**vectorizer_defaults, **(vectorizer_kwargs or {})}

    q25_list: List[float] = []
    med_list: List[float] = []
    q75_list: List[float] = []
    times: List[float] = []
    all_scores: List[Iterable[float]] = []

    for df_k, k in zip(topk_dfs, top_ks):
        vec = AbstractGraphTransformer(
            nbits=nbits, decomposition_function=df_k, **vec_kwargs
        )
        if graph_estimator is not None:
            estimator = _build_graph_estimator(graph_estimator, vec, estimator_kwargs)
        else:
            estimator = estimator_factory(vec, **estimator_kwargs)
        scores, mean, std, elapsed, avg_mistakes = performance_fn(
            estimator, graphs, targets, **performance_kwargs
        )
        _ = (mean, std, avg_mistakes)  # keep compatibility but unused at module level
        s = np.array(scores, dtype=float).ravel()
        if s.size:
            q25, med, q75 = np.quantile(s, [0.25, 0.5, 0.75])
        else:
            q25 = med = q75 = float("nan")
        q25_list.append(float(q25))
        med_list.append(float(med))
        q75_list.append(float(q75))
        times.append(float(elapsed))
        all_scores.append(scores)

    top_ks_list = list(top_ks)
    try:
        x_log = np.log(np.asarray(top_ks_list, dtype=float))
        auc_median = float(np.trapezoid(med_list, x=x_log))
    except Exception:
        auc_median = float("nan")

    return {
        "top_ks": top_ks_list,
        "q25": q25_list,
        "median": med_list,
        "q75": q75_list,
        "times": times,
        "scores": all_scores,
        "auc_median": auc_median,
        "nbits": nbits,
        "vectorizer_kwargs": vec_kwargs,
        "estimator_kwargs": estimator_kwargs,
        "performance_kwargs": performance_kwargs,
    }


def estimate_topk_auc_mean(
    graphs: Sequence[Any],
    targets: Sequence[Any],
    *,
    topk_df: TopKDecomposition,
    graph_estimator: GraphEstimator,
    performance_fn: Callable[..., Any],
    estimator_kwargs: Optional[Mapping[str, Any]] = None,
    performance_kwargs: Optional[Mapping[str, Any]] = None,
    vectorizer_kwargs: Optional[Mapping[str, Any]] = None,
) -> float:
    """
    Estimate the mean ROC-AUC for a single Top-K decomposition.

    This helper is intended for repeated calls to build a distribution of
    average scores (e.g., to compute q25/q75 across runs).
    """

    estimator_kwargs = dict(estimator_kwargs or {})
    performance_kwargs = dict(performance_kwargs or {})
    base_transformer = graph_estimator.transformer
    vec_defaults = {
        "return_dense": base_transformer.return_dense,
        "n_jobs": base_transformer.n_jobs,
    }
    vec_kwargs = {**vec_defaults, **(vectorizer_kwargs or {})}
    vec = AbstractGraphTransformer(
        nbits=base_transformer.nbits,
        decomposition_function=topk_df,
        **vec_kwargs,
    )
    estimator = _build_graph_estimator(graph_estimator, vec, estimator_kwargs)
    scores, mean, std, elapsed, avg_mistakes = performance_fn(
        estimator, graphs, targets, **performance_kwargs
    )
    _ = (mean, std, elapsed, avg_mistakes)
    s = np.array(scores, dtype=float).ravel()
    if s.size:
        return float(np.mean(s))
    return float("nan")


def plot_topk_roc_curve(
    results: Mapping[str, Any],
    *,
    show: bool = True,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    plot_iqr: bool = True,
) -> plt.Axes:
    """Plot IQR band (25-75%) and median ROC-AUC over Top-K."""

    top_ks = results["top_ks"]
    q25 = results["q25"]
    med = results["median"]
    q75 = results["q75"]
    auc_median = results.get("auc_median")
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    try:
        color = next(ax._get_lines.prop_cycler)["color"]
    except Exception:
        color = None

    if plot_iqr:
        ax.fill_between(
            top_ks,
            q25,
            q75,
            color=(color if color else "C0"),
            alpha=0.2,
            label=(None if label else "ROC-AUC IQR"),
        )

    if label:
        if auc_median is not None and np.isfinite(auc_median):
            label_text = f"{label} \n(AUC_med={auc_median:.3f})"
        else:
            label_text = label
    else:
        if auc_median is not None and np.isfinite(auc_median):
            label_text = f"ROC-AUC median \n(AUC_med={auc_median:.3f})"
        else:
            label_text = "ROC-AUC median"

    ax.plot(top_ks, med, linewidth=2.5, marker="o", color=color, label=label_text)
    ax.set_xlabel("Top-K selected labels")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Top-K ROC-AUC vs K")
    ax.set_xscale("log")

    try:
        ax.set_xticks(top_ks)
        ax.set_xticklabels(
            [str(int(k)) if float(k).is_integer() else str(k) for k in top_ks]
        )
    except Exception:
        pass

    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=min(max(1, len(labels)), 4),
            frameon=False,
        )
        try:
            ax.figure.subplots_adjust(bottom=0.25)
        except Exception:
            pass

    if show:
        plt.show()
    return ax


def plot_topk_roc_curves(
    graphs: Sequence[Any],
    targets: Sequence[Any],
    *,
    dfs: Sequence[TopKDecomposition],
    labels: Sequence[str],
    graph_estimator: Optional[GraphEstimator] = None,
    nbits: Optional[int] = None,
    top_ks: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
    estimator_factory: Optional[Callable[..., Any]] = None,
    performance_fn: Callable[..., Any],
    estimator_kwargs: Optional[Mapping[str, Any]] = None,
    performance_kwargs: Optional[Mapping[str, Any]] = None,
    make_topk_kwargs: Optional[Mapping[str, Any]] = None,
    plot_kwargs: Optional[Mapping[str, Any]] = None,
    show: bool = True,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Axes, List[Mapping[str, Any]]]:
    """
    Convenience wrapper that ranks features per decomposition, evaluates ROC curves,
    and overlays them on a shared axis.

    When ``graph_estimator`` is provided, its transformer and estimator are used
    in place of ``nbits``/``estimator_factory``.
    """

    if len(dfs) != len(labels):
        raise ValueError("dfs and labels must have the same length")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    plot_kwargs = dict(plot_kwargs or {})
    results_list: List[Mapping[str, Any]] = []
    base_make_kwargs = dict(make_topk_kwargs or {})
    axis_topks = base_make_kwargs.get("top_ks", top_ks)
    if graph_estimator is None:
        if estimator_factory is None:
            raise ValueError("estimator_factory is required when graph_estimator is None.")
        if nbits is None:
            nbits = 14

    for df_i, label in zip(dfs, labels):
        mk_kwargs = dict(base_make_kwargs)
        mk_kwargs.update(
            {
                "top_ks": axis_topks,
                "decomposition_function": df_i,
            }
        )
        if graph_estimator is not None:
            mk_kwargs["graph_estimator"] = graph_estimator
        else:
            mk_kwargs.setdefault("nbits", nbits)
        topk_dfs, _, _ = make_topk_df(
            graphs,
            targets,
            **mk_kwargs,
        )
        if graph_estimator is not None:
            results = compute_topk_roc_results(
                graphs,
                targets,
                topk_dfs=topk_dfs,
                top_ks=mk_kwargs["top_ks"],
                graph_estimator=graph_estimator,
                nbits=mk_kwargs.get("nbits"),
                performance_fn=performance_fn,
                estimator_kwargs=estimator_kwargs,
                performance_kwargs=performance_kwargs,
            )
        else:
            results = compute_topk_roc_results(
                graphs,
                targets,
                topk_dfs=topk_dfs,
                top_ks=mk_kwargs["top_ks"],
                nbits=mk_kwargs["nbits"],
                estimator_factory=estimator_factory,
                performance_fn=performance_fn,
                estimator_kwargs=estimator_kwargs,
                performance_kwargs=performance_kwargs,
            )
        results_list.append(results)
        plot_topk_roc_curve(results, show=False, ax=ax, label=label, **plot_kwargs)

    try:
        ax.set_xscale("log")
        ax.set_xticks(axis_topks)
        ax.set_xticklabels(
            [str(int(k)) if float(k).is_integer() else str(k) for k in axis_topks]
        )
    except Exception:
        pass

    handles, labels_ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=min(max(1, len(labels_)), 4),
            frameon=False,
        )
        try:
            ax.figure.subplots_adjust(bottom=0.25)
        except Exception:
            pass

    if show:
        plt.show()
    return ax, results_list

__all__ = [
    "make_topk_df",
    "compute_topk_roc_results",
    "estimate_topk_auc_mean",
    "plot_topk_roc_curve",
    "plot_topk_roc_curves",
]
