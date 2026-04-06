"""ML-oriented AbstractGraph namespace.

Keep optional modules lazy so estimator-only workflows do not require torch or
other heavier visualization/top-k dependencies to import successfully.
"""

from __future__ import annotations

from abstractgraph_ml.estimators import GraphEstimator, IsolationForestProba
from abstractgraph_ml.feasibility import *  # noqa: F401,F403

_importance_import_error = None
_topk_import_error = None
_neural_import_error = None

_NEURAL_EXPORTS = {
    "NeuralGraphEstimator",
    "InputAdapterLinear",
    "InputAdapterFactorized",
}
_IMPORTANCE_EXPORTS = {
    "annotate_graph_node_saliency",
    "plot_graph_node_saliency",
    "plot_graph_node_saliency_with_estimator",
    "plot_feature_family_importance",
}
_TOPK_EXPORTS = {
    "make_topk_df",
    "TopKDecomposition",
}

__all__ = [
    "GraphEstimator",
    "IsolationForestProba",
    "NeuralGraphEstimator",
    "InputAdapterLinear",
    "InputAdapterFactorized",
    "annotate_graph_node_saliency",
    "plot_graph_node_saliency",
    "plot_graph_node_saliency_with_estimator",
    "plot_feature_family_importance",
    "make_topk_df",
    "TopKDecomposition",
]


def __getattr__(name: str):
    global _importance_import_error, _topk_import_error, _neural_import_error

    if name in _IMPORTANCE_EXPORTS:
        try:
            from abstractgraph_ml import importance as _importance

            value = getattr(_importance, name)
            globals()[name] = value
            return value
        except Exception as exc:
            _importance_import_error = exc
            raise ImportError(
                "Importance plotting helpers require optional visualization dependencies that could not be imported."
            ) from exc

    if name in _TOPK_EXPORTS:
        try:
            from abstractgraph_ml import topk as _topk

            value = getattr(_topk, name)
            globals()[name] = value
            return value
        except Exception as exc:
            _topk_import_error = exc
            raise ImportError(
                "Top-k helpers require optional dependencies that could not be imported."
            ) from exc

    if name in _NEURAL_EXPORTS:
        try:
            from abstractgraph_ml import neural as _neural

            value = getattr(_neural, name)
            globals()[name] = value
            return value
        except Exception as exc:
            _neural_import_error = exc
            raise ImportError(
                "Neural components require optional torch dependencies that could not be imported."
            ) from exc

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
