"""ML-oriented AbstractGraph namespace."""

from abstractgraph_ml.estimators import GraphEstimator, IsolationForestProba
from abstractgraph_ml.feasibility import *  # noqa: F401,F403
from abstractgraph_ml.importance import *  # noqa: F401,F403
from abstractgraph_ml.topk import *  # noqa: F401,F403

_NEURAL_EXPORTS = {
    "NeuralGraphEstimator",
    "InputAdapterLinear",
    "InputAdapterFactorized",
}

_neural_import_error = None
try:
    from abstractgraph_ml.neural import InputAdapterFactorized, InputAdapterLinear, NeuralGraphEstimator
except (ImportError, OSError) as exc:
    _neural_import_error = exc

__all__ = [
    "GraphEstimator",
    "IsolationForestProba",
    "NeuralGraphEstimator",
    "InputAdapterLinear",
    "InputAdapterFactorized",
]


def __getattr__(name: str):
    if name in _NEURAL_EXPORTS and _neural_import_error is not None:
        raise ImportError(
            "Neural components require optional torch dependencies that could not be imported."
        ) from _neural_import_error
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
