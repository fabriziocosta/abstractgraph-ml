"""Smoke test for the extracted abstractgraph-ml package."""

from __future__ import annotations

from abstractgraph_ml.estimators import GraphEstimator, IsolationForestProba


def main() -> None:
    """Run a minimal ML import smoke test."""
    print("estimators", GraphEstimator.__name__, IsolationForestProba.__name__)


if __name__ == "__main__":
    main()
