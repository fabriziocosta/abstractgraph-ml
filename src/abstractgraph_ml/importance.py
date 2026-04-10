"""
Graph importance annotation and plotting helpers extracted from the notebook.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import Normalize

from abstractgraph.display import display_grouped_graphs, get_color, stable_hash
from abstractgraph.feature_subgraphs import feature_subgraphs
from abstractgraph.graphs import AbstractGraph
from abstractgraph.labels import graph_hash_label_function_factory


def annotate_graph_node_saliency(
    graph: nx.Graph,
    graph_estimator: "GraphEstimator",
    *,
    node_agg: str = "max",
    edge_stat: str = "mean",
) -> Tuple[nx.Graph, Mapping[Any, float], Mapping[Tuple[Any, Any], float]]:
    """
    Return a copy of ``graph`` annotated with node/edge saliency derived from
    ``graph_estimator`` (which must already be fitted).

    The procedure is:
    1) Decompose the input graph with the estimator's transformer and apply the
       label function to interpretation nodes.
    2) Fetch the ranked feature ids from the estimator and convert ranks into
       linear scores in (0, 1], assigning higher scores to earlier ranks.
    3) For each base node, aggregate the scores of mapped interpretation nodes
       according to ``node_agg``.
    4) For each base edge, aggregate the endpoint scores according to
       ``edge_stat``.

    Args:
        graph: Input graph to annotate.
        graph_estimator: Fitted graph estimator used to rank features.
        node_agg: Aggregation strategy over interpretation nodes ("max", "mean", "sum").
        edge_stat: Edge aggregation ("mean", "min", "max").

    Returns:
        Tuple of annotated graph copy, node importance map, edge importance map.
    """
    if graph_estimator.transformer is None:
        raise ValueError("graph_estimator.transformer must be set to annotate graph.")
    decomposition_function = graph_estimator.transformer.decomposition_function
    if decomposition_function is None:
        raise ValueError("graph_estimator.transformer.decomposition_function must be set.")
    nbits = graph_estimator.transformer.nbits
    ranked_feature_ids = graph_estimator.get_ranked_feature_ids(fit_if_needed=False)

    ag = AbstractGraph(
        graph=graph, label_function=graph_hash_label_function_factory(nbits)
    )
    ag.create_default_interpretation_node()
    ag = decomposition_function(ag)
    ag.apply_label_function()

    inverse: Mapping[Any, set] = {node: set() for node in ag.base_graph.nodes()}
    for img_id, data in ag.interpretation_graph.nodes(data=True):
        mapped_subgraph = data.get("mapped_subgraph", data.get("association"))
        if mapped_subgraph is None:
            continue
        for pre_id in mapped_subgraph.nodes():
            if pre_id in inverse:
                inverse[pre_id].add(img_id)

    # Normalize ranked feature ids into descending scores in (0, 1],
    # giving earlier (more important) features larger weights.
    ranked = list(ranked_feature_ids)
    if ranked:
        n = len(ranked)
        # Map each feature label to a linear rank score: 1.0, 1 - 1/n, ..., 1/n.
        score_map = {lbl: (n - i) / n for i, lbl in enumerate(ranked)}
    else:
        score_map = {}

    node_score = {node: 0.0 for node in ag.base_graph.nodes()}
    for node, img_ids in inverse.items():
        vals = []
        for img_id in img_ids:
            lbl = ag.interpretation_graph.nodes[img_id].get("label")
            if lbl in score_map:
                vals.append(score_map[lbl])
        if not vals:
            node_score[node] = 0.0
        else:
            if node_agg == "mean":
                node_score[node] = float(np.mean(vals))
            elif node_agg == "sum":
                node_score[node] = float(np.sum(vals))
            else:
                node_score[node] = float(np.max(vals))

    edge_scores: dict[Tuple[Any, Any], float] = {}
    for u, v in ag.base_graph.edges():
        a, b = node_score.get(u, 0.0), node_score.get(v, 0.0)
        if edge_stat == "min":
            s = min(a, b)
        elif edge_stat == "max":
            s = max(a, b)
        else:
            s = 0.5 * (a + b)
        edge_scores[(u, v)] = s

    G_out = graph.copy()
    for n in G_out.nodes():
        G_out.nodes[n]["importance"] = float(node_score.get(n, 0.0))
    for u, v in G_out.edges():
        s = edge_scores.get((u, v))
        if s is None and not G_out.is_directed():
            s = edge_scores.get((v, u), 0.0)
        G_out.edges[u, v]["importance"] = float(0.0 if s is None else s)

    return G_out, node_score, edge_scores


def plot_graph_node_saliency(
    graph: nx.Graph,
    *,
    cmap: str = "YlOrRd",
    color_value_range: Tuple[float, float] = (0.18, 1.0),
    width_range: Tuple[float, float] = (1.5, 9.0),
    node_size_range: Tuple[float, float] = (140.0, 280.0),
    size: Tuple[float, float] = (7, 6),
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """
    Plot a graph that already exposes ``node['importance']`` / ``edge['importance']``.

    Args:
        graph: Graph with importance values on nodes/edges.
        cmap: Matplotlib colormap name.
        color_value_range: Range of normalized color values to use.
        width_range: Min/max edge widths.
        node_size_range: Min/max node sizes.
        size: Figure size when creating a new axis.
        ax: Optional axis to draw into.
        show: Whether to show the figure when a new axis is created.

    Returns:
        Matplotlib axis containing the plot.
    """
    edges = list(graph.edges())
    edge_vals = [float(graph.edges[e].get("importance", 0.0)) for e in edges]
    if edge_vals:
        vmin, vmax = float(np.min(edge_vals)), float(np.max(edge_vals))
        if np.isclose(vmax, vmin):
            vmin = 0.0
    else:
        vmin = vmax = 0.0

    norm = Normalize(vmin=vmin, vmax=vmax if vmax > vmin else 1.0)
    try:
        cmap_obj = plt.colormaps.get_cmap(cmap)
    except Exception:
        cmap_obj = plt.get_cmap(cmap)

    rlo, rhi = color_value_range
    rlo = max(0.0, min(1.0, float(rlo)))
    rhi = max(0.0, min(1.0, float(rhi)))
    if rhi < rlo:
        rlo, rhi = rhi, rlo

    if vmax > vmin:
        s_norm = [float(norm(s)) for s in edge_vals]
    else:
        s_norm = [0.0 for _ in edge_vals]
    # Use a mild power-law expansion so mid-range importance reads more clearly.
    edge_strength = [float(np.power(x, 0.6)) for x in s_norm]
    col_vals = [rlo + x * (rhi - rlo) for x in edge_strength]
    colors = [cmap_obj(cv) for cv in col_vals]

    wmin, wmax = width_range
    if vmax > vmin:
        widths = [wmin + (wmax - wmin) * strength for strength in edge_strength]
    else:
        widths = [wmin for _ in edge_vals]

    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=size)
        created_ax = True
    pos = nx.kamada_kawai_layout(graph) if graph.number_of_nodes() > 0 else {}
    node_colors = []
    node_vals = []
    for n, d in graph.nodes(data=True):
        lbl = d.get("label")
        if lbl is not None:
            node_colors.append(get_color(lbl, cmap_name="hsv"))
        else:
            node_colors.append(get_color(stable_hash(str(n)), cmap_name="hsv"))
        node_vals.append(float(d.get("importance", 0.0)))

    if node_vals:
        nmin, nmax = float(np.min(node_vals)), float(np.max(node_vals))
        if np.isclose(nmax, nmin):
            node_strength = [1.0 for _ in node_vals] if nmax > 0 else [0.0 for _ in node_vals]
        else:
            node_norm = Normalize(vmin=nmin, vmax=nmax)
            node_strength = [float(np.power(node_norm(v), 0.55)) for v in node_vals]
    else:
        node_strength = []
    ns_min, ns_max = node_size_range
    node_sizes = [
        ns_min + (ns_max - ns_min) * strength
        for strength in (node_strength or [0.0 for _ in graph.nodes()])
    ]

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="black",
        linewidths=0.9,
        ax=ax,
    )
    if edges:
        nx.draw_networkx_edges(
            graph, pos, edgelist=edges, edge_color=colors, width=widths, ax=ax
        )

    ax.set_title("")
    ax.axis("off")
    if created_ax and show:
        plt.show()
    return ax


def plot_graph_node_saliency_with_estimator(
    graph: nx.Graph,
    graph_estimator: "GraphEstimator",
    *,
    node_agg: str = "max", #node_agg: Aggregation strategy over interpretation nodes ("max", "mean", "sum").
    edge_stat: str = "mean", #edge_stat: Edge aggregation ("mean", "min", "max").
    cmap: str = "YlOrRd",
    color_value_range: Tuple[float, float] = (0.18, 1.0),
    width_range: Tuple[float, float] = (1.5, 9.0),
    node_size_range: Tuple[float, float] = (140.0, 280.0),
    size: Tuple[float, float] = (7, 6),
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> Tuple[plt.Axes, Mapping[Any, float], Mapping[Tuple[Any, Any], float]]:
    """
    Convenience helper that annotates and plots a graph in one call.

    Args:
        graph: Input graph to annotate and plot.
        graph_estimator: Fitted graph estimator used to rank features.
        node_agg: Aggregation strategy over interpretation nodes ("max", "mean", "sum").
        edge_stat: Edge aggregation ("mean", "min", "max").
        cmap: Matplotlib colormap name.
        color_value_range: Range of normalized color values to use.
        width_range: Min/max edge widths.
        node_size_range: Min/max node sizes.
        size: Figure size when creating a new axis.
        ax: Optional axis to draw into.
        show: Whether to show the figure when a new axis is created.

    Returns:
        Tuple of axis, node importance map, and edge importance map.
    """
    annotated_graph, node_scores, edge_scores = annotate_graph_node_saliency(
        graph=graph,
        graph_estimator=graph_estimator,
        node_agg=node_agg,
        edge_stat=edge_stat,
    )
    ax = plot_graph_node_saliency(
        graph=annotated_graph,
        cmap=cmap,
        color_value_range=color_value_range,
        width_range=width_range,
        node_size_range=node_size_range,
        size=size,
        ax=ax,
        show=show,
    )
    return ax, node_scores, edge_scores


def plot_graph_node_saliency_grid(
    graphs: Iterable[nx.Graph],
    graph_estimator: "GraphEstimator",
    *,
    n_elements_per_row: int = 5,
    figsize_per_graph: Tuple[float, float] = (3.0, 3.0),
    suptitle: Optional[str] = None,
    titles: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> Tuple[Optional[plt.Figure], Any]:
    """
    Render a grid of annotated graphs using ``plot_graph_node_saliency_with_estimator``.

    Args:
        graphs: Iterable of graphs to annotate and plot.
        graph_estimator: Fitted graph estimator used to rank features.
        n_elements_per_row: Number of graphs per row in the grid.
        figsize_per_graph: Size per subplot.
        suptitle: Optional figure title.
        titles: Optional per-graph titles.
        **kwargs: Passed through to ``plot_graph_node_saliency_with_estimator``.

    Returns:
        Tuple of figure and axes (or (None, None) when no graphs).
    """
    graphs_list = list(graphs or [])
    if not graphs_list:
        print("No graphs to plot.")
        return None, None

    cols = max(1, int(n_elements_per_row))
    rows = int(math.ceil(len(graphs_list) / float(cols)))
    fig_w = max(1.0, cols * float(figsize_per_graph[0]))
    fig_h = max(1.0, rows * float(figsize_per_graph[1]))
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))

    if isinstance(axes, (list, tuple)):
        axes_list = list(axes)
    else:
        axes_list = list(np.atleast_1d(axes).ravel())

    for i, G in enumerate(graphs_list):
        if i >= len(axes_list):
            break
        ax = axes_list[i]
        plot_graph_node_saliency_with_estimator(
            graph=G,
            graph_estimator=graph_estimator,
            ax=ax,
            show=False,
            **kwargs,
        )
        if titles and i < len(titles):
            ax.set_title(titles[i])
        else:
            ax.set_title(ax.get_title() or f"Graph {i}")

    for j in range(i + 1, len(axes_list)):
        axes_list[j].axis("off")

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    plt.show()
    return fig, axes


def display_topk_feature_subgraphs(
    graphs: Iterable[nx.Graph],
    graph_estimator: "GraphEstimator",
    *,
    top_k_features: int = 8,
    n_graphs_per_line: int = 6,
    size: Tuple[float, float] = (2.2, 2.2),
    title_formatter: Optional[Callable[[Any], str]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """Display grouped representative subgraphs for the estimator's top features.

    The function vectorizes ``graphs`` once, finds one graph containing each top
    ranked feature id, collects unique mapped subgraphs for those feature ids,
    and renders them in grouped boxes using ``display_grouped_graphs``.

    Args:
        graphs: Candidate graphs to search for top-feature occurrences.
        graph_estimator: Fitted graph estimator with a configured transformer.
        top_k_features: Maximum number of ranked feature ids to visualize.
        n_graphs_per_line: Number of subgraph cells per row in the grouped view.
        size: Per-cell subplot size in inches.
        title_formatter: Optional formatter for group titles.
        show: Whether to call ``plt.show()`` in the grouped display.

    Returns:
        Matplotlib figure, or ``None`` when no matching feature subgraphs exist.
    """
    if graph_estimator.transformer is None:
        raise ValueError("graph_estimator.transformer must be set to display top features.")
    decomposition_function = graph_estimator.transformer.decomposition_function
    if decomposition_function is None:
        raise ValueError("graph_estimator.transformer.decomposition_function must be set.")
    nbits = graph_estimator.transformer.nbits

    graphs_list = list(graphs or [])
    if not graphs_list:
        print("No graphs were provided.")
        return None

    ranked_feature_ids = list(graph_estimator.get_ranked_feature_ids(fit_if_needed=False))
    X = np.asarray(graph_estimator.transformer.transform(graphs_list))

    selected_feature_ids = []
    selected_graph_ids = set()
    for feature_id in ranked_feature_ids:
        graph_indices = np.flatnonzero(X[:, feature_id] > 0)
        if graph_indices.size == 0:
            continue
        selected_feature_ids.append(feature_id)
        selected_graph_ids.add(int(graph_indices[0]))
        if len(selected_feature_ids) >= int(top_k_features):
            break

    selected_graphs = [graphs_list[graph_idx] for graph_idx in sorted(selected_graph_ids)]
    feature_subgraph_map = feature_subgraphs(
        selected_graphs,
        decomposition_function=decomposition_function,
        nbits=nbits,
    )
    feature_groups = [
        (feature_id, feature_subgraph_map[feature_id])
        for feature_id in selected_feature_ids
        if feature_subgraph_map.get(feature_id)
    ]

    if not feature_groups:
        print("No subgraphs were found for the top-ranked features.")
        return None

    if title_formatter is None:
        title_formatter = lambda feature_id: f"feature_id={feature_id}"

    return display_grouped_graphs(
        feature_groups,
        n_graphs_per_line=n_graphs_per_line,
        size=size,
        title_formatter=title_formatter,
        show=show,
    )


__all__ = [
    "annotate_graph_node_saliency",
    "display_topk_feature_subgraphs",
    "plot_graph_node_saliency",
    "plot_graph_node_saliency_with_estimator",
    "plot_graph_node_saliency_grid",
]
