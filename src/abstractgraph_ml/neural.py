"""Hierarchical Transformer models over AbstractGraph decompositions.

Each input graph is decomposed into an :class:`abstractgraph.graphs.AbstractGraph`.
Mapped subgraphs are encoded as permutation-invariant element sets, and their
interpretation-node embeddings are combined by a graph-level Transformer.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abstractgraph.graphs import AbstractGraph, graph_to_abstract_graph, get_mapped_subgraph
from abstractgraph.hashing import hash_bounded


Tensor = torch.Tensor
TaskMode = Literal["classification", "regression"]
ElementMode = Literal["node", "edge"]


def _default_device() -> torch.device:
    """Select an available accelerator, including Apple's Metal backend."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _label_id(value: Any, nbits: int, *, interpretation_hash: bool = False) -> int:
    """Map labels into the embedding table, reserving 0 for padding and 1 for missing."""
    if value is None:
        return 1
    size = 1 << nbits
    if interpretation_hash and isinstance(value, (int, np.integer)) and 1 < int(value) < size:
        return int(value)
    return int(hash_bounded(value, nbits=nbits))


def _numeric_attribute(value: Any, *, context: str) -> np.ndarray:
    if value is None:
        return np.zeros((0,), dtype=np.float32)
    try:
        result = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} attribute must be a numeric vector.") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{context} attribute must contain only finite values.")
    return result


def _node_attribute(data: Dict[str, Any]) -> np.ndarray:
    if data.get("attribute") is not None:
        return _numeric_attribute(data.get("attribute"), context="Node")
    if any(name in data for name in ("atomic_num", "formal_charge", "aromatic")):
        return np.asarray(
            [
                float(data.get("atomic_num", 0.0)) / 100.0,
                float(data.get("formal_charge", 0.0)) / 4.0,
                float(bool(data.get("aromatic", False))),
            ],
            dtype=np.float32,
        )
    return np.zeros((0,), dtype=np.float32)


def _edge_attribute(data: Dict[str, Any]) -> np.ndarray:
    if data.get("attribute") is not None:
        return _numeric_attribute(data.get("attribute"), context="Edge")
    if any(name in data for name in ("bond_order", "aromatic")):
        return np.asarray(
            [
                float(data.get("bond_order", 1.0)) / 3.0,
                float(bool(data.get("aromatic", False))),
            ],
            dtype=np.float32,
        )
    return np.zeros((0,), dtype=np.float32)


def _graph_to_input(
    graph: nx.Graph,
    decomposition_function: Callable[[AbstractGraph], AbstractGraph],
    nbits: int,
    element_mode: ElementMode,
) -> Dict[str, Any]:
    abstract_graph = graph_to_abstract_graph(graph, decomposition_function, nbits=nbits)
    if not isinstance(abstract_graph, AbstractGraph):
        raise TypeError("decomposition_function must return an AbstractGraph.")

    interpretations: List[Dict[str, Any]] = []
    node_attribute_widths: List[int] = []
    edge_attribute_widths: List[int] = []
    for _, interpretation_data in abstract_graph.interpretation_graph.nodes(data=True):
        mapped = get_mapped_subgraph(interpretation_data)
        label = _label_id(interpretation_data.get("label"), nbits, interpretation_hash=True)
        element_labels: List[int] = []
        element_attributes: List[np.ndarray] = []
        endpoint_labels: List[Tuple[int, int]] = []
        endpoint_attributes: List[Tuple[np.ndarray, np.ndarray]] = []

        if mapped is not None:
            for node, data in mapped.nodes(data=True):
                node_label = _label_id(data.get("label"), nbits)
                node_attribute = _node_attribute(data)
                node_attribute_widths.append(node_attribute.size)

            if element_mode == "node":
                for _, data in mapped.nodes(data=True):
                    element_labels.append(_label_id(data.get("label"), nbits))
                    attribute = _node_attribute(data)
                    element_attributes.append(attribute)
            else:
                for source, target, data in mapped.edges(data=True):
                    element_labels.append(_label_id(data.get("label"), nbits))
                    attribute = _edge_attribute(data)
                    edge_attribute_widths.append(attribute.size)
                    element_attributes.append(attribute)
                    source_data = mapped.nodes[source]
                    target_data = mapped.nodes[target]
                    endpoint_labels.append(
                        (_label_id(source_data.get("label"), nbits), _label_id(target_data.get("label"), nbits))
                    )
                    source_attribute = _node_attribute(source_data)
                    target_attribute = _node_attribute(target_data)
                    endpoint_attributes.append((source_attribute, target_attribute))

        interpretations.append(
            {
                "label": label,
                "element_labels": element_labels,
                "element_attributes": element_attributes,
                "endpoint_labels": endpoint_labels,
                "endpoint_attributes": endpoint_attributes,
            }
        )

    return {
        "interpretations": interpretations,
        "node_attribute_widths": node_attribute_widths,
        "edge_attribute_widths": edge_attribute_widths,
    }


def _graph_to_gin_input(graph: nx.Graph, nbits: int) -> Dict[str, Any]:
    """Encode a labeled molecular graph for batched message passing."""
    nodes = list(graph.nodes(data=True))
    node_ids = {node: index for index, (node, _) in enumerate(nodes)}
    labels = np.zeros((len(nodes),), dtype=np.int64)
    attributes = np.zeros((len(nodes), 4), dtype=np.float32)
    for index, (node, data) in enumerate(nodes):
        labels[index] = _label_id(data.get("label"), nbits)
        attributes[index] = (
            float(data.get("atomic_num", 0.0)) / 100.0,
            float(data.get("formal_charge", 0.0)) / 4.0,
            float(bool(data.get("aromatic", False))),
            float(graph.degree[node]) / 8.0,
        )

    relation_ids = {"single": 0, "double": 1, "triple": 2, "aromatic": 3}
    edges: List[Tuple[int, int, int]] = []
    for source, target, data in graph.edges(data=True):
        relation = str(data.get("label", "")).lower()
        if relation not in relation_ids:
            bond_order = float(data.get("bond_order", 1.0))
            relation = "aromatic" if bool(data.get("aromatic", False)) or bond_order == 1.5 else (
                "triple" if bond_order >= 3 else "double" if bond_order >= 2 else "single"
            )
        edges.append((node_ids[source], node_ids[target], relation_ids[relation]))
    edge_array = np.asarray(edges, dtype=np.int64).reshape(-1, 3)
    return {"node_labels": labels, "node_attributes": attributes, "edges": edge_array}


def _infer_width(widths: Sequence[int], name: str) -> int:
    observed = {int(width) for width in widths if width > 0}
    if len(observed) > 1:
        raise ValueError(f"Inconsistent {name} attribute widths: {sorted(observed)}.")
    return next(iter(observed), 0)


def _attribute_tensor(values: Sequence[np.ndarray], width: int, *, context: str) -> Tensor:
    output = np.zeros((len(values), width), dtype=np.float32)
    for row, value in enumerate(values):
        if value.size == 0:
            continue
        if value.size != width:
            raise ValueError(
                f"{context} attribute width changed: expected {width}, received {value.size}."
            )
        output[row] = value
    return torch.as_tensor(output, dtype=torch.float32)


class _SummaryQueryLayer(nn.Module):
    """A Transformer-style block where one summary query reads a fixed set."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        summary: Tensor,
        elements: Tensor,
        key_padding_mask: Tensor,
        has_elements: Tensor,
    ) -> Tensor:
        # The query can also retain its own state. Element tokens are never queries
        # and are deliberately not updated or given positional encodings.
        query = self.query_norm(summary)
        memory = self.memory_norm(elements)
        update, _ = self.attention(
            query,
            memory,
            memory,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        update = update * has_elements[:, None, None]
        summary = summary + update
        return summary + self.ff(self.ff_norm(summary))


class _HierarchicalGraphModel(nn.Module):
    def __init__(
        self,
        *,
        nbits: int,
        element_mode: ElementMode,
        node_attribute_dim: int,
        edge_attribute_dim: int,
        local_d_model: int,
        local_nhead: int,
        local_num_layers: int,
        local_dim_feedforward: int,
        global_d_model: int,
        global_nhead: int,
        global_num_layers: int,
        global_dim_feedforward: int,
        dropout: float,
        embedding_dropout: float,
        output_dim: int,
    ) -> None:
        super().__init__()
        vocab_size = 1 << nbits
        self.element_mode = element_mode
        self.local_d_model = local_d_model
        self.global_d_model = global_d_model
        self.node_attribute_dim = node_attribute_dim
        self.edge_attribute_dim = edge_attribute_dim
        self.embedding_dropout = nn.Dropout(embedding_dropout)

        self.interpretation_embedding = nn.Embedding(vocab_size, local_d_model, padding_idx=0)
        self.element_label_embedding = nn.Embedding(vocab_size, local_d_model, padding_idx=0)
        self.node_attribute_projection = (
            nn.Linear(node_attribute_dim, local_d_model, bias=False) if node_attribute_dim else None
        )
        self.edge_attribute_projection = (
            nn.Linear(edge_attribute_dim, local_d_model, bias=False) if edge_attribute_dim else None
        )
        self.node_projection = nn.Linear(local_d_model, local_d_model)
        self.edge_projection = nn.Linear(local_d_model * 2, local_d_model)
        self.local_layers = nn.ModuleList(
            _SummaryQueryLayer(local_d_model, local_nhead, local_dim_feedforward, dropout)
            for _ in range(local_num_layers)
        )
        self.local_norm = nn.LayerNorm(local_d_model)
        self.to_global = nn.Linear(local_d_model, global_d_model)
        self.graph_token = nn.Parameter(torch.zeros(1, 1, global_d_model))
        nn.init.trunc_normal_(self.graph_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=global_d_model,
            nhead=global_nhead,
            dim_feedforward=global_dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.global_transformer = nn.TransformerEncoder(
            layer, num_layers=global_num_layers, enable_nested_tensor=False
        )
        self.global_norm = nn.LayerNorm(global_d_model)
        self.head = nn.Linear(global_d_model, output_dim)

    def _node_features(self, labels: Tensor, attributes: Tensor) -> Tensor:
        values = self.element_label_embedding(labels)
        if self.node_attribute_projection is not None:
            values = values + self.node_attribute_projection(attributes)
        return self.embedding_dropout(self.node_projection(values))

    def _encode_interpretations(self, items: Sequence[Dict[str, Any]]) -> Tensor:
        """Encode a batch of ragged mapped subgraphs in parallel."""
        device = self.graph_token.device
        if not items:
            return self.graph_token.new_empty((0, self.local_d_model))

        counts = [len(item["element_labels"]) for item in items]
        max_elements = max(1, max(counts))
        item_count = len(items)
        item_labels = torch.as_tensor([item["label"] for item in items], dtype=torch.long, device=device)
        element_labels_array = np.zeros((item_count, max_elements), dtype=np.int64)
        element_padding_array = np.ones((item_count, max_elements), dtype=np.bool_)

        if self.element_mode == "node":
            element_attributes_array = np.zeros(
                (item_count, max_elements, self.node_attribute_dim), dtype=np.float32
            )
            for row, item in enumerate(items):
                count = counts[row]
                if not count:
                    continue
                element_labels_array[row, :count] = item["element_labels"]
                element_padding_array[row, :count] = False
                for column, values in enumerate(item["element_attributes"]):
                    if values.size:
                        element_attributes_array[row, column] = values
            element_labels = torch.as_tensor(element_labels_array, dtype=torch.long, device=device)
            element_attributes = torch.as_tensor(element_attributes_array, dtype=torch.float32, device=device)
            elements = self._node_features(element_labels, element_attributes)
        else:
            endpoint_labels_array = np.zeros((item_count, max_elements, 2), dtype=np.int64)
            edge_attributes_array = np.zeros(
                (item_count, max_elements, self.edge_attribute_dim), dtype=np.float32
            )
            endpoint_attributes_array = np.zeros(
                (item_count, max_elements, 2, self.node_attribute_dim), dtype=np.float32
            )
            for row, item in enumerate(items):
                count = counts[row]
                if not count:
                    continue
                element_labels_array[row, :count] = item["element_labels"]
                endpoint_labels_array[row, :count] = item["endpoint_labels"]
                element_padding_array[row, :count] = False
                for column, values in enumerate(item["element_attributes"]):
                    if values.size:
                        edge_attributes_array[row, column] = values
                for column, pair in enumerate(item["endpoint_attributes"]):
                    for endpoint, values in enumerate(pair):
                        if values.size:
                            endpoint_attributes_array[row, column, endpoint] = values
            element_labels = torch.as_tensor(element_labels_array, dtype=torch.long, device=device)
            edge_attributes = torch.as_tensor(edge_attributes_array, dtype=torch.float32, device=device)
            endpoint_labels = torch.as_tensor(endpoint_labels_array, dtype=torch.long, device=device)
            endpoint_attributes = torch.as_tensor(endpoint_attributes_array, dtype=torch.float32, device=device)
            source = self._node_features(endpoint_labels[:, :, 0], endpoint_attributes[:, :, 0])
            target = self._node_features(endpoint_labels[:, :, 1], endpoint_attributes[:, :, 1])
            # Averaging makes undirected edge tokens independent of endpoint order.
            endpoints = (source + target) / 2.0
            edge_labels = self.element_label_embedding(element_labels)
            if self.edge_attribute_projection is not None:
                edge_labels = edge_labels + self.edge_attribute_projection(edge_attributes)
            elements = self.edge_projection(torch.cat((edge_labels, endpoints), dim=-1))

        empty_rows = np.asarray(counts) == 0
        element_padding_array[empty_rows, 0] = False
        element_padding_mask = torch.as_tensor(element_padding_array, dtype=torch.bool, device=device)
        has_elements = torch.as_tensor([count > 0 for count in counts], dtype=torch.bool, device=device)
        # MultiheadAttention cannot attend to an entirely masked row. Empty mapped
        # subgraphs get one harmless dummy key; its attention update is then zeroed.
        summary = self.embedding_dropout(self.interpretation_embedding(item_labels)).unsqueeze(1)
        for layer in self.local_layers:
            summary = layer(summary, elements, element_padding_mask, has_elements)
        return self.local_norm(summary[:, 0, :])

    def forward(self, batch: Sequence[Dict[str, Any]]) -> Tuple[Tensor, Tensor]:
        if not batch:
            empty = self.graph_token.new_empty((0, self.global_d_model))
            return empty, self.head(empty)

        graph_indices: List[int] = []
        graph_positions: List[int] = []
        interpretations: List[Dict[str, Any]] = []
        counts: List[int] = []
        for graph_index, graph in enumerate(batch):
            graph_items = graph["interpretations"]
            counts.append(len(graph_items))
            interpretations.extend(graph_items)
            graph_indices.extend([graph_index] * len(graph_items))
            graph_positions.extend(range(1, len(graph_items) + 1))

        local = self._encode_interpretations(interpretations)
        max_interpretations = max(counts, default=0)
        sequence = self.graph_token.new_zeros((len(batch), max_interpretations + 1, self.global_d_model))
        sequence[:, 0, :] = self.graph_token[0, 0, :]
        padding_mask_array = np.ones((len(batch), max_interpretations + 1), dtype=np.bool_)
        padding_mask_array[:, 0] = False
        for index, count in enumerate(counts):
            padding_mask_array[index, 1 : count + 1] = False
        if interpretations:
            graph_index_tensor = torch.as_tensor(graph_indices, dtype=torch.long, device=self.graph_token.device)
            graph_position_tensor = torch.as_tensor(graph_positions, dtype=torch.long, device=self.graph_token.device)
            sequence[graph_index_tensor, graph_position_tensor] = self.to_global(local)
        padding_mask = torch.as_tensor(padding_mask_array, dtype=torch.bool, device=self.graph_token.device)
        encoded = self.global_norm(self.global_transformer(sequence, src_key_padding_mask=padding_mask))
        graph_embedding = encoded[:, 0, :]
        return graph_embedding, self.head(graph_embedding)


class _GraphIsomorphismModel(nn.Module):
    """A relation-aware GIN for direct message passing over input graphs."""

    def __init__(
        self,
        *,
        nbits: int,
        hidden_dim: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_embedding = nn.Embedding(1 << nbits, hidden_dim, padding_idx=0)
        self.node_attribute_projection = nn.Linear(4, hidden_dim, bias=False)
        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dim * 5, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, hidden_dim),
            )
            for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden_dim * 2 * (num_layers + 1), hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def _collate(self, batch: Sequence[Dict[str, Any]]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        device = self.node_embedding.weight.device
        counts = [int(graph["node_labels"].size) for graph in batch]
        max_nodes = max(1, max(counts, default=0))
        node_labels = np.zeros((len(batch), max_nodes), dtype=np.int64)
        node_attributes = np.zeros((len(batch), max_nodes, 4), dtype=np.float32)
        edge_adjacencies = np.zeros((len(batch), max_nodes, max_nodes, 4), dtype=np.float32)
        node_mask = np.zeros((len(batch), max_nodes), dtype=np.bool_)
        for graph_index, graph in enumerate(batch):
            count = counts[graph_index]
            if count:
                node_labels[graph_index, :count] = graph["node_labels"]
                node_attributes[graph_index, :count] = graph["node_attributes"]
                node_mask[graph_index, :count] = True
            for source, target, relation in graph["edges"]:
                edge_adjacencies[graph_index, source, target, relation] = 1.0
                edge_adjacencies[graph_index, target, source, relation] = 1.0
        return (
            torch.as_tensor(node_labels, dtype=torch.long, device=device),
            torch.as_tensor(node_attributes, dtype=torch.float32, device=device),
            torch.as_tensor(edge_adjacencies, dtype=torch.float32, device=device),
            torch.as_tensor(node_mask, dtype=torch.bool, device=device),
        )

    def forward(self, batch: Sequence[Dict[str, Any]]) -> Tuple[Tensor, Tensor]:
        node_labels, node_attributes, edge_adjacencies, node_mask = self._collate(batch)
        mask = node_mask.unsqueeze(-1)
        hidden = self.node_embedding(node_labels) + self.node_attribute_projection(node_attributes)
        hidden = hidden * mask
        graph_representations = []

        def pool(values: Tensor) -> Tensor:
            total = values.sum(dim=1)
            count = node_mask.sum(dim=1, keepdim=True).clamp_min(1).to(values.dtype)
            return torch.cat((total, total / count), dim=-1)

        graph_representations.append(pool(hidden))
        for layer, norm in zip(self.layers, self.norms):
            neighbor_sums = torch.einsum("bijr,bjd->bird", edge_adjacencies, hidden)
            neighbor_sums = neighbor_sums.flatten(start_dim=2)
            update = layer(torch.cat((hidden, neighbor_sums), dim=-1))
            hidden = self.dropout(norm(hidden + update)) * mask
            graph_representations.append(pool(hidden))
        embedding = F.gelu(self.readout(torch.cat(graph_representations, dim=-1)))
        return embedding, self.head(embedding)


class NeuralGraphEstimator:
    """Train a hierarchical Transformer over AbstractGraph mapped subgraphs.

    ``decomposition_function`` receives the ``AbstractGraph`` created for each
    input graph and must return the decomposed ``AbstractGraph``. The selected
    ``element_mode`` controls which mapped-subgraph elements become local tokens.
    Set ``architecture="gin"`` to use direct relation-aware message passing on
    the input graph instead of the hierarchical decomposition encoder.
    If ``device`` is omitted, CUDA is preferred, then Apple MPS, then CPU.
    """

    def __init__(
        self,
        decomposition_function: Callable[[AbstractGraph], AbstractGraph],
        nbits: int = 10,
        element_mode: ElementMode = "node",
        architecture: Literal["hierarchical", "gin"] = "hierarchical",
        mode: TaskMode = "classification",
        num_classes: Optional[int] = None,
        output_dim: int = 1,
        local_d_model: int = 128,
        local_nhead: int = 4,
        local_num_layers: int = 2,
        local_dim_feedforward: int = 256,
        global_d_model: int = 128,
        global_nhead: int = 4,
        global_num_layers: int = 2,
        global_dim_feedforward: int = 256,
        dropout: float = 0.1,
        embedding_dropout: float = 0.0,
        epochs: int = 20,
        batch_size: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        lr_scheduler_patience: Optional[int] = None,
        lr_scheduler_factor: float = 0.5,
        lr_scheduler_min_lr: float = 1e-5,
        label_smoothing: float = 0.0,
        class_weight: Optional[Union[Literal["balanced"], Sequence[float]]] = None,
        val_split: float = 0.0,
        early_stopping_patience: Optional[int] = None,
        seed: Optional[int] = None,
        validation_seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = False,
    ) -> None:
        self.decomposition_function = decomposition_function
        self.nbits = int(nbits)
        self.element_mode = element_mode
        self.architecture = architecture
        self.mode = mode
        self._num_classes_param = None if num_classes is None else int(num_classes)
        self.num_classes = None if num_classes is None else int(num_classes)
        self.output_dim = int(output_dim)
        self.local_d_model = int(local_d_model)
        self.local_nhead = int(local_nhead)
        self.local_num_layers = int(local_num_layers)
        self.local_dim_feedforward = int(local_dim_feedforward)
        self.global_d_model = int(global_d_model)
        self.global_nhead = int(global_nhead)
        self.global_num_layers = int(global_num_layers)
        self.global_dim_feedforward = int(global_dim_feedforward)
        self.dropout = float(dropout)
        self.embedding_dropout = float(embedding_dropout)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = float(lr_scheduler_factor)
        self.lr_scheduler_min_lr = float(lr_scheduler_min_lr)
        self.label_smoothing = float(label_smoothing)
        self.class_weight = class_weight
        self.val_split = float(val_split)
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed
        self.validation_seed = validation_seed
        self.device = torch.device(device) if device is not None else _default_device()
        self.verbose = bool(verbose)
        self.model_: Optional[_HierarchicalGraphModel] = None
        self.classes_: Optional[np.ndarray] = None
        self.node_attribute_dim_: Optional[int] = None
        self.edge_attribute_dim_: Optional[int] = None
        self.history_: Dict[str, List[float]] = {}
        self.validation_targets_: Optional[np.ndarray] = None
        self.validation_proba_: Optional[np.ndarray] = None
        self._validate_params()

    def _validate_params(self) -> None:
        if self.element_mode not in ("node", "edge"):
            raise ValueError("element_mode must be 'node' or 'edge'.")
        if self.architecture not in ("hierarchical", "gin"):
            raise ValueError("architecture must be 'hierarchical' or 'gin'.")
        if self.mode not in ("classification", "regression"):
            raise ValueError("mode must be 'classification' or 'regression'.")
        if self.nbits < 2:
            raise ValueError("nbits must be at least 2.")
        for name, value in (
            ("local_d_model", self.local_d_model),
            ("local_nhead", self.local_nhead),
            ("global_d_model", self.global_d_model),
            ("global_nhead", self.global_nhead),
            ("batch_size", self.batch_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.local_d_model % self.local_nhead:
            raise ValueError("local_d_model must be divisible by local_nhead.")
        if self.global_d_model % self.global_nhead:
            raise ValueError("global_d_model must be divisible by global_nhead.")
        if self.local_num_layers < 0 or self.global_num_layers < 0:
            raise ValueError("Transformer layer counts cannot be negative.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.lr_scheduler_patience is not None and self.lr_scheduler_patience < 0:
            raise ValueError("lr_scheduler_patience must be non-negative or None.")
        if not 0.0 < self.lr_scheduler_factor < 1.0:
            raise ValueError("lr_scheduler_factor must be in (0, 1).")
        if self.lr_scheduler_min_lr <= 0.0:
            raise ValueError("lr_scheduler_min_lr must be positive.")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1).")
        if not 0.0 <= self.embedding_dropout < 1.0:
            raise ValueError("embedding_dropout must be in [0, 1).")
        if not 0.0 <= self.val_split < 1.0:
            raise ValueError("val_split must be in [0, 1).")
        if isinstance(self.class_weight, str) and self.class_weight != "balanced":
            raise ValueError("class_weight must be None, 'balanced', or a sequence of class weights.")
        if self.class_weight is not None and self.mode != "classification":
            raise ValueError("class_weight is only available in classification mode.")

    def _preprocess(self, graphs: Sequence[nx.Graph], *, fitting: bool = False) -> List[Dict[str, Any]]:
        if self.verbose and fitting:
            print(f"Preparing {len(graphs)} training graphs...", flush=True)
        prepared: List[Dict[str, Any]] = []
        progress_interval = max(1, math.ceil(len(graphs) / 10))
        for index, graph in enumerate(graphs, start=1):
            if self.architecture == "gin":
                prepared.append(_graph_to_gin_input(graph, self.nbits))
            else:
                prepared.append(_graph_to_input(graph, self.decomposition_function, self.nbits, self.element_mode))
            if self.verbose and fitting and (index % progress_interval == 0 or index == len(graphs)):
                print(f"Preprocessed {index}/{len(graphs)} graphs", flush=True)
        if self.architecture == "gin":
            self.node_attribute_dim_ = 4
            self.edge_attribute_dim_ = 0
            return prepared
        node_widths = [width for graph in prepared for width in graph["node_attribute_widths"]]
        edge_widths = [width for graph in prepared for width in graph["edge_attribute_widths"]]
        node_width = _infer_width(node_widths, "node")
        edge_width = _infer_width(edge_widths, "edge")
        if fitting:
            self.node_attribute_dim_ = node_width
            self.edge_attribute_dim_ = edge_width
        else:
            if self.node_attribute_dim_ is None or self.edge_attribute_dim_ is None:
                raise RuntimeError("Estimator has not been fitted. Call fit before inference.")
            if node_width not in (0, self.node_attribute_dim_):
                raise ValueError(
                    f"Node attribute width changed: expected {self.node_attribute_dim_}, received {node_width}."
                )
            if edge_width not in (0, self.edge_attribute_dim_):
                raise ValueError(
                    f"Edge attribute width changed: expected {self.edge_attribute_dim_}, received {edge_width}."
                )
        expected_node = int(self.node_attribute_dim_ or 0)
        expected_edge = int(self.edge_attribute_dim_ or 0)
        for graph in prepared:
            for item in graph["interpretations"]:
                for values in item["element_attributes"]:
                    expected = expected_node if self.element_mode == "node" else expected_edge
                    if values.size not in (0, expected):
                        kind = "Node" if self.element_mode == "node" else "Edge"
                        raise ValueError(f"{kind} attribute width changed: expected {expected}, received {values.size}.")
                if self.element_mode == "edge":
                    for pair in item["endpoint_attributes"]:
                        for values in pair:
                            if values.size not in (0, expected_node):
                                raise ValueError(
                                    f"Node attribute width changed: expected {expected_node}, received {values.size}."
                                )
        return prepared

    def _build_model(self) -> None:
        output_dim = self.num_classes if self.mode == "classification" else self.output_dim
        if output_dim is None or output_dim <= 0:
            raise ValueError("num_classes/output_dim must be positive.")
        if self.architecture == "gin":
            self.model_ = _GraphIsomorphismModel(
                nbits=self.nbits,
                hidden_dim=self.local_d_model,
                num_layers=self.local_num_layers,
                dim_feedforward=self.local_dim_feedforward,
                dropout=self.dropout,
                output_dim=int(output_dim),
            ).to(self.device)
            return
        model = _HierarchicalGraphModel(
            nbits=self.nbits,
            element_mode=self.element_mode,
            node_attribute_dim=int(self.node_attribute_dim_ or 0),
            edge_attribute_dim=int(self.edge_attribute_dim_ or 0),
            local_d_model=self.local_d_model,
            local_nhead=self.local_nhead,
            local_num_layers=self.local_num_layers,
            local_dim_feedforward=self.local_dim_feedforward,
            global_d_model=self.global_d_model,
            global_nhead=self.global_nhead,
            global_num_layers=self.global_num_layers,
            global_dim_feedforward=self.global_dim_feedforward,
            dropout=self.dropout,
            embedding_dropout=self.embedding_dropout,
            output_dim=int(output_dim),
        )
        self.model_ = model.to(self.device)

    def _ensure_fitted(self) -> None:
        if self.model_ is None:
            raise RuntimeError("Estimator has not been fitted. Call fit before inference.")

    def _encode_targets(self, targets: Sequence[Any]) -> Tensor:
        if self.mode == "classification":
            target_array = np.asarray(targets)
            classes = np.unique(target_array)
            if classes.size < 2:
                raise ValueError("Classification requires at least two classes in y.")
            if self.num_classes is not None and classes.size != self.num_classes:
                raise ValueError(f"Expected {self.num_classes} classes, found {classes.size}.")
            self.num_classes = int(classes.size)
            self.classes_ = classes
            lookup = {value: index for index, value in enumerate(classes.tolist())}
            try:
                indices = [lookup[value] for value in target_array.tolist()]
            except (KeyError, TypeError) as exc:
                raise ValueError("Classification targets must be consistently typed scalar labels.") from exc
            return torch.tensor(indices, dtype=torch.long, device=self.device)
        values = np.asarray(targets, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2 or values.shape[1] != self.output_dim:
            raise ValueError(f"Regression targets must have shape (n_samples, {self.output_dim}).")
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def fit(self, graphs: Sequence[nx.Graph], targets: Sequence[Any]) -> "NeuralGraphEstimator":
        if not graphs:
            raise ValueError("Cannot fit on an empty dataset.")
        if len(graphs) != len(targets):
            raise ValueError("graphs and targets must have the same length.")
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        examples = self._preprocess(graphs, fitting=True)
        y = self._encode_targets(targets)
        self._build_model()
        assert self.model_ is not None
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = None
        if self.lr_scheduler_patience is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience,
                min_lr=self.lr_scheduler_min_lr,
            )
        if self.mode == "classification":
            class_weights = None
            if isinstance(self.class_weight, str) and self.class_weight == "balanced":
                counts = torch.bincount(y, minlength=int(self.num_classes or 0)).to(torch.float32)
                class_weights = len(y) / (counts * int(self.num_classes or 1))
            elif self.class_weight is not None:
                class_weights_np = np.asarray(self.class_weight, dtype=np.float32)
                if class_weights_np.shape != (int(self.num_classes or 0),) or not np.all(np.isfinite(class_weights_np)):
                    raise ValueError("class_weight must contain one finite weight per class.")
                if np.any(class_weights_np <= 0):
                    raise ValueError("class weights must be positive.")
                class_weights = torch.as_tensor(class_weights_np, dtype=torch.float32)
            criterion: nn.Module = nn.CrossEntropyLoss(
                weight=None if class_weights is None else class_weights.to(self.device),
                label_smoothing=self.label_smoothing,
            )
        else:
            criterion = nn.MSELoss()
        n_samples = len(examples)
        indices = np.arange(n_samples)
        validation_count = int(n_samples * self.val_split)
        if validation_count >= n_samples:
            raise ValueError("val_split leaves no training examples.")
        if self.validation_seed is None:
            np.random.shuffle(indices)
        else:
            np.random.RandomState(self.validation_seed).shuffle(indices)
        if validation_count and self.mode == "classification":
            # Keep every class represented in validation so early stopping and
            # learning-rate scheduling are not driven by a skewed small split.
            random_state = np.random.RandomState(self.validation_seed) if self.validation_seed is not None else None
            validation_parts = []
            for class_index in range(int(self.num_classes or 0)):
                class_indices = np.flatnonzero(y.detach().cpu().numpy() == class_index)
                if random_state is None:
                    np.random.shuffle(class_indices)
                else:
                    random_state.shuffle(class_indices)
                count = max(1, int(round(len(class_indices) * self.val_split)))
                validation_parts.append(class_indices[:count])
            validation_indices = np.concatenate(validation_parts).astype(int).tolist()
            validation_set = set(validation_indices)
            training_indices = [int(index) for index in indices if int(index) not in validation_set]
        else:
            validation_indices = indices[:validation_count].tolist()
            training_indices = indices[validation_count:].tolist()
        if not training_indices:
            raise ValueError("Validation split leaves no training examples.")
        self.history_ = {"loss": [], "val_loss": [], "learning_rate": []}
        if self.mode == "classification":
            self.history_["accuracy"] = []
            self.history_["val_accuracy"] = []
        if self.verbose:
            print(
                f"Training {len(training_indices)} graphs for up to {self.epochs} epochs "
                f"({len(validation_indices)} validation graphs, batch_size={self.batch_size}, device={self.device})",
                flush=True,
            )
        best_state: Optional[Dict[str, Tensor]] = None
        best_loss = math.inf
        best_validation_accuracy = -math.inf
        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            self.model_.train()
            np.random.shuffle(training_indices)
            losses: List[Tensor] = []
            train_correct = torch.zeros((), dtype=torch.long, device=self.device)
            train_count = 0
            batch_starts = list(range(0, len(training_indices), self.batch_size))
            progress_interval = max(1, math.ceil(len(batch_starts) / 5))
            for batch_number, start in enumerate(batch_starts, start=1):
                batch_indices = training_indices[start : start + self.batch_size]
                optimizer.zero_grad(set_to_none=True)
                _, prediction = self.model_([examples[index] for index in batch_indices])
                batch_targets = y[batch_indices]
                loss = criterion(prediction, batch_targets)
                loss.backward()
                optimizer.step()
                losses.append(loss.detach())
                if self.mode == "classification":
                    train_correct = train_correct + (prediction.detach().argmax(dim=1) == batch_targets).sum()
                    train_count += len(batch_indices)
                if self.verbose and (batch_number % progress_interval == 0 or batch_number == len(batch_starts)):
                    progress = f"epoch {epoch + 1}/{self.epochs}, batch {batch_number}/{len(batch_starts)}"
                    if self.mode == "classification":
                        progress += f", train_accuracy={train_correct.item() / train_count:.3f}"
                    print(progress, flush=True)
            train_loss = float(torch.stack(losses).mean().cpu()) if losses else math.nan
            train_accuracy = train_correct.item() / train_count if train_count else None
            self.history_["loss"].append(train_loss)
            if train_accuracy is not None:
                self.history_["accuracy"].append(train_accuracy)
            validation_loss: Optional[float] = None
            validation_accuracy: Optional[float] = None
            if validation_indices:
                self.model_.eval()
                with torch.no_grad():
                    val_losses: List[Tensor] = []
                    val_correct = torch.zeros((), dtype=torch.long, device=self.device)
                    for start in range(0, len(validation_indices), self.batch_size):
                        batch_indices = validation_indices[start : start + self.batch_size]
                        _, prediction = self.model_([examples[index] for index in batch_indices])
                        batch_targets = y[batch_indices]
                        val_losses.append(criterion(prediction, batch_targets).detach())
                        if self.mode == "classification":
                            val_correct = val_correct + (prediction.argmax(dim=1) == batch_targets).sum()
                    validation_loss = float(torch.stack(val_losses).mean().cpu())
                    if self.mode == "classification":
                        validation_accuracy = val_correct.item() / len(validation_indices)
                        self.history_["val_accuracy"].append(validation_accuracy)
                    self.history_["val_loss"].append(validation_loss)
                    if scheduler is not None:
                        scheduler.step(validation_loss)
                if self.mode == "classification" and validation_accuracy is not None:
                    improved = validation_accuracy > best_validation_accuracy or (
                        validation_accuracy == best_validation_accuracy and validation_loss < best_loss
                    )
                else:
                    improved = validation_loss < best_loss
                if improved:
                    best_loss = validation_loss
                    if validation_accuracy is not None:
                        best_validation_accuracy = validation_accuracy
                    best_state = copy.deepcopy(self.model_.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
            if self.verbose:
                suffix = "" if validation_loss is None else f", val_loss={validation_loss:.4f}"
                if train_accuracy is not None:
                    suffix += f", train_accuracy={train_accuracy:.3f}"
                if validation_accuracy is not None:
                    suffix += f", val_accuracy={validation_accuracy:.3f}"
                current_lr = float(optimizer.param_groups[0]["lr"])
                self.history_["learning_rate"].append(current_lr)
                suffix += f", lr={current_lr:.2g}"
                print(f"epoch {epoch + 1}/{self.epochs}: loss={train_loss:.4f}{suffix}", flush=True)
            else:
                self.history_["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
            if (
                validation_indices
                and self.early_stopping_patience is not None
                and epochs_without_improvement >= self.early_stopping_patience
            ):
                if self.verbose:
                    print(f"Early stopping after epoch {epoch + 1}.", flush=True)
                break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        if validation_indices and self.mode == "classification":
            self.model_.eval()
            validation_probabilities = []
            with torch.no_grad():
                for start in range(0, len(validation_indices), self.batch_size):
                    batch_indices = validation_indices[start : start + self.batch_size]
                    _, prediction = self.model_([examples[index] for index in batch_indices])
                    validation_probabilities.append(F.softmax(prediction, dim=1).cpu().numpy())
            self.validation_targets_ = np.asarray(targets)[validation_indices]
            self.validation_proba_ = np.concatenate(validation_probabilities, axis=0)
        return self

    @torch.no_grad()
    def _predict_prepared(self, examples: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_fitted()
        assert self.model_ is not None
        self.model_.eval()
        embeddings: List[np.ndarray] = []
        predictions: List[np.ndarray] = []
        for start in range(0, len(examples), self.batch_size):
            embedding, prediction = self.model_(examples[start : start + self.batch_size])
            embeddings.append(embedding.cpu().numpy())
            predictions.append(prediction.cpu().numpy())
        embedding_dim = self.local_d_model if self.architecture == "gin" else self.global_d_model
        emb = np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, embedding_dim), np.float32)
        pred = np.concatenate(predictions, axis=0) if predictions else np.empty((0, 0), np.float32)
        return emb, pred

    def transform(self, graphs: Sequence[nx.Graph]) -> np.ndarray:
        """Return graph embeddings with shape ``(n_graphs, embedding_dim)``."""
        self._ensure_fitted()
        embedding_dim = self.local_d_model if self.architecture == "gin" else self.global_d_model
        if not graphs:
            return np.empty((0, embedding_dim), dtype=np.float32)
        examples = self._preprocess(graphs)
        embeddings, _ = self._predict_prepared(examples)
        return embeddings

    def predict_proba(self, graphs: Sequence[nx.Graph]) -> np.ndarray:
        if self.mode != "classification":
            raise AttributeError("predict_proba is only available in classification mode.")
        self._ensure_fitted()
        if not graphs:
            return np.empty((0, int(self.num_classes or 0)), dtype=np.float32)
        examples = self._preprocess(graphs)
        _, logits = self._predict_prepared(examples)
        return F.softmax(torch.as_tensor(logits), dim=-1).numpy()

    def predict(self, graphs: Sequence[nx.Graph]) -> np.ndarray:
        self._ensure_fitted()
        if not graphs:
            if self.mode == "classification":
                assert self.classes_ is not None
                return np.empty((0,), dtype=self.classes_.dtype)
            return np.empty((0, self.output_dim), dtype=np.float32)
        examples = self._preprocess(graphs)
        _, prediction = self._predict_prepared(examples)
        if self.mode == "classification":
            assert self.classes_ is not None
            return self.classes_[prediction.argmax(axis=1)]
        return prediction

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        del deep
        return {
            "decomposition_function": self.decomposition_function,
            "nbits": self.nbits,
            "element_mode": self.element_mode,
            "architecture": self.architecture,
            "mode": self.mode,
            "num_classes": self._num_classes_param,
            "output_dim": self.output_dim,
            "local_d_model": self.local_d_model,
            "local_nhead": self.local_nhead,
            "local_num_layers": self.local_num_layers,
            "local_dim_feedforward": self.local_dim_feedforward,
            "global_d_model": self.global_d_model,
            "global_nhead": self.global_nhead,
            "global_num_layers": self.global_num_layers,
            "global_dim_feedforward": self.global_dim_feedforward,
            "dropout": self.dropout,
            "embedding_dropout": self.embedding_dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "lr_scheduler_patience": self.lr_scheduler_patience,
            "lr_scheduler_factor": self.lr_scheduler_factor,
            "lr_scheduler_min_lr": self.lr_scheduler_min_lr,
            "label_smoothing": self.label_smoothing,
            "class_weight": self.class_weight,
            "val_split": self.val_split,
            "early_stopping_patience": self.early_stopping_patience,
            "seed": self.seed,
            "validation_seed": self.validation_seed,
            "device": self.device,
            "verbose": self.verbose,
        }

    def set_params(self, **params: Any) -> "NeuralGraphEstimator":
        valid = self.get_params()
        unknown = set(params) - set(valid)
        if unknown:
            raise ValueError(f"Invalid parameter(s): {sorted(unknown)}")
        for name, value in params.items():
            setattr(self, name, value)
            if name == "num_classes":
                self._num_classes_param = None if value is None else int(value)
        self._validate_params()
        self.reset()
        return self

    def reset(self) -> "NeuralGraphEstimator":
        """Clear learned model and data dimensions while retaining configuration."""
        self.model_ = None
        self.classes_ = None
        self.num_classes = self._num_classes_param
        self.node_attribute_dim_ = None
        self.edge_attribute_dim_ = None
        self.history_ = {}
        return self
