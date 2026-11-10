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
                node_attribute = _numeric_attribute(data.get("attribute"), context="Node")
                node_attribute_widths.append(node_attribute.size)

            if element_mode == "node":
                for _, data in mapped.nodes(data=True):
                    element_labels.append(_label_id(data.get("label"), nbits))
                    attribute = _numeric_attribute(data.get("attribute"), context="Node")
                    element_attributes.append(attribute)
            else:
                for source, target, data in mapped.edges(data=True):
                    element_labels.append(_label_id(data.get("label"), nbits))
                    attribute = _numeric_attribute(data.get("attribute"), context="Edge")
                    edge_attribute_widths.append(attribute.size)
                    element_attributes.append(attribute)
                    source_data = mapped.nodes[source]
                    target_data = mapped.nodes[target]
                    endpoint_labels.append(
                        (_label_id(source_data.get("label"), nbits), _label_id(target_data.get("label"), nbits))
                    )
                    source_attribute = _numeric_attribute(source_data.get("attribute"), context="Node")
                    target_attribute = _numeric_attribute(target_data.get("attribute"), context="Node")
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

    def forward(self, summary: Tensor, elements: Tensor) -> Tensor:
        # The query can also retain its own state. Element tokens are never queries
        # and are deliberately not updated or given positional encodings.
        query = self.query_norm(summary)
        memory = self.memory_norm(elements)
        update, _ = self.attention(query, memory, memory, need_weights=False)
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
        output_dim: int,
    ) -> None:
        super().__init__()
        vocab_size = 1 << nbits
        self.element_mode = element_mode
        self.local_d_model = local_d_model
        self.global_d_model = global_d_model
        self.node_attribute_dim = node_attribute_dim
        self.edge_attribute_dim = edge_attribute_dim

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
        return self.node_projection(values)

    def _encode_interpretation(self, item: Dict[str, Any]) -> Tensor:
        device = self.graph_token.device
        summary = self.interpretation_embedding(torch.tensor([item["label"]], device=device)).unsqueeze(0)
        element_labels = torch.as_tensor(item["element_labels"], dtype=torch.long, device=device)
        count = int(element_labels.numel())
        if self.element_mode == "node":
            attributes = _attribute_tensor(
                item["element_attributes"], self.node_attribute_dim, context="Node"
            ).to(device)
            elements = self._node_features(element_labels, attributes).unsqueeze(0)
        else:
            edge_attributes = _attribute_tensor(
                item["element_attributes"], self.edge_attribute_dim, context="Edge"
            ).to(device)
            endpoint_labels = torch.as_tensor(item["endpoint_labels"], dtype=torch.long, device=device)
            endpoint_attributes = item["endpoint_attributes"]
            endpoint_node_features: List[Tensor] = []
            for endpoint in range(2):
                attrs = [pair[endpoint] for pair in endpoint_attributes]
                attrs_tensor = _attribute_tensor(attrs, self.node_attribute_dim, context="Node").to(device)
                endpoint_node_features.append(self._node_features(endpoint_labels[:, endpoint], attrs_tensor))
            if count:
                # Averaging makes undirected edge tokens independent of endpoint order.
                endpoints = (endpoint_node_features[0] + endpoint_node_features[1]) / 2.0
                edge_labels = self.element_label_embedding(element_labels)
                if self.edge_attribute_projection is not None:
                    edge_labels = edge_labels + self.edge_attribute_projection(edge_attributes)
                edge_vector = self.edge_projection(torch.cat((edge_labels, endpoints), dim=-1))
                elements = edge_vector.unsqueeze(0)
            else:
                elements = self.graph_token.new_empty((1, 0, self.local_d_model))

        for layer in self.local_layers:
            if count:
                summary = layer(summary, elements)
            else:
                # An empty mapped subgraph is represented by its hash token alone.
                summary = summary + layer.ff(layer.ff_norm(summary))
        return self.local_norm(summary.squeeze(0).squeeze(0))

    def forward(self, batch: Sequence[Dict[str, Any]]) -> Tuple[Tensor, Tensor]:
        graph_embeddings: List[Tensor] = []
        for graph in batch:
            interpretations = graph["interpretations"]
            local = [self._encode_interpretation(item) for item in interpretations]
            tokens = [self.graph_token.squeeze(0).squeeze(0)]
            tokens.extend(self.to_global(value) for value in local)
            sequence = torch.stack(tokens, dim=0).unsqueeze(0)
            encoded = self.global_norm(self.global_transformer(sequence))
            graph_embeddings.append(encoded[0, 0])
        graph_embedding = torch.stack(graph_embeddings, dim=0)
        return graph_embedding, self.head(graph_embedding)


class NeuralGraphEstimator:
    """Train a hierarchical Transformer over AbstractGraph mapped subgraphs.

    ``decomposition_function`` receives the ``AbstractGraph`` created for each
    input graph and must return the decomposed ``AbstractGraph``. The selected
    ``element_mode`` controls which mapped-subgraph elements become local tokens.
    """

    def __init__(
        self,
        decomposition_function: Callable[[AbstractGraph], AbstractGraph],
        nbits: int = 10,
        element_mode: ElementMode = "node",
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
        epochs: int = 20,
        batch_size: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        val_split: float = 0.0,
        early_stopping_patience: Optional[int] = None,
        seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = False,
    ) -> None:
        self.decomposition_function = decomposition_function
        self.nbits = int(nbits)
        self.element_mode = element_mode
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
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.val_split = float(val_split)
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.verbose = bool(verbose)
        self.model_: Optional[_HierarchicalGraphModel] = None
        self.classes_: Optional[np.ndarray] = None
        self.node_attribute_dim_: Optional[int] = None
        self.edge_attribute_dim_: Optional[int] = None
        self.history_: Dict[str, List[float]] = {}
        self._validate_params()

    def _validate_params(self) -> None:
        if self.element_mode not in ("node", "edge"):
            raise ValueError("element_mode must be 'node' or 'edge'.")
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
        if not 0.0 <= self.val_split < 1.0:
            raise ValueError("val_split must be in [0, 1).")

    def _preprocess(self, graphs: Sequence[nx.Graph], *, fitting: bool = False) -> List[Dict[str, Any]]:
        prepared = [
            _graph_to_input(graph, self.decomposition_function, self.nbits, self.element_mode)
            for graph in graphs
        ]
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
        criterion: nn.Module = nn.CrossEntropyLoss() if self.mode == "classification" else nn.MSELoss()
        n_samples = len(examples)
        indices = np.arange(n_samples)
        validation_count = int(n_samples * self.val_split)
        if validation_count >= n_samples:
            raise ValueError("val_split leaves no training examples.")
        np.random.shuffle(indices)
        validation_indices = indices[:validation_count].tolist()
        training_indices = indices[validation_count:].tolist()
        if not training_indices:
            raise ValueError("Validation split leaves no training examples.")
        self.history_ = {"loss": [], "val_loss": []}
        best_state: Optional[Dict[str, Tensor]] = None
        best_loss = math.inf
        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            self.model_.train()
            np.random.shuffle(training_indices)
            losses: List[float] = []
            for start in range(0, len(training_indices), self.batch_size):
                batch_indices = training_indices[start : start + self.batch_size]
                optimizer.zero_grad(set_to_none=True)
                _, prediction = self.model_([examples[index] for index in batch_indices])
                loss = criterion(prediction, y[batch_indices])
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            train_loss = float(np.mean(losses)) if losses else math.nan
            self.history_["loss"].append(train_loss)
            validation_loss: Optional[float] = None
            if validation_indices:
                self.model_.eval()
                with torch.no_grad():
                    val_losses = []
                    for start in range(0, len(validation_indices), self.batch_size):
                        batch_indices = validation_indices[start : start + self.batch_size]
                        _, prediction = self.model_([examples[index] for index in batch_indices])
                        val_losses.append(float(criterion(prediction, y[batch_indices]).cpu()))
                    validation_loss = float(np.mean(val_losses))
                    self.history_["val_loss"].append(validation_loss)
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    best_state = copy.deepcopy(self.model_.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                if self.early_stopping_patience is not None and epochs_without_improvement >= self.early_stopping_patience:
                    break
            if self.verbose:
                suffix = "" if validation_loss is None else f", val_loss={validation_loss:.4f}"
                print(f"epoch {epoch + 1}/{self.epochs}: loss={train_loss:.4f}{suffix}")
        if best_state is not None:
            self.model_.load_state_dict(best_state)
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
        emb = np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, self.global_d_model), np.float32)
        pred = np.concatenate(predictions, axis=0) if predictions else np.empty((0, 0), np.float32)
        return emb, pred

    def transform(self, graphs: Sequence[nx.Graph]) -> np.ndarray:
        """Return graph-token embeddings with shape ``(n_graphs, global_d_model)``."""
        self._ensure_fitted()
        if not graphs:
            return np.empty((0, self.global_d_model), dtype=np.float32)
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
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "val_split": self.val_split,
            "early_stopping_patience": self.early_stopping_patience,
            "seed": self.seed,
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
