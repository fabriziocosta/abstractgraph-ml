"""
Neural graph estimators and Transformer-based models for AbstractGraph.

Overview
- Scikit-style estimator that turns AbstractGraph node features into graph-level
  predictions using a configurable Transformer encoder and a small head.
- Clean separation between graph preprocessing (decomposition + hashing) and
  neural modeling (adapters + encoder + head).

AbstractGraph to tensors
- An AbstractGraph pairs a preimage graph with an image graph whose nodes carry
  labels and associations (subgraphs of the preimage). Decomposition operators
  build the image graph; hashing converts labels into a fixed feature space.
- The node vectorizer (`AbstractGraphNodeTransformer`) produces one dense matrix
  per graph with shape [n_nodes, n_features]. By convention:
  - Column 0 is a bias/“node exists” indicator (pooling yields node count).
  - Column 1 corresponds to degree-related count when pooled.
  - Remaining columns are hashed label features (2**nbits buckets).

Input adapters
- High-dimensional hashed features are projected to `d_model` via an input adapter
  to stabilize and reduce compute before attention:
  - "linear": single `nn.Linear(in_dim -> d_model)`.
  - "factorized": two linear maps with a low-rank bottleneck
    `in_dim -> bottleneck -> d_model` (keeps the mapping linear). Optional SVD
    initialization approximates a random Gaussian map with rank=bottleneck.
- Optional auxiliary losses can regularize adapters:
  - Whitening on the factorized first projection.
  - L1 sparsity on the second projection.

Transformer encoder and pooling
- `nn.TransformerEncoder` with `norm_first=True`, `batch_first=True`, configurable
  `nhead`, `num_layers`, `dim_feedforward`, `dropout`, and activation.
- Pooling options: "mean", "max", or "cls". With "cls", a learnable token is
  prepended and used as the graph representation.
- A final layer norm produces graph embeddings for the prediction head.

Prediction heads
- Classification: `GraphTransformerClassifier(encoder, num_classes)` uses an
  `nn.Linear(d_model, num_classes)` head.
- Regression: `GraphTransformerRegressor(encoder, out_dim)` uses an
  `nn.Linear(d_model, out_dim)` head.

Training APIs
- `fit(graphs, targets)`: supervised learning for classification or regression.
  Trains encoder + head jointly with AdamW, optional validation split, and early
  stopping on "val_loss" or task-specific metric ("val_acc" / "val_mse").
- `pre_train(graphs, decomposition_function, nbits)`: unsupervised pre-training
  of the encoder using a motif-level prototype InfoNCE objective over image-node
  associations (motifs). Only the encoder parameters are optimized; the head is
  not used and is reported as 0 trainables.
  Decoupled decomposition: The `decomposition_function` here is used solely to
  build an AbstractGraph per input graph in order to extract motif assignments
  (image-node labels and the indices of their associated preimage nodes). These
  assignments define the contrastive task (prototype per motif label). The node
  features fed to the encoder still come from the estimator's configured
  `node_vectorizer`; they do not depend on this pre-train decomposition. This
  allows you to pre-train with a decomposition tailored for robust motif signals
  (e.g., larger neighborhoods, different label hashing `nbits`) while keeping a
  different decomposition or feature recipe in the `node_vectorizer` for the
  supervised phases. In short: pre-train decomposition shapes the pretext task;
  the node vectorizer governs the encoder inputs.
- `fine_tune(graphs, targets, ...)`: parameter-efficient supervised adaptation
  via LoRA. Freezes base weights, injects trainable low-rank adapters into
  selected Linear layers, and optimizes only LoRA (and, if no LoRA targets,
  unfreezes the head as a fallback).

LoRA adapters
- Each targeted `nn.Linear` is wrapped in `LoRAInjectedLinear`:
  `y = base(x) + (alpha / r) * B(A(Dropout(x)))`, where A∈R^{r×in}, B∈R^{out×r}.
  The base layer’s weights and bias remain frozen; only A/B are trained.
- Scope controls which linears receive LoRA:
  - "encoder_only": all linear submodules under `encoder.*`.
  - "head_only": only the prediction head.
  - "ffn_only": only encoder MLP layers (`encoder.layers.*.(linear1|linear2)`).
  - "adapter_only": only the input adapter linears
    (`encoder.input_adapter.*`).
  - "all_linear": every `nn.Linear` in the model (encoder, adapters, head).
- Additional knobs: rank `lora_r`, scaling `lora_alpha` (effective scale is
  `alpha / r`), and `lora_dropout` on the LoRA branch.

Hot/cold start and resetting
- The estimator caches a compiled model after the first call; subsequent calls to
  `fit`, `pre_train`, or `fine_tune` continue from current weights (hot start).
- `set_params(...)` invalidates the compiled model to rebuild on next train.
- `reset(reseed: Optional[int])` explicitly drops the compiled model and LoRA
  state while preserving hyperparameters, enabling a clean re-initialization.

Logging and histories
- Verbose mode prints parameter counts and epoch progress lines for supervised
  training and pre-training (with consistent "epoch i/N" headers). Histories are
  stored per phase (`history_`, `pretrain_history_`, `finetune_history_`).

Typical workflows
- From scratch with labels: `fit(...)`.
- Semi-supervised: `pre_train(...)` → `fit(...)`.
- Parameter-efficient adaptation: `pre_train(...)` → `fine_tune(lora_scope="encoder_only")`.
- No full-head training: `pre_train(...)` → `fine_tune(lora_scope="all_linear")` so the
  head adapts via LoRA without changing base weights.
"""

import math
import time
from datetime import timedelta
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the provided node-level vectorizer to obtain per-node embeddings
from abstractgraph.graphs import AbstractGraph, graphs_to_abstract_graphs
from abstractgraph.vectorize import AbstractGraphNodeTransformer


Tensor = torch.Tensor


class LoRAInjectedLinear(nn.Module):
    """
    Lightweight LoRA wrapper for a Linear layer.

    Applies a low-rank adaptation in parallel to a frozen base Linear:
        y = base(x) + scale * (B(A(dropout(x))))

    Where A in R^{r x in}, B in R^{out x r}, and scale = alpha / r.

    Bias handling: the base layer's bias (if any) is used as-is and kept frozen.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRAInjectedLinear expects an nn.Linear as base")
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")
        self.base = base
        # Freeze base parameters
        for p in self.base.parameters():
            p.requires_grad = False

        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(r)
        self.lora_dropout = nn.Dropout(p=float(dropout)) if dropout and dropout > 0.0 else nn.Identity()

        # LoRA parameters (initialize A with small random, B to zeros so delta starts at 0)
        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(self.r, self.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r, device=device, dtype=dtype))
        # Kaiming uniform for A keeps scale reasonable; B zero -> no change at init
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def _effective_weight(self) -> Tensor:
        # Compute dynamic effective weight: W_eff = W_base + scale * (B @ A)
        # Shapes: base.W[out,in], A[r,in], B[out,r]
        return self.base.weight + self.scaling * (self.lora_B @ self.lora_A)

    @property
    def weight(self) -> Tensor:  # for modules that access .weight directly (e.g., MHA)
        return self._effective_weight()

    @property
    def bias(self) -> Optional[Tensor]:  # proxy base bias
        return self.base.bias

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base(x)
        x_d = self.lora_dropout(x)
        # [*, in] -> [*, r] via A^T, then -> [*, out] via B^T
        delta = F.linear(x_d, self.lora_A)  # weight [r, in]
        delta = F.linear(delta, self.lora_B)  # weight [out, r]
        return base_out + self.scaling * delta


class InputAdapterLinear(nn.Module):
    """
    Simple linear projection from `in_dim` to `d_model`.

    Args:
        in_dim: Input feature dimension.
        d_model: Output feature dimension.
        bias: Whether to include a bias term.
    """

    def __init__(self, in_dim: int, d_model: int, bias: bool = True):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(d_model)
        self.proj = nn.Linear(in_dim, d_model, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, N, in_dim]
        return self.proj(x)  # -> [B, N, d_model]


class InputAdapterFactorized(nn.Module):
    """
    Factorized low-rank projection: `in_dim -> bottleneck -> d_model`.

    Reduces parameter count/compute for high-dimensional inputs by using a
    bottleneck of rank `bottleneck` (b << min(in_dim, d_model)). No nonlinearity
    is applied to keep the mapping linear.

    Args:
        in_dim: Input feature dimension.
        d_model: Output feature dimension.
        bottleneck: Rank of the bottleneck projection.
        first_bias: Whether the first projection includes bias.
        second_bias: Whether the second projection includes bias.
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int,
        bottleneck: int,
        first_bias: bool = False,
        second_bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(d_model)
        if bottleneck <= 0 or bottleneck >= max(in_dim, d_model):
            raise ValueError("bottleneck must be >0 and less than max(in_dim, d_model)")
        self.proj1 = nn.Linear(in_dim, bottleneck, bias=first_bias)
        self.proj2 = nn.Linear(bottleneck, d_model, bias=second_bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj1(x)
        x = self.proj2(x)
        return x

    @torch.no_grad()
    def svd_init(self, in_dim: int, d_model: int, bottleneck: int) -> None:
        """
        Initialize proj1/proj2 so that proj2(proj1(x)) approximates a random
        Gaussian linear map of shape [in_dim, d_model] with rank `bottleneck`.

        Sets:
          W_full ≈ U[:, :b] @ diag(S[:b]) @ Vh[:b, :]
          proj1.weight = (U[:, :b] @ diag(sqrt(S[:b])))^T
          proj2.weight = (diag(sqrt(S[:b])) @ Vh[:b, :])^T
        Biases are left at their existing initialization.

        Args:
            in_dim: Input feature dimension.
            d_model: Output feature dimension.
            bottleneck: Rank used for the factorized approximation.

        Returns:
            None
        """
        device = self.proj1.weight.device
        dtype = self.proj1.weight.dtype
        b = int(bottleneck)
        # Random target matrix
        W = torch.randn(in_dim, d_model, device=device, dtype=dtype)
        # Compute truncated SVD via full then slice; shapes: U[in, r], S[r], Vh[r, d]
        # torch.linalg.svd returns U[in,in], S[min(in,d)], Vh[d,d] when full_matrices=True
        try:
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        except RuntimeError:
            # Fallback to CPU if device lacks SVD kernel
            W_cpu = W.cpu()
            Uc, Sc, Vhc = torch.linalg.svd(W_cpu, full_matrices=False)
            U, S, Vh = Uc.to(device=device, dtype=dtype), Sc.to(device=device, dtype=dtype), Vhc.to(device=device, dtype=dtype)
        Ub = U[:, :b]
        Sb = S[:b]
        Vhb = Vh[:b, :]
        rootS = torch.sqrt(Sb).unsqueeze(0)  # [1,b]
        # Compute factors
        W1 = Ub * rootS  # [in_dim, b]
        W2 = (rootS.T) * Vhb  # [b, d_model]
        # Assign to Linear weights (note Linear uses weight shape [out, in])
        with torch.no_grad():
            self.proj1.weight.copy_(W1.T)
            self.proj2.weight.copy_(W2.T)


ADAPTER_REGISTRY: Dict[str, type] = {}


def register_adapter(name: str, adapter_cls: type) -> None:
    """
    Register a node input adapter for GraphTransformerEncoder.

    Args:
        name: Registry key.
        adapter_cls: nn.Module class implementing the adapter.

    Returns:
        None
    """
    if not issubclass(adapter_cls, nn.Module):
        raise TypeError("adapter_cls must be a subclass of nn.Module")
    ADAPTER_REGISTRY[name] = adapter_cls


def build_adapter(name: str, **kwargs) -> nn.Module:
    """
    Construct a registered adapter by name.

    Args:
        name: Registry key.
        **kwargs: Adapter constructor arguments.

    Returns:
        Instantiated adapter module.
    """
    if name not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown adapter_type '{name}'. Available: {sorted(ADAPTER_REGISTRY.keys())}")
    return ADAPTER_REGISTRY[name](**kwargs)


register_adapter("linear", InputAdapterLinear)
register_adapter("factorized", InputAdapterFactorized)


class GraphTransformerEncoder(nn.Module):
    """
    A Transformer encoder operating on sets/sequences of node embeddings.

    - Optional learnable [CLS] token for graph-level representation.
    - Supports attention masks for padded nodes.
    - Aggregates to graph embedding via `pooling` if CLS is disabled.

    Args:
        in_dim: Input node feature dimension.
        d_model: Transformer embedding dimension.
        nhead: Number of attention heads.
        num_layers: Number of encoder layers.
        dim_feedforward: Feed-forward hidden dimension.
        dropout: Dropout probability in encoder layers.
        activation: Activation function ("relu" or "gelu").
        pooling: Graph pooling strategy ("mean", "max", "cls").
        adapter_type: Input adapter type ("linear" or "factorized").
        adapter: Optional custom adapter module; overrides adapter_type.
        adapter_bottleneck: Bottleneck rank for factorized adapter.
        adapter_bias: Whether the adapter includes bias terms.
        adapter_init: Adapter initialization ("svd" or "default").
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: Literal["relu", "gelu"] = "relu",
        pooling: Literal["mean", "max", "cls"] = "mean",
        # Input adapter configuration
        adapter_type: Literal["linear", "factorized"] = "linear",
        adapter: Optional[nn.Module] = None,
        adapter_bottleneck: Optional[int] = None,
        adapter_bias: bool = True,
        adapter_init: Optional[Literal["svd", "default"]] = None,
    ):
        super().__init__()
        self.pooling = pooling
        self.use_cls_token = (pooling == "cls")

        # Build input adapter
        if adapter is not None:
            if not isinstance(adapter, nn.Module):
                raise TypeError("adapter must be an nn.Module")
            if hasattr(adapter, "out_dim") and int(getattr(adapter, "out_dim")) != int(d_model):
                raise ValueError("adapter.out_dim must match d_model")
            self.input_adapter = adapter
        else:
            if adapter_type == "linear":
                self.input_adapter = build_adapter("linear", in_dim=in_dim, d_model=d_model, bias=adapter_bias)
            elif adapter_type == "factorized":
                if adapter_bottleneck is None:
                    raise ValueError("adapter_bottleneck must be provided for adapter_type='factorized'")
                self.input_adapter = build_adapter(
                    "factorized",
                    in_dim=in_dim,
                    d_model=d_model,
                    bottleneck=int(adapter_bottleneck),
                    first_bias=False,
                    second_bias=adapter_bias,
                )
                # Default to SVD init when factorized unless explicitly disabled
                init_mode = adapter_init or "svd"
                if init_mode == "svd":
                    try:
                        self.input_adapter.svd_init(in_dim, d_model, int(adapter_bottleneck))
                    except Exception:
                        # Leave default initialization on failure
                        pass
            else:
                raise ValueError("adapter_type must be 'linear' or 'factorized'")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.norm = nn.LayerNorm(d_model)

    def _prepend_cls(self, x: Tensor, key_padding_mask: Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
        # x: [B, N, D]; mask: [B, N], where True indicates padding
        B = x.size(0)
        cls_tok = self.cls_token.expand(B, -1, -1)  # [B,1,D]
        x = torch.cat([cls_tok, x], dim=1)  # [B, N+1, D]
        if key_padding_mask is not None:
            pad = torch.zeros((B, 1), dtype=key_padding_mask.dtype, device=key_padding_mask.device)
            key_padding_mask = torch.cat([pad, key_padding_mask], dim=1)
        return x, key_padding_mask

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: [B, N, in_dim] raw node embeddings.
            key_padding_mask: [B, N] with True for padding positions.

        Returns:
            node_embeddings: [B, N(+1), D] encoded node embeddings (includes CLS if enabled).
            graph_embeddings: [B, D] graph-level pooled embeddings.
        """
        x = self.input_adapter(x)  # [B, N, D]

        if self.use_cls_token:
            x, key_padding_mask = self._prepend_cls(x, key_padding_mask)

        # Transformer expects key_padding_mask where True denotes positions that are NOT valid
        out = self.encoder(x, src_key_padding_mask=key_padding_mask)
        out = self.norm(out)

        # Graph-level embedding
        if self.use_cls_token:
            graph_emb = out[:, 0, :]  # [CLS]
            node_emb = out  # includes CLS at index 0
        else:
            node_emb = out
            if self.pooling == "mean":
                if key_padding_mask is None:
                    graph_emb = node_emb.mean(dim=1)
                else:
                    # mask True=padded; invert to get valid positions
                    valid = (~key_padding_mask).float().unsqueeze(-1)  # [B,N,1]
                    summed = (node_emb * valid).sum(dim=1)
                    counts = valid.sum(dim=1).clamp_min(1.0)
                    graph_emb = summed / counts
            elif self.pooling == "max":
                if key_padding_mask is None:
                    graph_emb, _ = node_emb.max(dim=1)
                else:
                    # Set padded positions to very low
                    masked = node_emb.masked_fill(key_padding_mask.unsqueeze(-1), float("-inf"))
                    graph_emb, _ = masked.max(dim=1)
            else:
                # Fallback to mean
                graph_emb = node_emb.mean(dim=1)

        return node_emb, graph_emb


class GraphTransformerClassifier(nn.Module):
    """
    Wraps GraphTransformerEncoder with a classification head.

    Args:
        encoder: GraphTransformerEncoder instance.
        num_classes: Number of classes for classification.
    """

    def __init__(self, encoder: GraphTransformerEncoder, num_classes: int):
        super().__init__()
        self.encoder = encoder
        d_model = encoder.norm.normalized_shape[0]
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        node_emb, graph_emb = self.encoder(x, key_padding_mask=key_padding_mask)
        logits = self.head(graph_emb)
        return node_emb, graph_emb, logits


class GraphTransformerRegressor(nn.Module):
    """
    Wraps GraphTransformerEncoder with a regression head.

    Args:
        encoder: GraphTransformerEncoder instance.
        out_dim: Output dimension for regression.
    """

    def __init__(self, encoder: GraphTransformerEncoder, out_dim: int = 1):
        super().__init__()
        self.encoder = encoder
        d_model = encoder.norm.normalized_shape[0]
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        node_emb, graph_emb = self.encoder(x, key_padding_mask=key_padding_mask)
        preds = self.head(graph_emb)
        return node_emb, graph_emb, preds


# -----------------------------
# Data utilities (padding/collate)
# -----------------------------

def _to_tensor(x: Union[np.ndarray, "scipy.sparse.spmatrix", Tensor]) -> Tensor:
    if isinstance(x, Tensor):
        return x
    if hasattr(x, "toarray"):
        x = x.toarray()
    return torch.from_numpy(np.asarray(x, dtype=np.float32))


def pad_batch(batch: List[Tuple[Tensor, Optional[int]]], add_cls: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Pad a list of variable-length node-embedding matrices to a uniform length.

    Args:
        batch: List of (nodes x features) tensors.
        add_cls: Whether a CLS token will be prepended later (kept for clarity).

    Returns:
        padded: [B, Nmax, F] batch tensor.
        key_padding_mask: [B, Nmax] mask with True for PAD positions.
    """
    lengths = [b.size(0) for b, _ in batch]
    B = len(batch)
    Nmax = max(lengths) if lengths else 0
    Fdim = batch[0][0].size(1) if Nmax > 0 else 0

    padded = torch.zeros((B, Nmax, Fdim), dtype=batch[0][0].dtype)
    mask = torch.ones((B, Nmax), dtype=torch.bool)

    for i, (t, _) in enumerate(batch):
        n = t.size(0)
        padded[i, :n, :] = t
        mask[i, :n] = False  # False = valid token, True = PAD (PyTorch semantics)
    return padded, mask


# -----------------------------
# Scikit-style estimator
# -----------------------------

class NeuralGraphEstimator:
    """
    Transformer-based graph classifier with a scikit-like interface.

    Scikit-style interface with node-level vectorization and transformer pooling.

    Notes:
        - Input dimension is inferred on first call by vectorizing one sample.
        - Padding/masking handles variable-size graphs per batch.
    """

    def __init__(
        self,
        node_vectorizer: AbstractGraphNodeTransformer,  # node-level feature extractor
        num_classes: Optional[int] = None,  # class count for classification
        output_dim: int = 1,  # regression output dimensionality
        mode: Literal["classification", "regression"] = "classification",  # task type
        # Transformer hyperparameters
        d_model: int = 128,  # transformer hidden size
        nhead: int = 4,  # attention heads
        num_layers: int = 2,  # transformer depth
        dim_feedforward: int = 256,  # FFN width
        dropout: float = 0.1,  # dropout rate
        activation: Literal["relu", "gelu"] = "relu",  # activation function
        pooling: Literal["mean", "max", "cls"] = "mean",  # graph pooling strategy
        # Input adapter configuration
        adapter_type: Literal["linear", "factorized"] = "linear",  # input adapter type
        adapter: Optional[nn.Module] = None,  # custom adapter module override
        adapter_bottleneck: Optional[int] = None,  # bottleneck width for factorized adapter
        adapter_bias: bool = True,  # include bias in adapter
        adapter_init: Optional[Literal["svd", "default"]] = None,  # adapter initialization scheme
        # LoRA configuration (for fine-tuning)
        lora_r: int = 8,  # LoRA rank
        lora_alpha: float = 16.0,  # LoRA scaling factor
        lora_dropout: float = 0.0,  # LoRA dropout on adapter inputs
        lora_scope: Literal["encoder_only", "head_only", "ffn_only", "adapter_only", "all_linear"] = "encoder_only",  # LoRA target scope
        # Training hyperparameters
        epochs: int = 20,  # training epochs
        batch_size: int = 16,  # minibatch size
        lr: float = 1e-3,  # learning rate
        weight_decay: float = 0.0,  # L2 weight decay
        class_weights: Optional[Union[List[float], np.ndarray, Tensor]] = None,  # optional class weighting tensor
        val_split: float = 0.0,  # validation split fraction
        early_stopping_patience: Optional[int] = None,  # epochs to wait before stopping
        early_stopping_metric: Literal["val_loss", "val_acc"] = "val_loss",  # metric to monitor
        early_stopping_mode: Literal["min", "max"] = "min",  # minimize or maximize metric
        seed: Optional[int] = None,  # RNG seed for reproducibility
        device: Optional[Union[str, torch.device]] = None,  # training device override
        verbose: bool = True,  # log training progress
        # Auxiliary losses (regularization)
        aux_whitening_weight: float = 0.0,  # whitening loss weight
        aux_sparsity_w2_weight: float = 0.0,  # sparsity weight for W2
    ) -> None:
        self.node_vectorizer = node_vectorizer  # node-level feature extractor
        self.mode = mode  # classification or regression
        self.num_classes = (None if num_classes is None else int(num_classes))  # class count for classification
        self.output_dim = int(output_dim)  # regression output dimensionality

        # Save model hparams
        self.d_model = int(d_model)  # transformer hidden size
        self.nhead = int(nhead)  # attention heads
        self.num_layers = int(num_layers)  # transformer depth
        self.dim_feedforward = int(dim_feedforward)  # FFN width
        self.dropout = float(dropout)  # dropout rate
        self.activation = activation  # activation function
        self.pooling = pooling  # graph pooling strategy
        # Unify control: CLS usage is solely determined by pooling
        self.use_cls_token = (pooling == "cls")  # whether to insert CLS token
        self.adapter_type = adapter_type  # input adapter type
        self.adapter = adapter  # custom adapter override
        self.adapter_bottleneck = adapter_bottleneck  # bottleneck width for factorized adapter
        self.adapter_bias = bool(adapter_bias)  # include bias in adapter
        self.adapter_init = adapter_init  # adapter initialization scheme
        # LoRA
        self.lora_r = int(lora_r)  # LoRA rank
        self.lora_alpha = float(lora_alpha)  # LoRA scaling factor
        self.lora_dropout = float(lora_dropout)  # LoRA dropout on adapter inputs
        self.lora_scope = lora_scope  # which modules to LoRA-wrap

        # Save training hparams
        self.epochs = int(epochs)  # training epochs
        self.batch_size = int(batch_size)  # minibatch size
        self.lr = float(lr)  # learning rate
        self.weight_decay = float(weight_decay)  # L2 weight decay
        self.val_split = float(val_split)  # validation split fraction
        self.early_stopping_patience = early_stopping_patience  # epochs to wait before stopping
        self.early_stopping_metric = early_stopping_metric  # metric to monitor
        self.early_stopping_mode = early_stopping_mode  # minimize or maximize metric
        self.verbose = bool(verbose)  # log training progress
        # Aux loss weights
        self.aux_whitening_weight = float(aux_whitening_weight)  # whitening loss weight
        self.aux_sparsity_w2_weight = float(aux_sparsity_w2_weight)  # sparsity weight for W2

        if class_weights is not None and not isinstance(class_weights, Tensor):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.class_weights = class_weights  # optional class weighting tensor

        # Device & seed
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")  # training device
        if seed is not None:  # RNG seed for reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Lazy-initialized components
        self.input_dim_: Optional[int] = None  # inferred input feature dim
        self.model_: Optional[Union[GraphTransformerClassifier, GraphTransformerRegressor]] = None  # built model instance
        self.history_: Dict[str, List[float]] = {}  # training history metrics
        # LoRA bookkeeping
        self._lora_active: bool = False
        self._lora_wrapped_: List[str] = []

    # ------------------
    # Private utilities
    # ------------------
    def _infer_input_dim(self, X: Sequence[Any]) -> int:
        if self.input_dim_ is not None:
            return self.input_dim_
        if len(X) == 0:
            raise ValueError("Empty dataset; cannot infer input dimension.")
        # vectorize first sample to get feature dimension
        sample = self.node_vectorizer.transform([X[0]])[0]
        # sample: (n_nodes, n_features) dense or csr
        if hasattr(sample, "shape") and len(sample.shape) == 2:
            self.input_dim_ = int(sample.shape[1])
        else:
            t = _to_tensor(sample)
            self.input_dim_ = int(t.size(1))
        return self.input_dim_

    def _ensure_model(self, X: Sequence[Any]) -> None:
        in_dim = self._infer_input_dim(X)
        if self.model_ is None:
            encoder = GraphTransformerEncoder(
                in_dim=in_dim,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                activation=self.activation,
                pooling=self.pooling,
                adapter=self.adapter,
                adapter_type=self.adapter_type,
                adapter_bottleneck=self.adapter_bottleneck,
                adapter_bias=self.adapter_bias,
                adapter_init=self.adapter_init,
            )
            if self.mode == "classification":
                if self.num_classes is None:
                    raise ValueError("num_classes must be provided in classification mode.")
                self.model_ = GraphTransformerClassifier(encoder, num_classes=int(self.num_classes)).to(self.device)
            elif self.mode == "regression":
                self.model_ = GraphTransformerRegressor(encoder, out_dim=int(self.output_dim)).to(self.device)
            else:
                raise ValueError("mode must be 'classification' or 'regression'")

    def _vectorize_graphs(self, X: Sequence[Any]) -> List[Tensor]:
        mats = self.node_vectorizer.transform(list(X))
        tensors = [_to_tensor(m).float() for m in mats]
        return tensors

    def _iter_minibatches(self, X: Sequence[Any], y: Optional[Sequence[int]] = None, shuffle: bool = True):
        """
        Backward-compatible iterator that vectorizes on-the-fly.
        Prefer using `_iter_minibatches_from_tensors` when possible.

        Args:
            X: Sequence of input graphs.
            y: Optional targets aligned to X.
            shuffle: Whether to shuffle indices.

        Returns:
            Iterator yielding (padded, mask, yb) batches.
        """
        tensors = self._vectorize_graphs(X)
        yield from self._iter_minibatches_from_tensors(tensors, y=y, shuffle=shuffle)

    def _iter_minibatches_from_tensors(
        self,
        tensors: Sequence[Tensor],
        y: Optional[Sequence[Any]] = None,
        shuffle: bool = True,
        y_dtype: Optional[torch.dtype] = None,
    ):
        idx = np.arange(len(tensors))
        if shuffle:
            np.random.shuffle(idx)
        for start in range(0, len(tensors), self.batch_size):
            sl = idx[start : start + self.batch_size]
            mats = [tensors[i] for i in sl]
            batch = [(m, None) for m in mats]
            padded, mask = pad_batch(batch, add_cls=self.use_cls_token)
            if y is None:
                yb = None
            else:
                # Default dtype: long for classification, float for regression
                if y_dtype is None:
                    y_dtype = torch.long if self.mode == "classification" else torch.float32
                y_slice = [y[i] for i in sl]
                yb = torch.tensor(y_slice, dtype=y_dtype)
            yield (
                padded.to(self.device),
                mask.to(self.device),
                (None if yb is None else yb.to(self.device)),
            )

    def _resolve_num_classes(self, y: Sequence[Any]) -> None:
        if self.mode != "classification":
            return
        if self.num_classes is not None:
            return
        import numpy as _np
        y_arr = _np.asarray(y)
        classes = _np.unique(y_arr)
        if classes.size < 2:
            raise ValueError("Classification requires at least two classes in y.")
        self.num_classes = int(classes.size)

    def _validate_class_weights(self) -> None:
        if self.mode != "classification":
            return
        if self.class_weights is None or self.num_classes is None:
            return
        if int(self.class_weights.numel()) != int(self.num_classes):
            raise ValueError(
                f"class_weights length ({int(self.class_weights.numel())}) does not match num_classes ({int(self.num_classes)})."
            )

    def _build_criterion(self) -> nn.Module:
        if self.mode == "classification":
            if self.class_weights is not None:
                cw = self.class_weights.to(self.device)
                return nn.CrossEntropyLoss(weight=cw)
            return nn.CrossEntropyLoss()
        return nn.MSELoss()

    def _vectorize_and_split(
        self,
        X: Sequence[Any],
        y: Sequence[Any],
    ) -> Tuple[List[Tensor], Sequence[Any], List[Tensor], Sequence[Any]]:
        n_total = len(X)
        all_tensors = self._vectorize_graphs(X)
        if self.val_split and self.val_split > 0.0:
            n_val = int(self.val_split * n_total)
            perm = np.random.permutation(n_total)
            val_idx = set(perm[:n_val].tolist())
            train_idx = [i for i in range(n_total) if i not in val_idx]
            train_tensors = [all_tensors[i] for i in train_idx]
            y_train = [y[i] for i in train_idx]
            val_tensors = [all_tensors[i] for i in val_idx]
            y_val = [y[i] for i in val_idx]
            return train_tensors, y_train, val_tensors, y_val
        return all_tensors, y, [], []

    def _vectorize_and_split_with_valsplit(
        self,
        X: Sequence[Any],
        y: Sequence[Any],
        val_split: float,
    ) -> Tuple[List[Tensor], Sequence[Any], List[Tensor], Sequence[Any]]:
        n_total = len(X)
        all_tensors = self._vectorize_graphs(X)
        if val_split and val_split > 0.0:
            n_val = int(val_split * n_total)
            perm = np.random.permutation(n_total)
            val_idx = set(perm[:n_val].tolist())
            train_idx = [i for i in range(n_total) if i not in val_idx]
            train_tensors = [all_tensors[i] for i in train_idx]
            y_train = [y[i] for i in train_idx]
            val_tensors = [all_tensors[i] for i in val_idx]
            y_val = [y[i] for i in val_idx]
            return train_tensors, y_train, val_tensors, y_val
        return all_tensors, y, [], []

    def _init_history(self) -> None:
        self.history_ = {"loss": [], "val_loss": []}
        if self.mode == "classification":
            self.history_.update({"acc": [], "val_acc": []})
        else:
            self.history_.update({"mse": [], "val_mse": []})

    def _init_history_dict(self) -> Dict[str, List[float]]:
        history = {"loss": [], "val_loss": []}
        if self.mode == "classification":
            history.update({"acc": [], "val_acc": []})
        else:
            history.update({"mse": [], "val_mse": []})
        return history

    def _validate_early_stopping_config(
        self,
        metric: Literal["val_loss", "val_acc", "val_mse"],
        mode: Literal["min", "max"],
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError("early_stopping_mode must be 'min' or 'max'")
        if self.mode == "classification":
            allowed = ("val_loss", "val_acc")
        else:
            allowed = ("val_loss", "val_mse")
        if metric not in allowed:
            raise ValueError(f"early_stopping_metric must be one of {allowed}")

    def _train_one_epoch(
        self,
        model: nn.Module,
        train_tensors: Sequence[Tensor],
        y_train: Sequence[Any],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        use_aux_losses: bool = True,
    ) -> Tuple[float, Dict[str, float]]:
        epoch_loss = 0.0
        n_batches = 0
        correct = 0
        total = 0
        se_accum = 0.0
        for xb, mask, yb in self._iter_minibatches_from_tensors(
            train_tensors, y_train, shuffle=True, y_dtype=(torch.long if self.mode == "classification" else torch.float32)
        ):
            optimizer.zero_grad()
            _, _, out = model(xb, key_padding_mask=mask)
            if self.mode == "classification":
                loss = criterion(out, yb)
            else:
                if yb.dim() == 1 and isinstance(self.model_, GraphTransformerRegressor) and self.output_dim == 1:
                    yb = yb.unsqueeze(-1)
                loss = criterion(out, yb)
            if use_aux_losses and (self.aux_whitening_weight > 0.0 or self.aux_sparsity_w2_weight > 0.0):
                aux_loss = self._compute_aux_loss(xb, mask)
                if aux_loss != 0.0:
                    loss = loss + aux_loss
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            n_batches += 1
            if self.mode == "classification":
                pred = out.argmax(dim=-1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())
            else:
                se_accum += float(F.mse_loss(out.detach(), yb, reduction="mean").cpu())
        epoch_loss = epoch_loss / max(n_batches, 1)
        metrics: Dict[str, float] = {}
        if self.mode == "classification":
            metrics["train_acc"] = (correct / max(total, 1)) if total else 0.0
            metrics["train_err"] = float((total - correct) if total else 0)
        else:
            metrics["train_mse"] = se_accum / max(n_batches, 1)
        metrics["total"] = float(total)
        metrics["correct"] = float(correct)
        return epoch_loss, metrics

    def _eval_one_epoch(
        self,
        model: nn.Module,
        val_tensors: Sequence[Tensor],
        y_val: Sequence[Any],
        criterion: nn.Module,
    ) -> Tuple[Optional[float], Dict[str, float]]:
        if not val_tensors:
            return None, {}
        model.eval()
        vloss = 0.0
        vcnt = 0
        vcorrect = 0
        vtotal = 0
        vmse_accum = 0.0
        with torch.no_grad():
            for xb, mask, yb in self._iter_minibatches_from_tensors(
                val_tensors, y_val, shuffle=False, y_dtype=(torch.long if self.mode == "classification" else torch.float32)
            ):
                _, _, out = model(xb, key_padding_mask=mask)
                if self.mode == "classification":
                    loss = criterion(out, yb)
                else:
                    if yb.dim() == 1 and isinstance(self.model_, GraphTransformerRegressor) and self.output_dim == 1:
                        yb = yb.unsqueeze(-1)
                    loss = criterion(out, yb)
                vloss += float(loss.detach().cpu())
                vcnt += 1
                if self.mode == "classification":
                    vpred = out.argmax(dim=-1)
                    vcorrect += int((vpred == yb).sum().item())
                    vtotal += int(yb.numel())
                else:
                    vmse_accum += float(F.mse_loss(out, yb, reduction="mean").cpu())
        val_loss = vloss / max(vcnt, 1)
        metrics: Dict[str, float] = {}
        if self.mode == "classification":
            metrics["val_acc"] = (vcorrect / max(vtotal, 1)) if vtotal else 0.0
            metrics["val_err"] = float(vtotal - vcorrect)
        else:
            metrics["val_mse"] = vmse_accum / max(vcnt, 1)
        model.train()
        return val_loss, metrics

    def _compute_aux_loss(self, xb: Tensor, mask: Tensor) -> Tensor:
        enc = self.model_.encoder if self.model_ is not None else None
        adapter = getattr(enc, "input_adapter", None) if enc is not None else None
        aux_loss = 0.0
        if adapter is not None and isinstance(adapter, InputAdapterFactorized):
            if self.aux_whitening_weight > 0.0:
                z = adapter.proj1(xb)
                B, N, b = z.shape
                z2 = z.reshape(B * N, b)
                valid = (~mask).reshape(B * N)
                if valid.any():
                    z2 = z2[valid]
                    if z2.shape[0] > 1:
                        z2 = z2 - z2.mean(dim=0, keepdim=True)
                        C = (z2.T @ z2) / (z2.shape[0] - 1)
                        I = torch.eye(b, device=z2.device, dtype=z2.dtype)
                        L_white = (C - I).pow(2).mean()
                        aux_loss = aux_loss + (self.aux_whitening_weight * L_white)
            if self.aux_sparsity_w2_weight > 0.0:
                W2 = adapter.proj2.weight
                L_sp = W2.abs().mean()
                aux_loss = aux_loss + (self.aux_sparsity_w2_weight * L_sp)
        return aux_loss

    def _prepare_finetune_model(self, reset_lora: bool = False) -> List[Tensor]:
        assert self.model_ is not None
        model = self.model_
        for p in model.parameters():
            p.requires_grad = False
        wrapped = self._inject_lora_adapters(reset=reset_lora)
        if self.verbose:
            print(f"LoRA injected into {len(wrapped)} linear layer(s): {', '.join(wrapped) if wrapped else 'none'}")
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            if hasattr(model, "head") and isinstance(model.head, nn.Linear):
                for p in model.head.parameters():
                    p.requires_grad = True
                trainable_params = list(model.head.parameters())
                if self.verbose:
                    print("No LoRA targets found; unfreezing head for fine-tuning.")
            else:
                raise RuntimeError("No parameters available to fine-tune.")
        return trainable_params

    def _update_early_stopping(
        self,
        val_loss: Optional[float],
        val_metrics: Dict[str, float],
        best_metric: float,
        best_state: Optional[Dict[str, Tensor]],
        best_epoch: Optional[int],
        wait: int,
        epoch: int,
        metric: Literal["val_loss", "val_acc", "val_mse"],
        mode: Literal["min", "max"],
        patience: Optional[int],
    ) -> Tuple[float, Optional[Dict[str, Tensor]], Optional[int], int, bool]:
        if val_loss is None:
            return best_metric, best_state, best_epoch, wait, False
        if metric == "val_loss":
            current_metric = val_loss
        elif metric == "val_acc":
            current_metric = val_metrics.get("val_acc", 0.0)
        else:
            current_metric = val_metrics.get("val_mse", 0.0)
        improved = (
            (mode == "min" and current_metric < best_metric - 1e-6)
            or (mode == "max" and current_metric > best_metric + 1e-6)
        )
        if improved:
            best_metric = current_metric
            best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
            best_epoch = epoch + 1
            wait = 0
        else:
            wait += 1
        return best_metric, best_state, best_epoch, wait, (patience is not None and wait >= patience)

    def _log_epoch(
        self,
        epoch: int,
        epoch_num_width: int,
        epoch_loss: float,
        train_metrics: Dict[str, float],
        val_loss: Optional[float],
        val_metrics: Dict[str, float],
        start_time: float,
        total_epochs: int,
        label: str = "epoch",
        include_errors: bool = True,
    ) -> None:
        elapsed_sec = time.perf_counter() - start_time
        epochs_done = epoch + 1
        remaining = max(total_epochs - epochs_done, 0)
        avg_per_epoch = elapsed_sec / max(epochs_done, 1)
        eta_sec = avg_per_epoch * remaining
        elapsed_str = str(timedelta(seconds=int(elapsed_sec)))
        eta_str = str(timedelta(seconds=int(eta_sec)))
        epoch_header = f"{label} {epochs_done:>{epoch_num_width+2}d}/{total_epochs}"

        if self.mode == "classification":
            train_acc = train_metrics.get("train_acc", 0.0)
            tr_err = int(train_metrics.get("train_err", 0))
            if val_loss is None:
                line = (
                    f"{epoch_header}  | "
                    f"train_loss:{epoch_loss:9.4f}"
                    f" | train_acc:{train_acc:7.3f}"
                    f" | elapsed:{elapsed_str:>9}  eta:{eta_str:>9}"
                )
            else:
                val_acc = val_metrics.get("val_acc", 0.0)
                val_err = int(val_metrics.get("val_err", 0))
                if include_errors:
                    line = (
                        f"{epoch_header}  | "
                        f"train_loss:{epoch_loss:9.4f}  val_loss:{val_loss:9.4f} | "
                        f"train_acc:{train_acc:7.3f}  val_acc:{val_acc:7.3f} | "
                        f"tr_err:{tr_err:7d}  val_err:{val_err:7d} | "
                        f"elapsed:{elapsed_str:>9}  eta:{eta_str:>9}"
                    )
                else:
                    line = (
                        f"{epoch_header}  | "
                        f"train_loss:{epoch_loss:9.4f}  val_loss:{val_loss:9.4f} | "
                        f"train_acc:{train_acc:7.3f}  val_acc:{val_acc:7.3f} | "
                        f"elapsed:{elapsed_str:>9}  eta:{eta_str:>9}"
                    )
            print(line)
        else:
            train_mse = train_metrics.get("train_mse", 0.0)
            if val_loss is None:
                line = (
                    f"{epoch_header}  | "
                    f"train_loss:{epoch_loss:9.4f} | train_mse:{train_mse:9.4f} | "
                    f"elapsed:{elapsed_str:>9}  eta:{eta_str:>9}"
                )
            else:
                val_mse = val_metrics.get("val_mse", 0.0)
                line = (
                    f"{epoch_header}  | "
                    f"train_loss:{epoch_loss:9.4f}  val_loss:{val_loss:9.4f} | "
                    f"train_mse:{train_mse:9.4f}  val_mse:{val_mse:9.4f} | "
                    f"elapsed:{elapsed_str:>9}  eta:{eta_str:>9}"
                )
            print(line)

    def _log_pretrain_epoch(
        self,
        epoch: int,
        epoch_num_width: int,
        epoch_loss: float,
        val_loss: Optional[float],
        start_time: float,
        total_epochs: int,
        skipped_batches: int,
        total_batches: int,
        label: str = "pretrain",
    ) -> None:
        elapsed_sec = time.perf_counter() - start_time
        epochs_done = epoch + 1
        remaining = max(total_epochs - epochs_done, 0)
        avg_per_epoch = elapsed_sec / max(epochs_done, 1)
        eta_sec = avg_per_epoch * remaining
        elapsed_str = str(timedelta(seconds=int(elapsed_sec)))
        eta_str = str(timedelta(seconds=int(eta_sec)))
        epoch_header = f"{label} {epochs_done:>{epoch_num_width+2}d}/{total_epochs}"

        if val_loss is None or math.isnan(val_loss):
            line = (
                f"{epoch_header}  | "
                f"train_loss:{epoch_loss:9.4f} | "
                f"elapsed:{elapsed_str:>9}  eta:{eta_str:>9}"
            )
        else:
            line = (
                f"{epoch_header}  | "
                f"train_loss:{epoch_loss:9.4f}  val_loss:{val_loss:9.4f} | "
                f"elapsed:{elapsed_str:>9}  eta:{eta_str:>9}"
            )
        if skipped_batches > 0:
            line = f"{line} | skipped:{skipped_batches}/{total_batches}"
        print(line)

    def _extract_motif_assignments(self, abstract_graph: AbstractGraph) -> List[Tuple[int, List[int]]]:
        """
        Map image node labels to their associated preimage node indices.

        Args:
            abstract_graph: AbstractGraph instance with labeled image nodes.

        Returns:
            List of (label, node_indices) pairs for each image node.
        """
        base_nodes = list(abstract_graph.preimage_graph.nodes())
        node_index = {node_id: i for i, node_id in enumerate(base_nodes)}
        motifs: List[Tuple[int, List[int]]] = []
        for _, data in abstract_graph.image_graph.nodes(data=True):
            label = data.get("label", None)
            if label is None:
                continue
            try:
                label_int = int(label)
            except (TypeError, ValueError):
                continue
            association = data.get("association", None)
            if association is None:
                continue
            idxs = [node_index[n] for n in association.nodes() if n in node_index]
            if idxs:
                motifs.append((label_int, idxs))
        return motifs

    def _motif_contrastive_loss(
        self,
        node_emb: Tensor,
        batch_motifs: Sequence[List[Tuple[int, List[int]]]],
        tau_base: float = 0.1,
    ) -> Optional[Tensor]:
        """
        Compute prototype InfoNCE loss over motif embeddings in a batch.

        Args:
            node_emb: Node embeddings [B, N(+1), D] from the encoder.
            batch_motifs: Per-graph motif assignments (label, node indices).
            tau_base: Base temperature multiplier for frequency-aware scaling.

        Returns:
            Loss scalar, or None if no valid motifs or only one motif type.
        """
        if self.model_ is None:
            return None
        if self.model_.encoder.use_cls_token:
            node_emb = node_emb[:, 1:, :]

        embeddings: List[Tensor] = []
        labels: List[int] = []
        for b_idx, motifs in enumerate(batch_motifs):
            for label, idxs in motifs:
                if not idxs:
                    continue
                idxs_tensor = torch.tensor(idxs, device=node_emb.device, dtype=torch.long)
                idxs_tensor = idxs_tensor[idxs_tensor < node_emb.size(1)]
                if idxs_tensor.numel() == 0:
                    continue
                emb = node_emb[b_idx, idxs_tensor, :].sum(dim=0)
                embeddings.append(emb)
                labels.append(label)

        if not embeddings:
            return None

        labels_tensor = torch.tensor(labels, device=node_emb.device, dtype=torch.long)
        unique_labels, inverse = labels_tensor.unique(return_inverse=True)
        num_types = unique_labels.numel()
        if num_types < 2:
            return None

        emb_tensor = torch.stack(embeddings, dim=0)  # [M, D]
        proto_sums = torch.zeros(num_types, emb_tensor.size(1), device=node_emb.device)
        proto_counts = torch.zeros(num_types, device=node_emb.device)
        proto_sums.index_add_(0, inverse, emb_tensor)
        proto_counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=proto_counts.dtype))
        prototypes = proto_sums / proto_counts.unsqueeze(1)

        emb_norm = F.normalize(emb_tensor, p=2, dim=1)
        prot_norm = F.normalize(prototypes, p=2, dim=1)
        sim = emb_norm @ prot_norm.t()

        tau = tau_base * torch.log(proto_counts + 1.0)
        tau = tau.clamp_min(1e-6)
        logits = sim / tau.unsqueeze(0)
        log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        loss = -log_probs[torch.arange(emb_tensor.size(0), device=node_emb.device), inverse].mean()
        return loss

    def _run_training_loop(
        self,
        model: nn.Module,
        train_tensors: Sequence[Tensor],
        y_train: Sequence[Any],
        val_tensors: Sequence[Tensor],
        y_val: Sequence[Any],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        epochs: int,
        early_stopping_metric: Literal["val_loss", "val_acc", "val_mse"],
        early_stopping_mode: Literal["min", "max"],
        early_stopping_patience: Optional[int],
        history: Dict[str, List[float]],
        label: str,
        include_errors: bool,
        use_aux_losses: bool,
        restore_message: bool,
    ) -> Tuple[Optional[Dict[str, Tensor]], Optional[int], float]:
        best_metric = float("inf") if early_stopping_mode == "min" else float("-inf")
        best_state: Optional[Dict[str, Tensor]] = None
        best_epoch: Optional[int] = None
        wait = 0

        start_time = time.perf_counter()
        epoch_num_width = max(2, len(str(epochs)))

        for epoch in range(epochs):
            epoch_loss, train_metrics = self._train_one_epoch(
                model,
                train_tensors,
                y_train,
                criterion,
                optimizer,
                use_aux_losses=use_aux_losses,
            )
            history["loss"].append(epoch_loss)
            if self.mode == "classification":
                history["acc"].append(train_metrics.get("train_acc", 0.0))
            else:
                history["mse"].append(train_metrics.get("train_mse", 0.0))

            val_loss, val_metrics = self._eval_one_epoch(model, val_tensors, y_val, criterion)
            if val_loss is not None:
                history["val_loss"].append(val_loss)
                if self.mode == "classification":
                    history["val_acc"].append(val_metrics.get("val_acc", 0.0))
                else:
                    history["val_mse"].append(val_metrics.get("val_mse", 0.0))
                best_metric, best_state, best_epoch, wait, stop = self._update_early_stopping(
                    val_loss,
                    val_metrics,
                    best_metric,
                    best_state,
                    best_epoch,
                    wait,
                    epoch,
                    metric=early_stopping_metric,
                    mode=early_stopping_mode,
                    patience=early_stopping_patience,
                )
                if stop:
                    if self.verbose:
                        if restore_message:
                            metric_name = early_stopping_metric
                            msg_epoch = best_epoch if best_epoch is not None else epoch + 1
                            print(
                                f"Early stopping at epoch {epoch+1}; restoring best epoch {msg_epoch} "
                                f"({metric_name}={best_metric:.4f})"
                            )
                        else:
                            print(f"Early stopping at epoch {epoch+1}; best {early_stopping_metric}={best_metric:.4f}")
                    break

            if self.verbose:
                self._log_epoch(
                    epoch,
                    epoch_num_width,
                    epoch_loss,
                    train_metrics,
                    val_loss,
                    val_metrics,
                    start_time,
                    total_epochs=epochs,
                    label=label,
                    include_errors=include_errors,
                )

        return best_state, best_epoch, best_metric

    # --------------
    # Public API
    # --------------
    def pre_train(
        self,
        graphs: Sequence[nx.Graph],
        decomposition_function: Callable[[AbstractGraph], AbstractGraph],
        nbits: int,
        n_jobs: int = -1,
    ) -> "NeuralGraphEstimator":
        """
        Pre-train the encoder with motif-level prototype InfoNCE.

        Args:
            graphs: Sequence of input graphs.
            decomposition_function: AbstractGraph decomposition function to build image nodes.
            nbits: Hash bit width for image node labels.
            n_jobs: Number of workers for AbstractGraph construction.

        Returns:
            Self.
        """
        if not graphs:
            return self

        abstract_graphs = graphs_to_abstract_graphs(
            graphs,
            decomposition_function=decomposition_function,
            nbits=nbits,
            n_jobs=n_jobs,
        )
        tensors = self._vectorize_graphs(graphs)
        motifs = [self._extract_motif_assignments(ag) for ag in abstract_graphs]

        self._ensure_model(graphs)
        assert self.model_ is not None
        model = self.model_
        model.train()

        for p in model.encoder.parameters():
            p.requires_grad = True

        if self.verbose:
            total_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
            enc_params = total_params
            head_params = 0
            print(f"Model parameters: total={total_params:,} (encoder={enc_params:,}, head={head_params:,})")

        optimizer = torch.optim.AdamW(
            [p for p in model.encoder.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        n_total = len(tensors)
        if self.val_split and self.val_split > 0.0:
            n_val = int(self.val_split * n_total)
            perm = np.random.permutation(n_total)
            val_idx = set(perm[:n_val].tolist())
            train_idx = [i for i in range(n_total) if i not in val_idx]
            train_tensors = [tensors[i] for i in train_idx]
            train_motifs = [motifs[i] for i in train_idx]
            val_tensors = [tensors[i] for i in val_idx]
            val_motifs = [motifs[i] for i in val_idx]
        else:
            train_tensors = tensors
            train_motifs = motifs
            val_tensors = []
            val_motifs = []

        def _iter_batches(
            batch_tensors: Sequence[Tensor],
            batch_motifs: Sequence[List[Tuple[int, List[int]]]],
            shuffle: bool = True,
        ):
            idx = np.arange(len(batch_tensors))
            if shuffle:
                np.random.shuffle(idx)
            for start in range(0, len(batch_tensors), self.batch_size):
                sl = idx[start : start + self.batch_size]
                mats = [batch_tensors[i] for i in sl]
                motifs_sl = [batch_motifs[i] for i in sl]
                batch = [(m, None) for m in mats]
                padded, mask = pad_batch(batch, add_cls=self.use_cls_token)
                yield padded.to(self.device), mask.to(self.device), motifs_sl

        self.pretrain_history_ = {"loss": [], "val_loss": []}
        best_metric = float("inf")
        best_state: Optional[Dict[str, Tensor]] = None
        best_epoch: Optional[int] = None
        wait = 0
        start_time = time.perf_counter()
        epoch_num_width = max(2, len(str(self.epochs)))

        for epoch in range(self.epochs):
            model.train()
            loss_sum = 0.0
            batch_count = 0
            skipped = 0
            total_batches = 0
            for xb, mask, batch_motifs in _iter_batches(train_tensors, train_motifs, shuffle=True):
                total_batches += 1
                optimizer.zero_grad()
                node_emb, _ = model.encoder(xb, key_padding_mask=mask)
                loss = self._motif_contrastive_loss(node_emb, batch_motifs, tau_base=0.1)
                if loss is None:
                    skipped += 1
                    continue
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach().cpu())
                batch_count += 1

            epoch_loss = loss_sum / batch_count if batch_count > 0 else float("nan")
            self.pretrain_history_["loss"].append(epoch_loss)

            val_loss = None
            if val_tensors:
                model.eval()
                val_loss_sum = 0.0
                val_count = 0
                with torch.no_grad():
                    for xb, mask, batch_motifs in _iter_batches(val_tensors, val_motifs, shuffle=False):
                        node_emb, _ = model.encoder(xb, key_padding_mask=mask)
                        loss = self._motif_contrastive_loss(node_emb, batch_motifs, tau_base=0.1)
                        if loss is None:
                            continue
                        val_loss_sum += float(loss.detach().cpu())
                        val_count += 1
                if val_count > 0:
                    val_loss = val_loss_sum / val_count
                    self.pretrain_history_["val_loss"].append(val_loss)

                if val_loss is not None:
                    best_metric, best_state, best_epoch, wait, stop = self._update_early_stopping(
                        val_loss,
                        {},
                        best_metric,
                        best_state,
                        best_epoch,
                        wait,
                        epoch,
                        metric="val_loss",
                        mode="min",
                        patience=self.early_stopping_patience,
                    )
                    if stop:
                        if self.verbose:
                            msg_epoch = best_epoch if best_epoch is not None else epoch + 1
                            print(
                                f"Early stopping at epoch {epoch+1}; restoring best epoch {msg_epoch} "
                                f"(val_loss={best_metric:.4f})"
                            )
                        break

            if self.verbose:
                # Use the same label as standard training to unify console output
                self._log_pretrain_epoch(
                    epoch,
                    epoch_num_width,
                    epoch_loss,
                    val_loss,
                    start_time,
                    total_epochs=self.epochs,
                    skipped_batches=skipped,
                    total_batches=total_batches,
                    label="epoch",
                )

        if best_state is not None:
            model.load_state_dict(best_state)

        return self

    def fit(self, graphs: Sequence[nx.Graph], targets: Sequence[Any]) -> "NeuralGraphEstimator":
        """Train the estimator (classification or regression).

        Args:
            X: sequence of graphs (compatible with AbstractGraphNodeTransformer).
            y: targets (class indices for classification; floats/arrays for regression).

        Returns:
            Self.
        """
        self._resolve_num_classes(targets)
        self._validate_class_weights()

        self._ensure_model(graphs)
        assert self.model_ is not None
        model = self.model_
        model.train()
        # Print parameter count
        if self.verbose:
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            enc_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
            head_params = total_params - enc_params
            print(f"Model parameters: total={total_params:,} (encoder={enc_params:,}, head={head_params:,})")

        criterion = self._build_criterion()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        train_tensors, y_train, val_tensors, y_val = self._vectorize_and_split(graphs, targets)

        # Simple training loop (optional early stopping on val split)
        self._validate_early_stopping_config(self.early_stopping_metric, self.early_stopping_mode)

        self._init_history()
        best_state, best_epoch, best_metric = self._run_training_loop(
            model,
            train_tensors,
            y_train,
            val_tensors,
            y_val,
            criterion,
            optimizer,
            epochs=self.epochs,
            early_stopping_metric=self.early_stopping_metric,
            early_stopping_mode=self.early_stopping_mode,
            early_stopping_patience=self.early_stopping_patience,
            history=self.history_,
            label="epoch",
            include_errors=True,
            use_aux_losses=True,
            restore_message=True,
        )

        # Load best validation state if tracked
        if best_state is not None:
            model.load_state_dict(best_state)

        return self

    # ------------------
    # LoRA utilities
    # ------------------
    def _iter_all_named_modules(self, root: nn.Module):
        # Yields (full_name, module) for all submodules (depth-first)
        for name, module in root.named_modules():
            yield name, module

    def _get_parent_and_attr(self, root: nn.Module, full_name: str) -> Tuple[nn.Module, str]:
        parts = full_name.split(".") if full_name else []
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        return parent, (parts[-1] if parts else "")

    def _should_wrap_linear_with_lora(self, full_name: str) -> bool:
        scope = self.lora_scope
        # All linear layers: always wrap
        if scope == "all_linear":
            return True
        # Only head
        if scope == "head_only":
            return full_name == "head" or full_name.startswith("head.")
        # Only encoder (all its linear submodules)
        if scope == "encoder_only":
            return full_name.startswith("encoder.")
        # Only FFN MLP inside encoder layers (linear1, linear2)
        if scope == "ffn_only":
            return ("encoder.layers" in full_name) and (full_name.endswith(".linear1") or full_name.endswith(".linear2"))
        # Only input adapter linears
        if scope == "adapter_only":
            return full_name.startswith("encoder.input_adapter.")
        return False

    def _inject_lora_adapters(self, reset: bool = False) -> List[str]:
        """
        Replace target nn.Linear modules with LoRAInjectedLinear wrappers.

        If already active and not reset, reuses existing injection.
        Returns the list of wrapped module names (relative to model_ root).

        Args:
            reset: Whether to reinject and reinitialize adapters.

        Returns:
            List of wrapped module names.
        """
        assert self.model_ is not None
        if self._lora_active and not reset:
            return list(self._lora_wrapped_)

        # Identify targets first to avoid mutating while iterating
        targets: List[str] = []
        for full_name, module in self.model_.named_modules():
            if isinstance(module, nn.Linear):
                if isinstance(module, LoRAInjectedLinear):
                    continue
                if self._should_wrap_linear_with_lora(full_name):
                    targets.append(full_name)
        # Perform replacement
        wrapped: List[str] = []
        for name in targets:
            parent, attr = self._get_parent_and_attr(self.model_, name)
            orig = getattr(parent, attr)
            if isinstance(orig, LoRAInjectedLinear):
                continue
            if not isinstance(orig, nn.Linear):
                continue
            setattr(parent, attr, LoRAInjectedLinear(orig, r=self.lora_r, alpha=self.lora_alpha, dropout=self.lora_dropout))
            wrapped.append(name)

        self._lora_active = True
        self._lora_wrapped_ = wrapped
        return wrapped

    @torch.no_grad()
    def predict_proba(self, graphs: Sequence[nx.Graph]) -> np.ndarray:
        if self.mode != "classification":
            raise AttributeError("predict_proba is only available in classification mode.")
        self._ensure_model(graphs)
        assert self.model_ is not None
        self.model_.eval()

        tensors = self._vectorize_graphs(graphs)
        probs: List[np.ndarray] = []
        for xb, mask, _ in self._iter_minibatches_from_tensors(tensors, y=None, shuffle=False):
            _, _, logits = self.model_(xb, key_padding_mask=mask)
            p = F.softmax(logits, dim=-1).cpu().numpy()
            probs.append(p)
        n_classes = int(self.num_classes) if self.num_classes is not None else 0
        return np.vstack(probs) if probs else np.zeros((0, n_classes), dtype=np.float32)

    @torch.no_grad()
    def predict(self, graphs: Sequence[nx.Graph]) -> np.ndarray:
        if self.mode == "classification":
            proba = self.predict_proba(graphs)
            return proba.argmax(axis=1)
        else:
            self._ensure_model(graphs)
            assert self.model_ is not None
            self.model_.eval()
            tensors = self._vectorize_graphs(graphs)
            outs: List[np.ndarray] = []
            for xb, mask, _ in self._iter_minibatches_from_tensors(tensors, y=None, shuffle=False):
                _, _, pred = self.model_(xb, key_padding_mask=mask)
                outs.append(pred.cpu().numpy())
            return np.vstack(outs) if outs else np.zeros((0, self.output_dim), dtype=np.float32)

    @torch.no_grad()
    def transform(self, graphs: Sequence[nx.Graph]) -> np.ndarray:
        """Return graph-level embeddings after the encoder (pre-classifier).

        Args:
            graphs: Sequence of input graphs.

        Returns:
            Array of graph embeddings with shape [n_samples, d_model].
        """
        self._ensure_model(graphs)
        assert self.model_ is not None
        self.model_.eval()

        tensors = self._vectorize_graphs(graphs)
        embs: List[np.ndarray] = []
        for xb, mask, _ in self._iter_minibatches_from_tensors(tensors, y=None, shuffle=False):
            _, g, _ = self.model_(xb, key_padding_mask=mask)
            embs.append(g.cpu().numpy())
        return np.vstack(embs) if embs else np.zeros((0, self.d_model), dtype=np.float32)

    @torch.no_grad()
    def node_transform(self, graphs: Sequence[nx.Graph]) -> List[np.ndarray]:
        """Return per-graph lists of per-node embeddings after the encoder.

        If CLS is enabled, the first row corresponds to the CLS token; callers
        can drop it if they want strictly node-aligned outputs.

        Args:
            graphs: Sequence of input graphs.

        Returns:
            List of per-graph node embedding arrays.
        """
        self._ensure_model(graphs)
        assert self.model_ is not None
        self.model_.eval()

        tensors = self._vectorize_graphs(graphs)
        outputs: List[np.ndarray] = []
        for xb, mask, _ in self._iter_minibatches_from_tensors(tensors, y=None, shuffle=False):
            n, _, _ = self.model_(xb, key_padding_mask=mask)
            # Remove padding rows; need original lengths per item in batch
            # mask: [B, Nmax] True=PAD; we slice per sample
            for i in range(mask.size(0)):
                ni = n[i]
                mi = mask[i]
                # If CLS used, keep it; else, trim to valid tokens
                if self.model_.encoder.use_cls_token:
                    # CLS present at index 0; keep it and non-pad tokens
                    # Determine N with padding: mi.size(0) equals tokens without CLS
                    valid_len = int((~mi).sum().item())
                    outputs.append(ni[: (valid_len + 1)].cpu().numpy())
                else:
                    valid_len = int((~mi).sum().item())
                    outputs.append(ni[:valid_len].cpu().numpy())
        return outputs

    def plot(self, graphs: Sequence[nx.Graph], scatter_kwargs: Optional[dict] = None, ax=None, viewport_to_quantile: Optional[float] = None):
        """
        Scatter plot of transformed features (graph embeddings).

        Args:
            X: Input graphs.
            scatter_kwargs: Forwarded to matplotlib's scatter.
            ax: Optional matplotlib Axes; if None, creates and shows a new figure.
            viewport_to_quantile: If set (0<q<1), set axis limits to the
                central quantile range (q) of the projected points.

        Returns:
            Matplotlib Axes with the plotted embeddings.
        """
        from matplotlib import pyplot as plt  # local import to avoid hard dependency at import time

        features = self.transform(graphs)
        if features.shape[1] < 2:
            raise ValueError("plot requires at least 2 feature dimensions.")

        features = np.asarray(features)
        scatter_kwargs = scatter_kwargs or {}
        if ax is None:
            fig, ax = plt.subplots()
            show = True
        else:
            show = False
            fig = ax.figure

        ax.scatter(features[:, 0], features[:, 1], **scatter_kwargs)
        ax.set_xlabel("feature_0")
        ax.set_ylabel("feature_1")

        if viewport_to_quantile is not None:
            q = float(viewport_to_quantile)
            if not 0 < q < 1:
                raise ValueError("viewport_to_quantile must be in (0, 1).")
            q_low = (1 - q) / 2
            q_high = 1 - q_low
            x_low, x_high = np.quantile(features[:, 0], [q_low, q_high])
            y_low, y_high = np.quantile(features[:, 1], [q_low, q_high])
            ax.set_xlim(x_low, x_high)
            ax.set_ylim(y_low, y_high)

        if show:
            fig.tight_layout()
            plt.show()

        return ax

    # Optional sklearn compatibility helpers
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "node_vectorizer": self.node_vectorizer,
            "mode": self.mode,
            "num_classes": self.num_classes,
            "output_dim": self.output_dim,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "activation": self.activation,
            "pooling": self.pooling,
            "adapter_type": self.adapter_type,
            "adapter": self.adapter,
            "adapter_bottleneck": self.adapter_bottleneck,
            "adapter_bias": self.adapter_bias,
            "adapter_init": self.adapter_init,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_scope": self.lora_scope,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "class_weights": self.class_weights,
            "val_split": self.val_split,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_metric": self.early_stopping_metric,
            "early_stopping_mode": self.early_stopping_mode,
            "verbose": self.verbose,
            "device": str(self.device),
            "aux_whitening_weight": self.aux_whitening_weight,
            "aux_sparsity_w2_weight": self.aux_sparsity_w2_weight,
        }

    def set_params(self, **params) -> "NeuralGraphEstimator":
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown parameter {k}")
            setattr(self, k, v)
        # Changing params invalidates the compiled model
        self.model_ = None
        self.input_dim_ = None
        return self

    def reset(self, reseed: Optional[int] = None) -> "NeuralGraphEstimator":
        """
        Reinitialize model weights while preserving hyperparameters.

        Discards the compiled model and any active LoRA injections so that the
        next call to `fit`, `pre_train`, or `fine_tune` builds a fresh model.
        Optionally reseeds RNGs for deterministic initialization.

        Args:
            reseed: Optional RNG seed to set for both PyTorch and NumPy. If
                None, keeps the current RNG state (non-deterministic init).

        Returns:
            Self, with weights cleared and ready for a clean re-fit.
        """
        # Drop compiled model to force a fresh build on next train call
        self.model_ = None
        # Clear cached input dim; will be re-inferred as needed
        self.input_dim_ = None
        # Reset LoRA bookkeeping
        self._lora_active = False
        self._lora_wrapped_ = []
        # Clear any training histories
        self.history_ = {}
        if hasattr(self, "pretrain_history_"):
            self.pretrain_history_ = {"loss": [], "val_loss": []}
        if hasattr(self, "finetune_history_"):
            self.finetune_history_ = {"loss": [], "val_loss": []}
        # Optional reseed
        if reseed is not None:
            torch.manual_seed(int(reseed))
            np.random.seed(int(reseed))
        return self

    # --------------
    # Fine-tuning with LoRA
    # --------------
    def fine_tune(
        self,
        graphs: Sequence[nx.Graph],
        targets: Sequence[Any],
        *,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        weight_decay: Optional[float] = None,
        val_split: Optional[float] = None,
        early_stopping_patience: Optional[int] = None,
        early_stopping_metric: Optional[Literal["val_loss", "val_acc", "val_mse"]] = None,
        early_stopping_mode: Optional[Literal["min", "max"]] = None,
        reset_lora: bool = False,
    ) -> "NeuralGraphEstimator":
        """
        Fine-tune the current model on a new dataset using LoRA adapters.

        Freezes all existing parameters, injects trainable low-rank adapters
        into selected Linear layers (per `lora_scope`), and optimizes only
        the LoRA parameters (and any newly introduced head if applicable).

        Args:
            X: sequence of graphs.
            y: targets for the new task.
            epochs: optional override for fine-tune epochs (defaults to `self.epochs`).
            lr: optional override for fine-tune learning rate (defaults to `self.lr`).
            weight_decay: optional override (defaults to `self.weight_decay`).
            val_split: optional override (defaults to `self.val_split`).
            early_stopping_patience: optional override.
            early_stopping_metric: optional override; for regression allowed: "val_loss" or "val_mse".
            early_stopping_mode: optional override: "min" or "max".
            reset_lora: if True, reinjects LoRA even if already present (reinitializes adapters).

        Returns:
            Self.
        """
        # Sanity checks for classification
        self._resolve_num_classes(targets)
        # Note: We do not change the head's out_dim automatically here to preserve weights.
        # Ensure compatibility externally if adapting to a different number of classes.

        # Ensure model exists
        self._ensure_model(graphs)
        assert self.model_ is not None
        model = self.model_
        trainable_params = self._prepare_finetune_model(reset_lora=reset_lora)

        if self.verbose:
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            enc_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
            head_params = total_params - enc_params
            print(f"Model parameters: total={total_params:,} (encoder={enc_params:,}, head={head_params:,})")

        # Loss & optim
        criterion = self._build_criterion()

        ft_epochs = int(epochs) if epochs is not None else self.epochs
        ft_lr = float(lr) if lr is not None else self.lr
        ft_wd = float(weight_decay) if weight_decay is not None else self.weight_decay
        optimizer = torch.optim.AdamW(trainable_params, lr=ft_lr, weight_decay=ft_wd)

        # Vectorize and split
        vs = self.val_split if val_split is None else float(val_split)
        train_tensors, y_train, val_tensors, y_val = self._vectorize_and_split_with_valsplit(graphs, targets, vs)

        # Early stopping configs
        es_patience = early_stopping_patience if early_stopping_patience is not None else self.early_stopping_patience
        es_metric = early_stopping_metric if early_stopping_metric is not None else self.early_stopping_metric
        es_mode = early_stopping_mode if early_stopping_mode is not None else self.early_stopping_mode
        self._validate_early_stopping_config(es_metric, es_mode)

        # Fine-tune history (separate from base training)
        self.finetune_history_ = self._init_history_dict()

        model.train()
        best_state, _, best_metric = self._run_training_loop(
            model,
            train_tensors,
            y_train,
            val_tensors,
            y_val,
            criterion,
            optimizer,
            epochs=ft_epochs,
            early_stopping_metric=es_metric,
            early_stopping_mode=es_mode,
            early_stopping_patience=es_patience,
            history=self.finetune_history_,
            label="epoch",
            include_errors=True,
            use_aux_losses=False,
            restore_message=True,
        )

        # Load best validation state if tracked
        if best_state is not None:
            model.load_state_dict(best_state)

        return self
