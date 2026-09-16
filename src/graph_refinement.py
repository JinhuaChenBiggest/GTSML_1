"""Sparse hidden-representation graph refinement for P5-A/P5-B."""

import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph

from src.data import _normalize_adj, _scipy_to_torch_sparse


def rank_normalize_confidence(node_confidence, confidence_floor=0.2):
    """Map within-view confidence ranks to [confidence_floor, 1]."""
    if not 0.0 <= confidence_floor <= 1.0:
        raise ValueError("confidence_floor must be in [0, 1]")
    confidence = node_confidence.detach().float().reshape(-1)
    if confidence.numel() == 0:
        raise ValueError("node_confidence cannot be empty")
    if not torch.isfinite(confidence).all():
        raise ValueError("node_confidence must contain only finite values")
    if confidence.numel() == 1:
        return torch.ones_like(confidence)

    order = torch.argsort(confidence, stable=True)
    ranks = torch.empty_like(confidence)
    ranks[order] = torch.linspace(
        0.0, 1.0, confidence.numel(), device=confidence.device
    )
    return confidence_floor + (1.0 - confidence_floor) * ranks


def apply_node_confidence_gate(adj, node_confidence, confidence_floor=0.2):
    """Scale sparse edges by the geometric mean of endpoint confidence."""
    if not 0.0 <= confidence_floor <= 1.0:
        raise ValueError("confidence_floor must be in [0, 1]")
    if torch.is_tensor(node_confidence):
        confidence = node_confidence.detach().float().cpu().numpy()
    else:
        confidence = np.asarray(node_confidence, dtype=np.float32)
    confidence = confidence.reshape(-1)
    if confidence.shape[0] != adj.shape[0]:
        raise ValueError("node_confidence must contain one value per node")
    if not np.isfinite(confidence).all():
        raise ValueError("node_confidence must contain only finite values")

    confidence = np.clip(confidence, confidence_floor, 1.0)
    gated = adj.tocoo(copy=True)
    gated.data *= np.sqrt(confidence[gated.row] * confidence[gated.col])
    return gated.tocsr()


def build_semantic_knn_adj(
    embedding,
    k=10,
    metric="cosine",
    node_confidence=None,
    confidence_floor=0.2,
):
    """Build a weighted, symmetric, normalized kNN graph from embeddings."""
    if embedding.ndim != 2:
        raise ValueError("embedding must have shape (samples, dimensions)")
    num_samples = embedding.shape[0]
    if num_samples < 2:
        raise ValueError("at least two samples are required")
    if not 1 <= k < num_samples:
        raise ValueError(f"k must be in [1, {num_samples - 1}], got {k}")

    features = embedding.detach().float().cpu().numpy()
    distances = kneighbors_graph(
        features,
        n_neighbors=k,
        mode="distance",
        metric=metric,
        include_self=False,
    )
    similarities = distances.tocsr(copy=True)
    if metric == "cosine":
        similarities.data = np.clip(1.0 - similarities.data, 0.0, 1.0)
    else:
        positive_distances = similarities.data[similarities.data > 0]
        scale = np.median(positive_distances) if positive_distances.size else 1.0
        similarities.data = np.exp(-similarities.data / max(scale, 1e-12))
    similarities.eliminate_zeros()
    if node_confidence is not None:
        similarities = apply_node_confidence_gate(
            similarities, node_confidence, confidence_floor=confidence_floor
        )
    similarities = similarities.maximum(similarities.T)
    return _scipy_to_torch_sparse(_normalize_adj(similarities))


def blend_sparse_adjs(base_adj, semantic_adj, base_weight=0.8):
    """Convexly combine two normalized sparse propagation operators."""
    if not 0.0 <= base_weight <= 1.0:
        raise ValueError("base_weight must be in [0, 1]")
    if not base_adj.is_sparse or not semantic_adj.is_sparse:
        raise TypeError("graph refinement expects sparse adjacency tensors")
    if base_adj.shape != semantic_adj.shape:
        raise ValueError("base and semantic graphs must have identical shapes")

    base_adj = base_adj.coalesce()
    semantic_adj = semantic_adj.coalesce().to(
        device=base_adj.device, dtype=base_adj.dtype
    )
    if base_weight == 1.0:
        return base_adj
    if base_weight == 0.0:
        return semantic_adj

    indices = torch.cat((base_adj.indices(), semantic_adj.indices()), dim=1)
    values = torch.cat(
        (
            base_adj.values() * base_weight,
            semantic_adj.values() * (1.0 - base_weight),
        )
    )
    return torch.sparse_coo_tensor(
        indices, values, base_adj.shape, device=base_adj.device
    ).coalesce()


def refine_graphs(
    embeddings,
    base_adjs,
    k=10,
    metric="cosine",
    base_weight=0.8,
    node_confidences=None,
    confidence_floor=0.2,
):
    """Build and blend one semantic graph per view."""
    if len(embeddings) != len(base_adjs):
        raise ValueError("embeddings and base_adjs must contain the same views")
    if node_confidences is not None and len(node_confidences) != len(embeddings):
        raise ValueError("node_confidences must contain one tensor per view")

    refined = []
    stats = []
    for view_index, (embedding, base_adj) in enumerate(zip(embeddings, base_adjs)):
        confidence = (
            None if node_confidences is None else node_confidences[view_index]
        )
        semantic = build_semantic_knn_adj(
            embedding,
            k=k,
            metric=metric,
            node_confidence=confidence,
            confidence_floor=confidence_floor,
        )
        mixed = blend_sparse_adjs(base_adj, semantic, base_weight=base_weight)
        refined.append(mixed)
        stats.append(
            {
                "view": view_index,
                "base_nnz": int(base_adj._nnz()),
                "semantic_nnz": int(semantic._nnz()),
                "refined_nnz": int(mixed._nnz()),
                "mean_confidence": (
                    None if confidence is None else float(confidence.mean().item())
                ),
            }
        )
    return refined, stats
