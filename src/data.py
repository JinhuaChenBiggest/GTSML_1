"""Multi-view dataset loading and semi-supervised split.

Supports standard multi-view .mat files under datasets/ (e.g. Handwritten,
OutdoorScene, Cora, CiteSeer, ALOI):
    - X: object array of shape (n_views, 1); each element is an (N, d_v) view.
    - y: (N, 1) labels (original labels may start from 1).
"""

import os

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler


def _to_dense(mat):
    if hasattr(mat, "toarray"):
        return mat.toarray()
    return np.asarray(mat)


def _normalize_adj(adj):
    """Symmetric normalization: A_hat = D^{-1/2} (A + I) D^{-1/2}."""
    adj = adj + sp.eye(adj.shape[0])
    deg = np.asarray(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_inv_sqrt[deg == 0] = 0.0
    d_mat = sp.diags(d_inv_sqrt)
    return (d_mat @ adj @ d_mat).tocoo()


def _scipy_to_torch_sparse(coo):
    coo = coo.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
    values = torch.from_numpy(coo.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size(coo.shape)).coalesce()


def build_adj_from_matrix(mat):
    """Normalize a given adjacency matrix into a torch sparse adjacency.

    Used for native-graph datasets (e.g. the citation adjacency in view 0 of
    Cora / CiteSeer).
    """
    if hasattr(mat, "tocoo"):
        adj = mat.tocsr()
    else:
        adj = sp.csr_matrix(np.asarray(mat, dtype=np.float32))
    adj = adj.maximum(adj.T)
    adj_norm = _normalize_adj(adj)
    return _scipy_to_torch_sparse(adj_norm)


def build_knn_adj(features, k=10, metric="cosine"):
    """Build a symmetric kNN graph from features and return a normalized
    torch sparse adjacency.

    Args:
        features: (N, d) ndarray.
        k: number of neighbors per node.
        metric: similarity metric ('cosine' suits high-dimensional features).
    """
    n = features.shape[0]
    k = min(k, n - 1)
    adj = kneighbors_graph(
        features, n_neighbors=k, mode="connectivity",
        metric=metric, include_self=False,
    )
    adj = adj.maximum(adj.T)
    adj_norm = _normalize_adj(adj)
    return _scipy_to_torch_sparse(adj_norm)


class MultiViewDataset:
    """Load a single multi-view .mat dataset.

    Attributes:
        views (list[Tensor]): standardized feature tensor per view, (N, d_v).
        adjs (list[Tensor] | None): normalized adjacency per view, or None.
        labels (Tensor): (N,) zero-based labels.
        num_views (int): number of feature views.
        dims (list[int]): feature dimension of each view.
        num_classes (int): number of classes.
    """

    def __init__(self, mat_path, normalize=True, build_graph=True,
                 knn_k=10, knn_metric="cosine", graph_view=None):
        if not os.path.exists(mat_path):
            raise FileNotFoundError(mat_path)
        data = sio.loadmat(mat_path)
        if "X" not in data or "y" not in data:
            raise KeyError(
                f"{mat_path} is not a standard multi-view file (needs 'X' and 'y')."
            )

        X = data["X"]
        y = np.asarray(data["y"]).ravel().astype(np.int64)

        classes = np.unique(y)
        remap = {c: i for i, c in enumerate(classes)}
        y = np.array([remap[c] for c in y], dtype=np.int64)

        n_views = X.shape[0]

        # Native graph: use the specified view as the adjacency graph, drop it
        # from the feature views, and let the remaining views share this graph.
        native_adj = None
        if graph_view is not None:
            if not (0 <= graph_view < n_views):
                raise ValueError(f"graph_view={graph_view} out of range [0,{n_views})")
            native_adj = build_adj_from_matrix(X[graph_view, 0])

        views = []
        dims = []
        adjs = []
        for v in range(n_views):
            if graph_view is not None and v == graph_view:
                continue
            xv = _to_dense(X[v, 0]).astype(np.float32)
            if normalize:
                xv = StandardScaler().fit_transform(xv).astype(np.float32)
            views.append(torch.from_numpy(xv))
            dims.append(xv.shape[1])
            if build_graph:
                if native_adj is not None:
                    adjs.append(native_adj)
                else:
                    adjs.append(build_knn_adj(xv, k=knn_k, metric=knn_metric))

        self.views = views
        self.adjs = adjs if build_graph else None
        self.num_views = len(views)
        self.labels = torch.from_numpy(y)
        self.dims = dims
        self.num_classes = int(len(classes))
        self.num_samples = int(y.shape[0])

    def to(self, device):
        self.views = [v.to(device) for v in self.views]
        if self.adjs is not None:
            self.adjs = [a.to(device) for a in self.adjs]
        self.labels = self.labels.to(device)
        return self


def semi_supervised_split(num_samples, labels, label_ratio, seed=0):
    """Stratified semi-supervised split by ratio.

    Args:
        num_samples: total number of samples.
        labels: (N,) label tensor used for stratified sampling.
        label_ratio: fraction of labeled samples over the whole dataset
            (e.g. 0.1 means 10%).
        seed: random seed.

    Returns:
        (labeled_idx, unlabeled_idx): two LongTensor index tensors.
    """
    rng = np.random.RandomState(seed)
    y = labels.cpu().numpy()
    labeled_mask = np.zeros(num_samples, dtype=bool)

    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        n_lab = max(1, int(round(len(idx_c) * label_ratio)))
        labeled_mask[idx_c[:n_lab]] = True

    labeled_idx = np.where(labeled_mask)[0]
    unlabeled_idx = np.where(~labeled_mask)[0]
    return (
        torch.from_numpy(labeled_idx).long(),
        torch.from_numpy(unlabeled_idx).long(),
    )
