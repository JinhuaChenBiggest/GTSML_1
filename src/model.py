"""Evidential deep learning (EDL/ENN) multi-view semi-supervised classifier.

Forward pass:
    - per view, a kNN (or native) graph plus a two-layer GCN performs message
      passing over the full graph (including unlabeled nodes);
    - the second GCN output is passed through softplus to form the view
      evidence evidence_v;
    - the multi-view evidences are fused by the Dempster-Shafer combination
      rule into evidence_a.

How unlabeled data is used: the GCN propagates features over all N nodes, so
unlabeled nodes participate in neighbor aggregation and gradients flow back
through them. The supervised loss is applied to labeled nodes only (masking).

Note: EDL requires non-negative evidence (alpha = evidence + 1 > 0), otherwise
lgamma/digamma produce NaN; hence softplus is applied to the layer output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    """Standard GCN layer: H' = A_hat (H W), A_hat is normalized adjacency."""

    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x, adj):
        support = self.linear(x)
        if adj.is_sparse:
            return torch.sparse.mm(adj, support)
        return adj @ support


def ds_combine_two(alpha1, alpha2, num_classes):
    """Dempster-Shafer combination of two Dirichlet opinions (TMC style).

    Inputs/outputs are Dirichlet parameters alpha = evidence + 1, shape (B, K).
    """
    alpha = {0: alpha1, 1: alpha2}
    b, S, E, u = {}, {}, {}, {}
    for v in range(2):
        S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
        E[v] = alpha[v] - 1
        b[v] = E[v] / S[v].expand(E[v].shape)
        u[v] = num_classes / S[v]

    bb = torch.bmm(b[0].view(-1, num_classes, 1), b[1].view(-1, 1, num_classes))
    bu = b[0] * u[1].expand(b[0].shape)
    ub = b[1] * u[0].expand(b[0].shape)

    bb_sum = torch.sum(bb, dim=(1, 2))
    bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
    C = bb_sum - bb_diag

    denom = (1 - C).view(-1, 1)
    b_a = (b[0] * b[1] + bu + ub) / denom.expand(b[0].shape)
    u_a = (u[0] * u[1]) / denom.expand(u[0].shape)

    S_a = num_classes / u_a
    e_a = b_a * S_a.expand(b_a.shape)
    alpha_a = e_a + 1
    return alpha_a


def ds_combine(alphas, num_classes):
    """Iteratively apply DS combination over a list of alphas."""
    alpha_a = alphas[0]
    for v in range(1, len(alphas)):
        alpha_a = ds_combine_two(alpha_a, alphas[v], num_classes)
    return alpha_a


def _fuse(evidences, fusion, num_views, num_classes):
    if fusion == "ds":
        alphas = [e + 1 for e in evidences]
        return ds_combine(alphas, num_classes) - 1
    if fusion == "sum":
        return sum(evidences)
    if fusion == "mean":
        return sum(evidences) / num_views
    raise ValueError(f"unknown fusion: {fusion}")


class EDLMultiView(nn.Module):
    """Supervised linear EDL classifier (the pre-GCN baseline): one linear
    layer per view produces evidence."""

    def __init__(self, dims, num_classes, fusion="ds"):
        super().__init__()
        self.num_views = len(dims)
        self.num_classes = num_classes
        self.fusion = fusion
        self.act = F.softplus
        self.classifiers = nn.ModuleList(
            [nn.Linear(d, num_classes) for d in dims]
        )

    def forward(self, views, adjs=None):
        evidences = [self.act(self.classifiers[v](views[v]))
                     for v in range(self.num_views)]
        evidence_a = _fuse(evidences, self.fusion, self.num_views, self.num_classes)
        return evidences, evidence_a


class EDLMultiViewGCN(nn.Module):
    """Multi-view semi-supervised EDL classifier: a two-layer GCN per view
    produces evidence, fused by the DS rule."""

    def __init__(self, dims, num_classes, hidden=64, dropout=0.5, fusion="ds"):
        super().__init__()
        self.num_views = len(dims)
        self.num_classes = num_classes
        self.fusion = fusion
        self.dropout = dropout
        self.act = F.softplus
        self.gc1 = nn.ModuleList([GraphConv(d, hidden) for d in dims])
        self.gc2 = nn.ModuleList(
            [GraphConv(hidden, num_classes) for _ in dims]
        )

    def encode(self, views, adjs):
        """Return deterministic first-layer embeddings for graph refinement."""
        return [
            F.relu(self.gc1[v](views[v], adjs[v]))
            for v in range(self.num_views)
        ]

    def forward(self, views, adjs):
        evidences = []
        for v in range(self.num_views):
            h = F.dropout(views[v], self.dropout, training=self.training)
            h = F.relu(self.gc1[v](h, adjs[v]))
            h = F.dropout(h, self.dropout, training=self.training)
            logits = self.gc2[v](h, adjs[v])
            evidences.append(self.act(logits))

        evidence_a = _fuse(evidences, self.fusion, self.num_views, self.num_classes)
        return evidences, evidence_a
