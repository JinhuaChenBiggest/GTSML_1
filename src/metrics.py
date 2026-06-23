"""Evaluation metrics: accuracy and macro-F1."""

import torch
from sklearn.metrics import accuracy_score, f1_score


@torch.no_grad()
def evaluate(model, views, adjs, labels, index, device):
    """Predict on the given index (e.g. unlabeled set) and return acc / macro-F1.

    - GCN version (adjs not None): message passing needs all nodes, so run a
      full-graph forward then slice by index.
    - Linear version (adjs is None): forward the index subset directly.
    Predictions take argmax over the fused evidence evidence_a.
    """
    model.eval()
    if adjs is None:
        sub_views = [v[index] for v in views]
        _, evidence_a = model(sub_views)
        alpha_a = evidence_a + 1
    else:
        _, evidence_a = model(views, adjs)
        alpha_a = evidence_a[index] + 1
    pred = torch.argmax(alpha_a, dim=1).cpu().numpy()
    true = labels[index].cpu().numpy()

    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred, average="macro")
    return acc, f1
