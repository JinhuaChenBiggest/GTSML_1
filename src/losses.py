"""EDL evidential cross-entropy loss (KL annealing + digamma form).

Inputs are the model's evidence outputs:
    - evidences: list of per-view evidences (layer output after softplus)
    - evidence_a: fused evidence
"""

import torch
import torch.nn.functional as F


def kl_divergence(alpha, num_classes, device):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device, useKL=True):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    if not useKL:
        return A

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_digamma_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    loss = edl_loss(
        torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device
    )
    return torch.mean(loss)


def get_loss(evidences, evidence_a, target, epoch_num, num_classes, annealing_step, gamma, device):
    """Full supervised loss (applied to labeled samples only).

    Args:
        evidences: list[(B, K)] per-view evidence.
        evidence_a: (B, K) fused evidence.
        target: (B,) integer labels.
        gamma: reserved weight (currently unused, kept for signature compat).
    """
    target = F.one_hot(target, num_classes)
    alpha_a = evidence_a + 1
    loss_acc = edl_digamma_loss(alpha_a, target, epoch_num, num_classes, annealing_step, device)
    for v in range(len(evidences)):
        alpha = evidences[v] + 1
        loss_acc += edl_digamma_loss(alpha, target, epoch_num, num_classes, annealing_step, device)
    loss_acc = loss_acc / (len(evidences) + 1)
    loss = loss_acc
    return loss
