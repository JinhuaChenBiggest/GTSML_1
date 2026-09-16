"""Multi-view semi-supervised EDL classification: train and evaluate.

Setup:
    - the dataset is split into a labeled and an unlabeled part only;
    - the EDL supervised loss is applied to the labeled part only;
    - evaluation predicts directly on the unlabeled data and reports Accuracy
      and macro-F1.

Example:
    python train.py --dataset Handwritten --label_ratio 0.1
"""

import argparse
import os
import time

import numpy as np
import torch

from src.data import MultiViewDataset, semi_supervised_split
from src.losses import get_loss
from src.metrics import evaluate
from src.model import EDLMultiView, EDLMultiViewGCN
from src.graph_refinement import rank_normalize_confidence, refine_graphs


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    p = argparse.ArgumentParser(description="Multi-view semi-supervised EDL classification")
    p.add_argument("--data_dir", type=str, default="datasets")
    p.add_argument("--dataset", type=str, default="Handwritten")
    p.add_argument("--model", type=str, default="gcn", choices=["linear", "gcn"],
                   help="linear=supervised linear (pre-GCN), gcn=GCN semi-supervised")
    p.add_argument("--label_ratio", type=float, default=0.1,
                   help="fraction of labeled samples over the whole dataset")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--annealing_step", type=int, default=50)
    p.add_argument("--gamma", type=float, default=1.0, help="reserved, currently unused")
    p.add_argument("--fusion", type=str, default="ds", choices=["ds", "sum", "mean"])
    p.add_argument("--hidden", type=int, default=64, help="GCN hidden dimension")
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--knn_k", type=int, default=10, help="neighbors per node for kNN graph")
    p.add_argument("--knn_metric", type=str, default="cosine")
    p.add_argument("--graph_view", type=int, default=-1,
                   help="view index used as the native adjacency graph "
                        "(e.g. 0 for Cora/CiteSeer); -1 means build a kNN graph")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=20)
    p.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    p.add_argument("--dynamic_graph", action="store_true",
                   help="periodically refine graphs from hidden representations")
    p.add_argument("--graph_warmup", type=int, default=50)
    p.add_argument("--graph_update_every", type=int, default=20)
    p.add_argument("--dynamic_knn_k", type=int, default=10)
    p.add_argument("--base_graph_weight", type=float, default=0.8)
    p.add_argument("--uncertainty_edge_gate", action="store_true",
                   help="gate new semantic edges with per-view EDL confidence")
    p.add_argument("--uncertainty_floor", type=float, default=0.2)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This project requires a CUDA-capable GPU, but none is available. "
            "Install a CUDA build of PyTorch and run on a machine with a GPU."
        )
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.cuda.manual_seed_all(args.seed)

    use_graph = args.model == "gcn"
    if args.uncertainty_edge_gate and not args.dynamic_graph:
        raise ValueError("--uncertainty_edge_gate requires --dynamic_graph")
    if args.dynamic_graph:
        if not use_graph:
            raise ValueError("--dynamic_graph requires --model gcn")
        if args.graph_warmup < 0:
            raise ValueError("graph_warmup must be non-negative")
        if args.graph_update_every <= 0:
            raise ValueError("graph_update_every must be positive")
        if args.dynamic_knn_k <= 0:
            raise ValueError("dynamic_knn_k must be positive")
        if not 0.0 <= args.base_graph_weight <= 1.0:
            raise ValueError("base_graph_weight must be in [0, 1]")
        if not 0.0 <= args.uncertainty_floor <= 1.0:
            raise ValueError("uncertainty_floor must be in [0, 1]")
    graph_view = args.graph_view if args.graph_view >= 0 else None
    mat_path = os.path.join(args.data_dir, args.dataset + ".mat")
    ds = MultiViewDataset(
        mat_path, normalize=True, build_graph=use_graph,
        knn_k=args.knn_k, knn_metric=args.knn_metric, graph_view=graph_view,
    ).to(device)
    graph_desc = (f" | native_graph=view{graph_view}" if (use_graph and graph_view is not None)
                  else (f" | knn_k={args.knn_k}" if use_graph else ""))
    print(f"[dataset] {args.dataset} | model={args.model} | samples={ds.num_samples} "
          f"| views={ds.num_views} | dims={ds.dims} | classes={ds.num_classes}{graph_desc}")
    if args.dynamic_graph:
        print(f"[dynamic-graph] warmup={args.graph_warmup} "
              f"| update_every={args.graph_update_every} | k={args.dynamic_knn_k} "
              f"| base_weight={args.base_graph_weight:.2f} "
              f"| uncertainty_gate={args.uncertainty_edge_gate}")

    labeled_idx, unlabeled_idx = semi_supervised_split(
        ds.num_samples, ds.labels, args.label_ratio
    )
    labeled_idx = labeled_idx.to(device)
    unlabeled_idx = unlabeled_idx.to(device)
    print(f"[split] labeled={len(labeled_idx)} "
          f"({len(labeled_idx) / ds.num_samples:.1%}) | unlabeled={len(unlabeled_idx)}")

    if use_graph:
        model = EDLMultiViewGCN(
            dims=ds.dims, num_classes=ds.num_classes,
            hidden=args.hidden, dropout=args.dropout, fusion=args.fusion,
        ).to(device)
    else:
        model = EDLMultiView(
            dims=ds.dims, num_classes=ds.num_classes, fusion=args.fusion,
        ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    base_adjs = ds.adjs
    current_adjs = list(ds.adjs) if use_graph else None

    lab_labels = ds.labels[labeled_idx]
    lab_views = [v[labeled_idx] for v in ds.views]

    best_acc, best_f1 = 0.0, 0.0
    acc, f1 = 0.0, 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        if use_graph:
            evidences, evidence_a = model(ds.views, current_adjs)
            lab_evidences = [e[labeled_idx] for e in evidences]
            lab_evidence_a = evidence_a[labeled_idx]
        else:
            lab_evidences, lab_evidence_a = model(lab_views)
        loss = get_loss(
            lab_evidences, lab_evidence_a, lab_labels,
            epoch_num=epoch, num_classes=ds.num_classes,
            annealing_step=args.annealing_step, gamma=args.gamma, device=device,
        )
        loss.backward()
        optimizer.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            acc, f1 = evaluate(
                model, ds.views, current_adjs, ds.labels, unlabeled_idx, device
            )
            if acc > best_acc:
                best_acc, best_f1 = acc, f1
            print(f"epoch {epoch:4d} | loss={loss.item():.4f} | "
                  f"unlabeled Acc={acc:.4f} F1={f1:.4f}")

        should_update_graph = (
            args.dynamic_graph
            and epoch >= args.graph_warmup
            and epoch < args.epochs
            and (epoch - args.graph_warmup) % args.graph_update_every == 0
        )
        if should_update_graph:
            started = time.perf_counter()
            model.eval()
            with torch.no_grad():
                embeddings = model.encode(ds.views, current_adjs)
                node_confidences = None
                raw_confidences = None
                if args.uncertainty_edge_gate:
                    view_evidences, _ = model(ds.views, current_adjs)
                    raw_confidences = [
                        (
                            1.0
                            - ds.num_classes
                            / (evidence + 1).sum(dim=1)
                        ).clamp(0.0, 1.0)
                        for evidence in view_evidences
                    ]
                    node_confidences = [
                        rank_normalize_confidence(
                            confidence, confidence_floor=args.uncertainty_floor
                        )
                        for confidence in raw_confidences
                    ]
            current_adjs, graph_stats = refine_graphs(
                embeddings,
                base_adjs,
                k=args.dynamic_knn_k,
                metric=args.knn_metric,
                base_weight=args.base_graph_weight,
                node_confidences=node_confidences,
                confidence_floor=args.uncertainty_floor,
            )
            elapsed = time.perf_counter() - started
            nnz = ",".join(str(item["refined_nnz"]) for item in graph_stats)
            confidence_desc = ""
            if node_confidences is not None:
                raw_means = ",".join(
                    f"{confidence.mean().item():.3f}"
                    for confidence in raw_confidences
                )
                gate_means = ",".join(
                    f'{item["mean_confidence"]:.3f}' for item in graph_stats
                )
                confidence_desc = (
                    f" | raw_confidence=[{raw_means}] "
                    f"| gate_confidence=[{gate_means}]"
                )
            print(f"[graph-update] epoch={epoch} | seconds={elapsed:.2f} "
                  f"| refined_nnz=[{nnz}]{confidence_desc}")

    print("=" * 60)
    print(f"[final] unlabeled Acc={acc:.4f} F1={f1:.4f}")
    print(f"[best]  unlabeled Acc={best_acc:.4f} F1={best_f1:.4f}")


if __name__ == "__main__":
    main()
