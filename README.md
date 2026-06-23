# GTSML

Multi-view semi-supervised image classification based on the Evidential Deep
Learning (EDL/ENN) framework, with a Graph Convolutional Network (GCN) backbone
to exploit unlabeled data.

## Overview

- **Evidential head**: each view produces non-negative evidence (via `softplus`);
  `alpha = evidence + 1` parameterizes a Dirichlet distribution.
- **Multi-view fusion**: per-view evidences are fused by the Dempster-Shafer
  combination rule into a fused evidence `evidence_a`.
- **Semi-supervised learning**: the dataset is split into a labeled and an
  unlabeled part. The EDL supervised loss is applied to labeled nodes only.
  A two-layer GCN propagates features over the full graph (including unlabeled
  nodes), so unlabeled data shapes the learned representations through message
  passing. Evaluation predicts directly on the unlabeled set.
- **Loss**: EDL evidential cross-entropy in the digamma form with KL annealing.

Two backbones are provided:

- `linear`: a supervised linear evidential classifier (pre-GCN baseline).
- `gcn`: a GCN-based semi-supervised evidential classifier.

## Project structure

```
GTSML/
├── datasets/            # multi-view .mat datasets (X: views, y: labels)
├── src/
│   ├── data.py          # .mat loading, normalization, graph building, split
│   ├── model.py         # GraphConv, DS fusion, linear and GCN models
│   ├── losses.py        # KL divergence, EDL digamma loss, total loss
│   └── metrics.py       # accuracy and macro-F1
├── train.py             # training and evaluation entry point
├── requirements.txt
└── README.md
```

## Data format

Standard multi-view `.mat` files placed under `datasets/`:

- `X`: object array of shape `(n_views, 1)`; each element is an `(N, d_v)` view.
- `y`: `(N, 1)` labels (labels starting from 1 are remapped to 0-based).

Supported datasets in this repo: `Handwritten`, `OutdoorScene`, `Cora`,
`CiteSeer`, `ALOI`. For native-graph datasets (`Cora`, `CiteSeer`) view 0 is the
citation adjacency and can be used directly as the graph via `--graph_view 0`.

## Installation

```bash
pip install -r requirements.txt
```

PyTorch (CPU build) can be installed from the official index if needed:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Usage

```bash
# GCN semi-supervised, kNN graph
python train.py --dataset Handwritten --model gcn --label_ratio 0.1

# GCN semi-supervised, native citation graph (view 0)
python train.py --dataset Cora     --model gcn --label_ratio 0.1 --graph_view 0
python train.py --dataset CiteSeer --model gcn --label_ratio 0.1 --graph_view 0

# Linear supervised baseline (pre-GCN)
python train.py --dataset Handwritten --model linear --label_ratio 0.1
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `Handwritten` | dataset name (file `datasets/<name>.mat`) |
| `--model` | `gcn` | `linear` or `gcn` |
| `--label_ratio` | `0.1` | labeled fraction over the whole dataset |
| `--graph_view` | `-1` | view index used as native graph; `-1` builds a kNN graph |
| `--knn_k` | `10` | neighbors per node for kNN graph |
| `--hidden` | `64` | GCN hidden dimension |
| `--dropout` | `0.5` | GCN dropout |
| `--epochs` | `300` | training epochs |
| `--lr` | `3e-3` | learning rate |

## Results (10% labels, unlabeled set, best Acc / F1)

| Dataset | Linear (pre-GCN) | GCN (kNN graph) | GCN (native graph) |
|---|---|---|---|
| Handwritten | 0.9528 / 0.9529 | 0.9733 / 0.9733 | - |
| ALOI | 0.7370 / 0.7102 | 0.8479 / 0.8430 | - |
| OutdoorScene | 0.7991 / 0.7978 | 0.7859 / 0.7874 | - |
| Cora | 0.5998 / 0.5832 | 0.7393 / 0.7321 | 0.7841 / 0.7708 |
| CiteSeer | 0.5560 / 0.5243 | 0.5950 / 0.5376 | 0.6715 / 0.6170 |

Numbers are from CPU runs and may vary slightly across environments and seeds.
