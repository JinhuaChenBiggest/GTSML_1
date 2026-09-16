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
- **Optional dynamic graph**: after a supervised warmup, sparse semantic
  kNN graphs are periodically rebuilt from first-layer GCN embeddings and
  blended with the immutable input graphs.
- **Optional uncertainty gate**: within-view ranks of EDL confidence
  scale newly created semantic edges without modifying the input graph.
- **Loss**: EDL evidential cross-entropy in the digamma form with KL annealing.

Two backbones are provided:

- `linear`: a supervised linear evidential classifier (pre-GCN baseline).
- `gcn`: a GCN-based semi-supervised evidential classifier.

## Project structure

```
GTSML/
├── datasets/            # multi-view .mat datasets (X: views, y: labels)
├── src/
│   ├── __init__.py
│   ├── data.py          # .mat loading, normalization, graph building, split
│   ├── graph_refinement.py # sparse embedding kNN graph construction and blend
│   ├── model.py         # GraphConv, DS fusion, linear and GCN models
│   ├── losses.py        # KL divergence, EDL digamma loss, total loss
│   └── metrics.py       # accuracy and macro-F1
├── train.py             # training and evaluation entry point
├── requirements.txt
├── LICENSE
└── README.md
```

## Data format

Standard multi-view `.mat` files placed under `datasets/`:

- `X`: object array of shape `(n_views, 1)`; each element is an `(N, d_v)` view.
- `y`: `(N, 1)` labels (labels starting from 1 are remapped to 0-based).

Supported datasets in this repo: `Handwritten`, `OutdoorScene`, `Cora`,
`CiteSeer`, `ALOI`. For native-graph datasets (`Cora`, `CiteSeer`) view 0 is the
citation adjacency and can be used directly as the graph via `--graph_view 0`.

## Requirements

This project is **GPU-only**: training and evaluation require a CUDA-capable
GPU. The program raises an error and exits if no GPU is available.

## Installation

```bash
pip install numpy scipy scikit-learn
# Install a CUDA build of PyTorch matching your CUDA driver, e.g. CUDA 12.4:
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Usage

```bash
# GCN semi-supervised, kNN graph
python train.py --dataset Handwritten --model gcn --label_ratio 0.1

# GCN semi-supervised, native citation graph (view 0)
python train.py --dataset Cora     --model gcn --label_ratio 0.1 --graph_view 0
python train.py --dataset CiteSeer --model gcn --label_ratio 0.1 --graph_view 0

# Hidden-representation dynamic graph
python train.py --dataset OutdoorScene --model gcn --dynamic_graph
python train.py --dataset Cora --model gcn --graph_view 0 --dynamic_graph

# Optional rank-normalized uncertainty edge gate
python train.py --dataset OutdoorScene --model gcn --dynamic_graph --uncertainty_edge_gate

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
| `--seed` | `0` | model/NumPy/PyTorch seed; the labeled split remains independently random |
| `--eval_every` | `20` | epochs between unlabeled-set evaluations |
| `--dynamic_graph` | off | enable periodic semantic graph refinement |
| `--graph_warmup` | `50` | fixed-graph epochs before the first update |
| `--graph_update_every` | `20` | epochs between graph updates |
| `--dynamic_knn_k` | `10` | neighbors in each hidden semantic graph |
| `--base_graph_weight` | `0.8` | immutable input-graph weight in the blend |
| `--uncertainty_edge_gate` | off | enable confidence gating on semantic edges |
| `--uncertainty_floor` | `0.2` | minimum rank-normalized node confidence |
| `--gpu` | `0` | CUDA device index |

## Results

The final dynamic-graph configuration was evaluated for 300 epochs with 10%
labeled samples. Results are the mean and sample standard deviation over five
independent random labeled/unlabeled splits. Cora and CiteSeer use their native
citation graph (`--graph_view 0`); the other datasets use cosine kNN graphs.

`Best Acc` is selected from evaluations every 20 epochs, and the paired F1 is
reported from the same epoch. Since this selection uses the unlabeled
evaluation set, the final-epoch results are also shown as a selection-free
reference. Fixed-graph values are the original three-run baselines and are
included only as an unpaired reference.

| Dataset | Fixed graph Acc / F1 | Dynamic graph best Acc / F1 | Dynamic graph final Acc / F1 |
|---|---|---|---|
| Handwritten | 0.9730 / 0.9730 | 0.9744+/-0.0035 / 0.9744+/-0.0035 | 0.9727+/-0.0033 / 0.9726+/-0.0034 |
| OutdoorScene | 0.7988 / 0.8012 | 0.7854+/-0.0110 / 0.7876+/-0.0107 | 0.7785+/-0.0129 / 0.7806+/-0.0124 |
| ALOI | 0.8481 / 0.8430 | 0.8520+/-0.0079 / 0.8480+/-0.0091 | 0.8520+/-0.0079 / 0.8480+/-0.0091 |
| Cora | 0.8047 / 0.7941 | 0.8166+/-0.0070 / 0.8029+/-0.0054 | 0.7805+/-0.0135 / 0.7627+/-0.0159 |
| CiteSeer | 0.6631 / 0.6265 | 0.6770+/-0.0084 / 0.6305+/-0.0108 | 0.6621+/-0.0099 / 0.6210+/-0.0087 |

The dynamic graph improves the best-epoch mean on Handwritten, ALOI, Cora, and
CiteSeer, but not on OutdoorScene. Final-epoch degradation is most pronounced
on Cora. Graph refinement is therefore optional rather than enabled by default.

## Method: I/O flow

A compact, paper-style walkthrough of how data flows through the model. Let
there be $N$ samples, $V$ views, and $K$ classes.

### 1. Input data

View $v$ provides a feature matrix $X^{(v)} \in \mathbb{R}^{N \times d_v}$, and
labels are $y \in \{0,\dots,K-1\}^{N}$. Each view is standardized per feature
(zero mean, unit variance). The index set is partitioned into a labeled subset
$\mathcal{L}$ and an unlabeled subset $\mathcal{U}$. For each class $c$, the
split samples $\max(1,\mathrm{round}(rN_c))$ labeled nodes, where $r$ is the
label ratio (e.g. $r=0.1$). The loss uses $\mathcal{L}$ only; evaluation is
performed on $\mathcal{U}$.

### 2. Graph construction (for non-graph datasets)

For each view a kNN graph is built from cosine similarity: every node is linked
to its $k$ nearest neighbors to form a binary adjacency $A^{(v)}$, which is then
symmetrized, given self-loops, and symmetrically normalized:

$$A \leftarrow \max(A, A^{\top}), \qquad \tilde{A} = A + I, \qquad \hat{A}^{(v)} = \tilde{D}^{-1/2}\,\tilde{A}\,\tilde{D}^{-1/2}, \qquad \tilde{D}_{ii} = \sum_{j}\tilde{A}_{ij}.$$

For native-graph datasets (Cora, CiteSeer), the citation adjacency stored in
view 0 is used directly (normalized the same way) and shared by the remaining
feature views.

### 3. Hidden-representation graph refinement

Dynamic refinement keeps each normalized input graph $\hat A_0^{(v)}$ as an
immutable anchor. After the warmup, deterministic first-layer embeddings are
extracted without dropout:

$$H_t^{(v)} = \mathrm{ReLU}\!\left(\hat A_t^{(v)}X^{(v)}W_1^{(v)}\right).$$

A sparse cosine kNN graph is built from each $H_t^{(v)}$. Its directed edges
are assigned non-negative cosine similarities, symmetrized, given self-loops,
and normalized into $\hat A_{\mathrm{sem},t}^{(v)}$. The propagation graph for
subsequent epochs is

$$\hat A_{t+1}^{(v)} = \beta\hat A_0^{(v)} + (1-\beta)\hat A_{\mathrm{sem},t}^{(v)}.$$

Graph construction runs outside autograd and retains only sparse top-k edges;
it never creates an $N\times N$ dense similarity tensor. Native citation graphs
remain the anchor for Cora/CiteSeer rather than being replaced. Setting
`base_graph_weight` to 1 recovers the fixed graph exactly.

### 4. Uncertainty edge gate

For each view, the optional edge gate computes raw node confidence from its own
evidence:

$$c_i^{(v)} = 1-u_i^{(v)}, \qquad u_i^{(v)}=\frac{K}{\sum_k\alpha_{ik}^{(v)}}.$$

Raw confidence is converted to an empirical within-view rank $r_i^{(v)}$ and
mapped into a bounded gate:

$$\tilde c_i^{(v)}=c_{\min}+(1-c_{\min})r_i^{(v)}.$$

This preserves uncertainty ordering while avoiding the class-count scale
problem observed with raw confidence on ALOI. Before semantic-graph
normalization, each new edge is scaled by its endpoint geometric mean:

$$w_{ij}^{(v)} \leftarrow w_{ij}^{(v)}
\sqrt{\tilde c_i^{(v)}\tilde c_j^{(v)}}.$$

Only semantic edges are gated; original kNN or citation edges remain unchanged.
The gate is recomputed outside autograd at every scheduled graph update.

### 5. Forward propagation

Per view, a two-layer GCN runs over the full graph (all $N$ nodes, including
unlabeled ones):

$$H^{(v)} = \mathrm{ReLU}\!\big(\hat{A}^{(v)} X^{(v)} W_1^{(v)}\big), \qquad Z^{(v)} = \hat{A}^{(v)} H^{(v)} W_2^{(v)}.$$

Non-negative evidence and the Dirichlet parameters follow:

$$e^{(v)} = \mathrm{softplus}(Z^{(v)}) \in \mathbb{R}_{\ge 0}^{N \times K}, \qquad \alpha^{(v)} = e^{(v)} + 1.$$

Each view yields a subjective opinion (belief masses and an uncertainty mass):

$$S^{(v)} = \sum_{k=1}^{K}\alpha^{(v)}_k, \qquad b^{(v)}_k = \frac{e^{(v)}_k}{S^{(v)}}, \qquad u^{(v)} = \frac{K}{S^{(v)}}, \qquad \sum_{k} b^{(v)}_k + u^{(v)} = 1.$$

### 6. Multi-view fusion (Dempster-Shafer)

Two opinions $(b^{1}, u^{1})$ and $(b^{2}, u^{2})$ are combined by

$$C = \sum_{i \ne j} b^{1}_i\, b^{2}_j, \qquad b_k = \frac{1}{1-C}\big(b^{1}_k b^{2}_k + b^{1}_k u^{2} + b^{2}_k u^{1}\big), \qquad u = \frac{1}{1-C}\, u^{1} u^{2},$$

where $C$ is the conflict coefficient. Applying this rule iteratively over all
views gives a fused opinion, from which the fused Dirichlet is recovered:

$$S = \frac{K}{u}, \qquad \hat{e}_k = b_k\, S, \qquad \alpha^{a} = \hat{e} + 1.$$

### 7. Loss (labeled nodes only)

For a node with one-hot label $\mathbf{y}$ and Dirichlet parameter $\alpha$, the
EDL digamma cross-entropy and its annealed KL regularizer are

$$\mathcal{L}_{\text{edl}}(\alpha, \mathbf{y}) = \sum_{k=1}^{K} y_k\big(\psi(S) - \psi(\alpha_k)\big), \qquad S = \sum_k \alpha_k,$$

$$\tilde{\alpha} = \mathbf{y} + (1 - \mathbf{y}) \odot \alpha$$

$$\mathcal{L}(\alpha, \mathbf{y}) = \mathcal{L}_{\text{edl}}(\alpha, \mathbf{y}) + \lambda_t \, \mathrm{KL}\!\big(\mathrm{Dir}(\tilde{\alpha}) \,\|\, \mathrm{Dir}(\mathbf{1})\big), \qquad \lambda_t = \min\!\big(1,\, t/T\big),$$

where $\psi$ is the digamma function, $t$ the current epoch and $T$ the
annealing step. The total objective averages the fused term and the $V$
per-view terms over the labeled set:

$$\mathcal{L}_{\text{total}} = \frac{1}{|\mathcal{L}|}\sum_{i \in \mathcal{L}} \frac{1}{V+1}\Big(\mathcal{L}(\alpha^{a}_i, \mathbf{y}_i) + \sum_{v=1}^{V} \mathcal{L}(\alpha^{(v)}_i, \mathbf{y}_i)\Big).$$

Even though the loss only touches labeled nodes, the full-graph GCN mixes
features from unlabeled nodes into labeled-node representations, so unlabeled
data shapes the parameters through message passing. This is the semi-supervised
learning signal.

### 8. Output and prediction

The model outputs the fused Dirichlet $\alpha^{a} \in \mathbb{R}^{N \times K}$.
The class probability is its mean and the predicted label is the argmax; the
mass $u$ provides a per-node uncertainty:

$$\hat{p}_k = \frac{\alpha^{a}_k}{S^{a}}, \qquad \hat{y} = \arg\max_{k}\, \alpha^{a}_k, \qquad u = \frac{K}{S^{a}}, \qquad S^{a} = \sum_k \alpha^{a}_k.$$

Predictions are read off the unlabeled set $\mathcal{U}$ and scored with
accuracy and macro-F1.

## Hardware

The reported results were produced on the following machine:

| Component | Specification |
|---|---|
| OS | Windows 11 Home, 64-bit (build 10.0.26200) |
| CPU | Intel Core i9-14900HX (24 cores / 32 threads) |
| RAM | 32 GB |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), driver 561.17, CUDA 12.6 |
| Python | 3.12 |
| PyTorch | 2.6.0+cu124 |

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

## References

1. M. Sensoy, L. Kaplan, and M. Kandemir. "Evidential Deep Learning to Quantify Classification Uncertainty." *NeurIPS*, 2018.
2. Z. Han, C. Zhang, H. Fu, and J. T. Zhou. "Trusted Multi-View Classification." *ICLR*, 2021.
3. Z. Han, C. Zhang, H. Fu, and J. T. Zhou. "Trusted Multi-View Classification with Dynamic Evidential Fusion." *IEEE TPAMI*, 45(2):2551-2566, 2023.
4. C. Xu, J. Si, Z. Guan, W. Zhao, Y. Wu, and X. Gao. "Reliable Conflictive Multi-View Learning." *AAAI*, 38(14):16129-16137, 2024.
5. M. Xie, Z. Han, C. Zhang, Y. Bai, and Q. Hu. "Exploring and Exploiting Uncertainty for Incomplete Multi-View Classification." *CVPR*, pp. 19873-19882, 2023.
6. T. N. Kipf and M. Welling. "Semi-Supervised Classification with Graph Convolutional Networks." *ICLR*, 2017.
7. P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio. "Graph Attention Networks." *ICLR*, 2018.
8. F. Wu, A. Souza, T. Zhang, C. Fifty, T. Yu, and K. Weinberger. "Simplifying Graph Convolutional Networks." *ICML*, pp. 6861-6871, 2019.
9. W. L. Hamilton, R. Ying, and J. Leskovec. "Inductive Representation Learning on Large Graphs." *NeurIPS*, pp. 1024-1034, 2017.
10. X. Zhao, F. Chen, S. Hu, and J.-H. Cho. "Uncertainty Aware Semi-Supervised Learning on Graph Data." *NeurIPS*, 2020.
11. M. Stadler, B. Charpentier, S. Geisler, D. Zügner, and S. Günnemann. "Graph Posterior Network: Bayesian Predictive Uncertainty for Node Classification." *NeurIPS*, 2021.
12. A. Amini, W. Schwarting, A. Soleimany, and D. Rus. "Deep Evidential Regression." *NeurIPS*, 2020.
13. B. Lakshminarayanan, A. Pritzel, and C. Blundell. "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." *NeurIPS*, 2017.
14. Y. Gal and Z. Ghahramani. "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." *ICML*, pp. 1050-1059, 2016.
15. D. P. Kingma and J. Ba. "Adam: A Method for Stochastic Optimization." *ICLR*, 2015.
