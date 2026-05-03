# PSNE: Poisson Stochastic Neighbor Embedding

Dimensionality reduction for Poisson count data.

PSNE embeds high-dimensional count matrices (e.g. neural spike counts,  text corpora) into 2D or 3D. It preserves pairwise structure via a Poisson KL divergence and a Hellinger-distance cost.

## Citation

If you use this code, please cite:

```bibtex
@article{mudrik2026neighbor,
  title={Neighbor Embedding for High-Dimensional Sparse Poisson Data},
  author={Mudrik, Noga and Charles, Adam S},
  journal={arXiv preprint arXiv:2604.16932},
  year={2026}
}
```
You can find the arXiv at: https://arxiv.org/abs/2604.16932

## Method

1. Poisson KL distance matrix. Asymmetric divergence between all sample pairs:

$$D_{ij} = \frac{1}{N}\sum_n \left[ x_{n,i} \log\frac{x_{n,i}+\epsilon}{x_{n,j}+\epsilon} + x_{n,j} - x_{n,i} \right]$$

2. High-dimensional joint probabilities $S$: convert $D$ into a symmetric probability matrix via a global weight exponent or adaptive per-point perplexity.
3. Low-dimensional joint probabilities $Q$: Cauchy kernel over the embedding coordinates, as in t-SNE.
4. Hellinger cost: minimize $H(S, Q)$ instead of KL divergence.
5. Optional group-lasso penalty: $\gamma \sum_n \|y_n\|_2$ promotes sparsity across embedding dimensions.
6. Optimizer: gradient descent with momentum and early exaggeration.

## Installation

```bash
git clone https://github.com/NogaMudrik/PSNE-Poisson-Stochastic-Neighbor-Embedding.git
cd PSNE-Poisson-Stochastic-Neighbor-Embedding
pip install -r requirements.txt
```

Core dependencies:

```
numpy==1.23.5
scipy==1.8.0
scikit-learn==1.7.2
matplotlib==3.8.2
seaborn==0.11.2
```

## Usage

Minimal example:

```python
import numpy as np
from psne.psne_core import PSNE

X = np.random.poisson(5, size=(50, 30)).astype(float)
model = PSNE(n_components=2, max_iter=500, eta=100.0, verbose=True)
embedding = model.fit_transform(X)
```

With your own data:

```python
import numpy as np
from psne.psne_core import PSNE

X = np.load('my_data.npy').astype(float)
assert np.all(X >= 0), 'PSNE requires non-negative input'

model = PSNE(
    n_components=3,
    s_mode='weight_exp',
    weight_exp=1.0,
    eta=200.0,
    max_iter=1000,
    gamma=0.0,
    use_momentum=True,
    use_early_exaggeration=True,
    verbose=True,
)
embedding = model.fit_transform(X)
```

Plotting:

```python
import matplotlib.pyplot as plt

labels = np.load('my_labels.npy')

fig, ax = plt.subplots()
ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', s=30)
ax.set_xlabel('$y_1$')
ax.set_ylabel('$y_2$')
plt.show()
```

For 3D:

```python
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=labels, cmap='tab10', s=30)
plt.show()
```

## Data format

- Shape: $(N, T)$ where $N$ is features (neurons, genes, words) and $T$ is samples (conditions, cells, documents).
- Type: `float` or `int` numpy array.
- Values: non-negative.

Samples are columns, features are rows. The output embedding has shape `(T, n_components)` with samples as rows. Remove all-zero samples before fitting.

## Parameters

Model:

| Parameter | Default | Description |
|---|---|---|
| `n_components` | 3 | Embedding dimensionality. |
| `s_mode` | `'weight_exp'` | How to build $S$: `'weight_exp'` (global) or `'perplexity'` (adaptive). |
| `weight_exp` | 1.0 | Weight exponent for `s_mode='weight_exp'`. Higher sharpens neighborhoods. |
| `perplexity` | 30.0 | Target perplexity for `s_mode='perplexity'`. Must be < number of samples. |
| `epsilon` | 1e-2 | Smoothing constant for Poisson KL. |
| `gamma` | 0.0 | Group-lasso regularization weight ($\gamma > 0$ enforces sparsity). |
| `random_state` | 42 | Random seed for initialization. |

Optimizer:

| Parameter | Default | Description |
|---|---|---|
| `eta` | 200.0 | Learning rate. |
| `max_iter` | 1000 | Maximum iterations. |
| `tol` | 1e-8 | Convergence tolerance on cost change. |
| `use_momentum` | True | Enable momentum. |
| `momentum_alpha` | 0.5 | Initial momentum coefficient. |
| `momentum_alpha_final` | 0.8 | Final momentum coefficient. |
| `momentum_switch_iter` | 250 | Iteration at which momentum switches. |
| `use_early_exaggeration` | True | Multiply $S$ by `exaggeration_factor` for the first iterations. |
| `exaggeration_factor` | 12.0 | Exaggeration multiplier. |
| `exaggeration_iters` | 250 | Number of exaggeration iterations. |

## Attributes (after fitting)

| Attribute | Shape | Description |
|---|---|---|
| `embedding_` | `(n_components, T)` | Learned embedding. `fit_transform` returns the transpose. |
| `cost_history_` | list | Total cost at each iteration. |
| `hellinger_history_` | list | Hellinger distance at each iteration. |
| `D_` | $(T, T)$ | Poisson KL distance matrix. |
| `S_` | $(T, T)$ | High-dimensional joint probabilities. |
| `Q_` | $(T, T)$ | Final low-dimensional joint probabilities. |
| `n_iter_` | int | Number of iterations run. |

## Demo

```bash
python psne_demo_nonlinear.py
```

Runs two synthetic datasets (3-group and 4-group XOR), compares PSNE against baselines, and saves embedding plots, cost curves, and `.npy` files.

## File structure

```
PSNE-Poisson-Stochastic-Neighbor-Embedding/
├── psne/
│   ├── __init__.py
│   ├── psne_core.py
│   ├── psne_config.py
│   └── psne_utils.py
├── psne_demo_nonlinear.py
├── requirements.txt
├── LICENSE
└── README.md
```
