# PSNE: Poisson Stochastic Neighbor Embedding

Dimensionality reduction for Poisson-distributed count data.

PSNE embeds high-dimensional count matrices (e.g., neural spike counts, scRNA-seq, text corpora) into 2D or 3D, preserving pairwise structure via Poisson KL divergence and a Hellinger-distance cost function.

---

## Method overview

1. **Poisson KL distance matrix** — Computes a pairwise asymmetric divergence $D_{ij} = \frac{1}{N}\sum_n \left[ x_{n,i} \log\frac{x_{n,i}+\epsilon}{x_{n,j}+\epsilon} + x_{n,j} - x_{n,i} \right]$ between all sample pairs.
2. **High-dimensional joint probabilities $S$** — Converts $D$ into a symmetric probability matrix via either a global weight exponent or adaptive per-point perplexity (t-SNE style).
3. **Low-dimensional joint probabilities $Q$** — Cauchy kernel over the embedding coordinates, as in t-SNE.
4. **Hellinger cost** — Minimizes the Hellinger distance $H(S, Q)$ instead of KL divergence.
5. **Optional group-lasso penalty** — Adds $\gamma \sum_n \|y_n\|_2$ to promote sparsity across embedding dimensions.
6. **Optimizer** — Gradient descent with momentum and early exaggeration.

---

## Installation

```bash
git clone https://github.com/NogaMudrik/psne.git
cd psne
pip install -r requirements.txt
```

### Requirements

Core dependencies (see `requirements.txt`):

```
numpy==1.23.5
scipy==1.8.0
scikit-learn==1.7.2
matplotlib==3.8.2
seaborn==0.11.2
```

---

## Quick start

### Minimal example

```python
import numpy as np
from psne.psne_core import PSNE

# X: features x samples, non-negative counts
# e.g., neurons x conditions, genes x cells, words x documents
X = np.random.poisson(5, size=(50, 30)).astype(float)

model = PSNE(n_components=2, max_iter=500, eta=100.0, verbose=True)
embedding = model.fit_transform(X)  # shape: (30, 2)
```

### Using your own data

```python
import numpy as np
from psne.psne_core import PSNE

# Load your count matrix: shape (N_features, T_samples)
X = np.load('my_data.npy').astype(float)

# Ensure non-negative
assert np.all(X >= 0), 'PSNE requires non-negative input'

# Fit
model = PSNE(
    n_components=3,       # embedding dimension (2 or 3)
    s_mode='weight_exp',  # 'weight_exp' or 'perplexity'
    weight_exp=1.0,       # controls neighborhood sharpness (higher = sharper)
    eta=200.0,            # learning rate
    max_iter=1000,        # optimization iterations
    gamma=0.0,            # group-lasso weight (0 = no sparsity)
    use_momentum=True,
    use_early_exaggeration=True,
    verbose=True,
)
embedding = model.fit_transform(X)  # shape: (T_samples, n_components)
```

### Plotting the result

```python
import matplotlib.pyplot as plt

labels = np.load('my_labels.npy')  # integer labels per sample

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

---

## Data format

| Requirement | Detail |
|---|---|
| Shape | `(N, T)` where `N` = features (neurons, genes, words), `T` = samples (conditions, cells, documents) |
| Type | `float` or `int` numpy array |
| Values | Non-negative (counts or rates). Zeros are handled via the `epsilon` parameter. |

**Important:** Samples are columns, features are rows. The output embedding has shape `(T, n_components)` with samples as rows.

---

## Parameters

### Model parameters

| Parameter | Default | Description |
|---|---|---|
| `n_components` | 3 | Embedding dimensionality |
| `s_mode` | `'weight_exp'` | How to build $S$: `'weight_exp'` (global) or `'perplexity'` (adaptive, t-SNE style) |
| `weight_exp` | 1.0 | Weight exponent for `s_mode='weight_exp'`. Higher values sharpen neighborhoods. |
| `perplexity` | 30.0 | Target perplexity for `s_mode='perplexity'`. Must be < number of samples. |
| `epsilon` | 1e-2 | Smoothing constant for Poisson KL (prevents log-of-zero) |
| `gamma` | 0.0 | Group-lasso regularization weight. Set > 0 to encourage sparse embeddings. |
| `random_state` | 42 | Random seed for initialization |

### Optimizer parameters

| Parameter | Default | Description |
|---|---|---|
| `eta` | 200.0 | Learning rate |
| `max_iter` | 1000 | Maximum iterations |
| `tol` | 1e-8 | Convergence tolerance on cost change |
| `use_momentum` | True | Enable momentum |
| `momentum_alpha` | 0.5 | Initial momentum coefficient |
| `momentum_alpha_final` | 0.8 | Final momentum coefficient |
| `momentum_switch_iter` | 250 | Iteration at which momentum switches from initial to final |
| `use_early_exaggeration` | True | Multiply $S$ by `exaggeration_factor` for the first iterations |
| `exaggeration_factor` | 12.0 | Exaggeration multiplier |
| `exaggeration_iters` | 250 | Number of exaggeration iterations |

---

## Attributes (after fitting)

| Attribute | Shape | Description |
|---|---|---|
| `embedding_` | `(n_components, T)` | Learned embedding (note: `fit_transform` returns the transpose `(T, n_components)`) |
| `cost_history_` | list | Total cost $W$ at each iteration |
| `hellinger_history_` | list | Hellinger distance $H(S,Q)$ at each iteration |
| `D_` | `(T, T)` | Poisson KL distance matrix |
| `S_` | `(T, T)` | High-dimensional joint probabilities |
| `Q_` | `(T, T)` | Low-dimensional joint probabilities (final) |
| `n_iter_` | int | Number of iterations run |

---

## Running the demo

The included demo script generates synthetic Poisson data on a manifold and compares PSNE variants against baselines:

```bash
python psne_demo_nonlinear.py
```

This runs two synthetic datasets (3-group and 4-group XOR) and saves embedding plots, cost convergence curves, and `.npy` files to output subdirectories.


## File structure

```
psne/
├── psne/
│   ├── __init__.py             # Package init, exports PSNE class
│   ├── psne_core.py            # All algorithm components + PSNE class
│   ├── psne_config.py          # Default hyperparameters
│   └── psne_utils.py           # Plotting utilities and assertion helpers
├── psne_test.py                # Unit tests
├── psne_demo_nonlinear.py      # Synthetic data demo script
├── requirements.txt
└── README.md
```

---

## Tips

- **Tuning `weight_exp`:** Start with 1.0. Lower values (e.g., 0.5) produce softer neighborhoods; higher values (e.g., 2.0) produce tighter clusters.
- **Learning rate:** If the cost diverges or oscillates, reduce `eta`. If convergence is slow, increase it.
- **Sparse embeddings:** Set `gamma > 0` (e.g., 0.01) to encourage the embedding to use fewer effective dimensions.
- **Perplexity mode:** Use `s_mode='perplexity'` for datasets where local neighborhood sizes vary. Behaves similarly to perplexity in t-SNE.
- **All-zero samples:** Samples (columns) with all-zero counts should be removed before fitting.

---
