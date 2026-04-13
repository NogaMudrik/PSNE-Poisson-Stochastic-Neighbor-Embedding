# -*- coding: utf-8 -*-
"""
DIMPO Demo Script — Synthetic Poisson Data

Demonstrates DIMPO on two synthetic Poisson datasets with known group structure:
  1. nonlinear_standard: 3-group
  2. nonlinear_xor:      4-group  variant (low counts, high sparsity)

Compares DIMPO variants against baselines (PCA, t-SNE, and optionally all run_baselines methods).
Results saved to subfolders encoding key parameters so runs do not overwrite each other.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dimpo'))
from dimpo_utils import save_fig, assert_finite, assert_nonnegative, checkEmptyList
from dimpo_core import DIMPO, compute_distance_matrix, compute_S

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
#from Run_baselines import run_all_baselines, get_default_baseline_config, BASELINE_METHOD_TITLES
plt.close('all')
# ====================================================================
# CONFIG
# ====================================================================
global today
today = '2026-03-15'
# --- Data params ---
N_CONDS_PER_GROUP_NONLINEAR =  20 #30 #30
N_NEURONS_NONLINEAR = 40 # 35#15 #40
N_CONDS_PER_GROUP_XOR = 30
N_NEURONS_XOR =  30 # 20 #30 #  20 #50
SEED = 42

# --- Rate params ---
RATE_BIAS_NONLINEAR = 1.0   # reduce this, e.g. 0.2
RATE_PEAK_NONLINEAR = 8.0   # reduce this, e.g. 2.0
RATE_BIAS_XOR = 0.1         # already low
RATE_PEAK_XOR = 2.5         # reduce this, e.g. 0.8

# --- DIMPO params ---
DIMPO_ETA = 100.0
DIMPO_MAX_ITER = 500
DIMPO_W_LOW = 0.5
DIMPO_W_MID = 1.0
DIMPO_W_HIGH = 2.0
DIMPO_GAMMA_SPARSE = 0.01
N_COMPONENTS = 2#  3  # 2 or 3

# --- What to run ---
RUN_DIMPO_W_LOW = True
RUN_DIMPO_W_MID = True
RUN_DIMPO_W_HIGH = True
RUN_DIMPO_SPARSE = True   # group lasso variant (gamma > 0)
RUN_BASELINES = False  # if False, only PCA + t-SNE are run as baselines
TSNE_PERPLEXITY = 10.0

# ====================================================================
# DATA GENERATION
# ====================================================================

def generate_synthetic_data_nonlinear(n_conds_per_group=30, n_neurons=40, seed=42, rate_bias=1.0, rate_peak=8.0):
    """
    Generate Poisson data on a Swiss Roll manifold.
    PCA will completely fail — groups are along the roll, not linearly separable.
    Returns X (n_neurons x n_samples), labels (n_samples,)
    """
    np.random.seed(seed)
    n_groups = 3
    data_groups = []
    label_list = []

    for g in range(n_groups):
        t_min = g * 2 * np.pi / 3 + 1.5 * np.pi
        t_max = (g + 1) * 2 * np.pi / 3 + 1.5 * np.pi
        t = np.random.uniform(t_min, t_max, n_conds_per_group)
        height = np.random.uniform(0, 10, n_conds_per_group)

        group_data = []
        for i in range(n_conds_per_group):
            rates = np.zeros(n_neurons)
            for n in range(n_neurons):
                n_pref_t = 1.5 * np.pi + 2 * np.pi * (n % 20) / 20
                n_pref_h = 10 * (n // 20) / 2
                dist_t = min(abs(t[i] - n_pref_t), 2 * np.pi - abs(t[i] - n_pref_t))
                dist_h = abs(height[i] - n_pref_h)
                rates[n] = rate_bias + rate_peak * np.exp(-dist_t**2 / 2 - dist_h**2 / 20)
            counts = np.random.poisson(rates)
            group_data.append(counts)

        group_data = np.array(group_data).T
        data_groups.append(group_data)
        label_list.extend([g] * n_conds_per_group)

    X = np.hstack(data_groups).astype(float)
    labels = np.array(label_list)
    assert X.shape == (n_neurons, n_groups * n_conds_per_group), \
        'nonlinear X shape %s unexpected' % str(X.shape)
    assert len(labels) == X.shape[1], 'labels length mismatch'
    return X, labels


def generate_synthetic_data_xor(n_conds_per_group=30, n_neurons=50, seed=42, rate_bias=0.1, rate_peak=2.5):
    """
    Swiss Roll + Poisson: 4 groups, low counts, high sparsity.
    Returns X (n_neurons x n_samples), labels (n_samples,)
    """
    np.random.seed(seed)
    n_groups = 4
    data_groups = []
    label_list = []
    all_t_values = []

    for g in range(n_groups):
        t_min = g * 1.5 + 1.5
        t_max = (g + 1) * 1.5 + 1.5

        group_data = []
       
        for _ in range(n_conds_per_group):
            t = np.random.uniform(t_min, t_max)
            all_t_values.append(t)
            h = np.random.uniform(0, 5)

            rates = np.zeros(n_neurons)
            for n in range(n_neurons):
                n_t = 1.5 + 6.0 * (n % 25) / 25
                n_h = 5.0 * (n // 25) / 2
                dist = np.sqrt((t - n_t)**2 + (h - n_h)**2)
                rates[n] = rate_bias + rate_peak * np.exp(-dist**2 / 3)

            counts = np.random.poisson(rates)
            group_data.append(counts)

        group_data = np.array(group_data).T
        data_groups.append(group_data)
        label_list.extend([g] * n_conds_per_group)

    X = np.hstack(data_groups).astype(float)
    labels = np.array(label_list)

    assert X.shape == (n_neurons, n_groups * n_conds_per_group), \
        'xor X shape %s unexpected' % str(X.shape)
    assert len(labels) == X.shape[1], 'labels length mismatch'

    print('Sparsity: %.1f%% zeros' % (100 * np.mean(X == 0)))
    print('Mean count: %.2f, Max count: %d' % (np.mean(X), int(np.max(X))))
    return X, labels, np.array(all_t_values)


# ====================================================================
# HELPER: RUN ONE DATASET
# ====================================================================

def run_demo_dataset(X, labels, n_groups, dataset_name, path_save, params_dict, t_values=None):
    """
    Runs DIMPO variants + baselines on X, saves figures and embeddings.

    Parameters
    ----------
    X          : np.ndarray (n_features x n_samples)
    labels     : np.ndarray (n_samples,) integer group labels
    n_groups   : int
    dataset_name: str  e.g. 'nonlinear_standard'
    path_save  : str  output directory
    """
    assert isinstance(X, np.ndarray) and X.ndim == 2, \
        'X must be 2D ndarray, got %s' % str(type(X))
    assert_nonnegative(X, 'X (%s)' % dataset_name)
    assert len(labels) == X.shape[1], \
        'labels length %d != n_samples %d' % (len(labels), X.shape[1])
    assert len(np.unique(labels)) == n_groups, \
        'Expected %d groups, got %d unique labels' % (n_groups, len(np.unique(labels)))

    # --- Filter all-zero samples ---
    sample_totals = X.sum(axis=0)
    n_before = X.shape[1]
    nonzero_mask = sample_totals > 0
    X = X[:, nonzero_mask]
    labels = labels[nonzero_mask]
    if t_values is not None:
        t_values = t_values[nonzero_mask]
    n_removed = n_before - X.shape[1]
    
    if n_removed > 0:
        print('Filtered %d all-zero samples: %d -> %d' % (n_removed, n_before, X.shape[1]))
    assert X.shape[1] > 0, 'All samples were zero after filtering for %s' % dataset_name

    n_features, n_samples = X.shape
    os.makedirs(path_save, exist_ok=True)

    print('\n' + '=' * 60)
    print('DATASET: %s | X: %d x %d | groups: %d | n_components: %d' % (dataset_name, n_features, n_samples, n_groups, N_COMPONENTS))
    print('Sparsity: %.1f%% zeros' % (100.0 * np.mean(X == 0)))
    print('=' * 60)

    # ----------------------------------------------------------------
    # Raw data heatmap
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    sns.heatmap(X, ax=ax, cmap='viridis', cbar_kws={'label': 'Count'})
    ax.set_xlabel('Samples (conditions)')
    ax.set_ylabel('Features (neurons)')
    ax.set_title(r'Raw count data $X \in \mathbb{R}^{%d \times %d}$' % (n_features, n_samples))
    plt.tight_layout()
    plt.show()
    save_fig('raw_count_data_heatmap', fig, path_save)

    # ----------------------------------------------------------------
    # Distance matrix D
    # ----------------------------------------------------------------
    print('\n--- Computing distance matrix $D$ ---')
    D = compute_distance_matrix(X, epsilon=1e-2)
    assert D.shape == (n_samples, n_samples), 'D shape mismatch: %s' % str(D.shape)
    assert_finite(D, 'D')
    print('D shape: %s, asymmetry: max|D-D^T| = %.4f' % (str(D.shape), np.max(np.abs(D - D.T))))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(D, ax=axes[0], cmap='coolwarm')
    axes[0].set_title(r'Distance matrix $D$ (Poisson KL)')
    sns.heatmap(D - D.T, ax=axes[1], cmap='coolwarm', center=0)
    axes[1].set_title(r'Asymmetry: $D - D^T$')
    plt.tight_layout()
    plt.show()
    save_fig('distance_matrix_and_asymmetry', fig, path_save)

    # ----------------------------------------------------------------
    # Joint probabilities S
    # ----------------------------------------------------------------
    S = compute_S(D, mode='weight_exp', weight_exp=1.0)
    assert S.shape == (n_samples, n_samples), 'S shape mismatch: %s' % str(S.shape)
    assert_finite(S, 'S')
    print('\nS sums to: %.6f, symmetric: max|S-S^T| = %.2e' % (np.sum(S), np.max(np.abs(S - S.T))))

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    sns.heatmap(S, ax=ax, cmap='viridis')
    ax.set_title(r'Joint probability matrix $S$')
    plt.tight_layout()
    plt.show()
    save_fig('joint_probability_matrix_S', fig, path_save)

    # ----------------------------------------------------------------
    # Run DIMPO variants
    # ----------------------------------------------------------------
    embeddings = {}
    models = {}

    if RUN_DIMPO_W_MID:
        print('\n--- DIMPO ($w$=%.1f) ---' % DIMPO_W_MID)
        model = DIMPO(n_components=N_COMPONENTS, s_mode='weight_exp', weight_exp=DIMPO_W_MID,
                      eta=DIMPO_ETA, max_iter=DIMPO_MAX_ITER, gamma=0.0,
                      use_momentum=True, use_early_exaggeration=True,
                      exaggeration_iters=100, exaggeration_factor=4.0,
                      verbose=True, random_state=SEED)
        t0 = time.time()
        embeddings['dimpo_wexp'] = model.fit_transform(X)
        
        models['dimpo_wexp'] = model
        models['dimpo_wexp'].runtime_sec = time.time() - t0

    if RUN_DIMPO_W_LOW:
        print('\n--- DIMPO ($w$=%.1f) ---' % DIMPO_W_LOW)
        model = DIMPO(n_components=N_COMPONENTS, s_mode='weight_exp', weight_exp=DIMPO_W_LOW,
                      eta=DIMPO_ETA * 2, max_iter=DIMPO_MAX_ITER, gamma=0.0,
                      use_momentum=True, use_early_exaggeration=True,
                      exaggeration_iters=100, exaggeration_factor=4.0,
                      verbose=True, random_state=SEED)
        t0 = time.time()
        embeddings['dimpo_wexp_low'] = model.fit_transform(X)       
        
        models['dimpo_wexp_low'] = model
        models['dimpo_wexp_low'].runtime_sec = time.time() - t0

    if RUN_DIMPO_W_HIGH:
        print('\n--- DIMPO ($w$=%.1f) ---' % DIMPO_W_HIGH)
        model = DIMPO(n_components=N_COMPONENTS, s_mode='weight_exp', weight_exp=DIMPO_W_HIGH,
                      eta=DIMPO_ETA, max_iter=DIMPO_MAX_ITER, gamma=0.0,
                      use_momentum=True, use_early_exaggeration=True,
                      exaggeration_iters=100, exaggeration_factor=4.0,
                      verbose=True, random_state=SEED)
        t0 = time.time()
        embeddings['dimpo_wexp2'] = model.fit_transform(X)
        models['dimpo_wexp2'] = model
        models['dimpo_wexp2'].runtime_sec = time.time() - t0

    if RUN_DIMPO_SPARSE:
        print('\n--- DIMPO (group lasso, gamma=%.3f) ---' % DIMPO_GAMMA_SPARSE)
        model = DIMPO(n_components=N_COMPONENTS, s_mode='weight_exp', weight_exp=DIMPO_W_MID,
                      eta=DIMPO_ETA, max_iter=DIMPO_MAX_ITER, gamma=DIMPO_GAMMA_SPARSE,
                      use_momentum=True, use_early_exaggeration=True,
                      exaggeration_iters=100, exaggeration_factor=4.0,
                      verbose=True, random_state=SEED)
        t0 = time.time()
        embeddings['dimpo_sparse'] = model.fit_transform(X)
        models['dimpo_sparse'] = model
        models['dimpo_sparse'].runtime_sec = time.time() - t0

    # ----------------------------------------------------------------
    # Run baselines
    # ----------------------------------------------------------------
    if RUN_BASELINES:
        baseline_config = get_default_baseline_config()
        baseline_config['n_components'] = N_COMPONENTS
        baseline_config['tsne_perplexity'] = TSNE_PERPLEXITY
        baseline_config['random_state'] = SEED
        baseline_config['to_save'] = True
        baseline_config['ling_xue_learning_rate'] = 0.1
        baseline_config['path_save'] = path_save
        baseline_embeddings = run_all_baselines(X, baseline_config)
    else:
        # Only PCA + t-SNE
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        pca = PCA(n_components=N_COMPONENTS, random_state=SEED)
        tsne = TSNE(n_components=N_COMPONENTS, perplexity=TSNE_PERPLEXITY,
                    learning_rate='auto', init='random', random_state=SEED)
        baseline_embeddings = {
            'pca': pca.fit_transform(X.T),
            'tsne': tsne.fit_transform(X.T),
        }
        print('PCA done. t-SNE done.')

    all_embeddings = dict(baseline_embeddings)
    all_embeddings.update(embeddings)

    for name, emb in all_embeddings.items():
        assert emb.shape == (n_samples, N_COMPONENTS), \
            'Embedding %s shape %s != (%d, %d)' % (name, str(emb.shape), n_samples, N_COMPONENTS)
        assert_finite(emb, 'Embedding %s' % name)

    # ----------------------------------------------------------------
    # Save embeddings + labels
    # ----------------------------------------------------------------
    for name, emb in all_embeddings.items():
        np.save(os.path.join(path_save, 'embedding_%s.npy' % name), emb)
    np.save(os.path.join(path_save, 'labels.npy'), labels)
    print('Saved %d embeddings and labels to %s' % (len(all_embeddings), path_save))
    np.save(os.path.join(path_save, 'data.npy'), X)
    print('Saved X to %s' % os.path.join(path_save, 'data.npy'))
    if t_values is not None:
        np.save(os.path.join(path_save, 't_values.npy'), t_values)
    #np.save(os.path.join(path_save, 't_values.npy'), t_values)
    timing_dict = {}
    for name, model in models.items():
        cost = model.cost_history_
        converge_iter = next((i for i in range(1, len(cost)) if abs(cost[i] - cost[i-1]) < 1e-4), len(cost))
        timing_dict[name] = {
            'runtime_sec': getattr(model, 'runtime_sec', np.nan),
            'converge_iter': converge_iter,
            'final_cost': float(cost[-1]),
            'n_iter_run': len(cost),
        }
    np.save(os.path.join(path_save, 'dimpo_timing.npy'), timing_dict)
    print('Saved timing to %s' % path_save)
    

    # ----------------------------------------------------------------
    # Method display titles and order
    # ----------------------------------------------------------------
    method_titles = {} # dict(BASELINE_METHOD_TITLES)
    method_titles.update({
        'dimpo_wexp':    r'DIMPO ($w$=%.1f)' % DIMPO_W_MID,
        'dimpo_wexp_low':r'DIMPO ($w$=%.1f)' % DIMPO_W_LOW,
        'dimpo_wexp2':   r'DIMPO ($w$=%.1f)' % DIMPO_W_HIGH,
        'dimpo_sparse':  r'DIMPO ($\gamma$=%.2f)' % DIMPO_GAMMA_SPARSE,
    })
    method_order = [m for m in method_titles.keys() if m in all_embeddings]

    # ----------------------------------------------------------------
    # Color config
    # ----------------------------------------------------------------
    tab_col = ['tab:red', 'tab:blue', 'tab:green', 'orange', 'purple',
               'tab:brown', 'tab:pink', 'tab:cyan']
    assert n_groups <= len(tab_col), 'Too many groups (%d) for color list' % n_groups
    color_map_groups = {g: tab_col[g] for g in range(n_groups)}
    color_list = [color_map_groups[l] for l in labels]

    # ----------------------------------------------------------------
    # Embedding plots — all methods
    # ----------------------------------------------------------------
    n_plots = len(method_order)
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))

    if N_COMPONENTS == 3:
        fig = plt.figure(figsize=(5 * n_cols, 7 * n_rows))
        fig.suptitle('%s — colored by group' % dataset_name, fontsize=20, fontweight='bold')

        for i, name in enumerate(method_order, 1):
            emb = all_embeddings[name]
            ax = fig.add_subplot(n_rows, n_cols, i, projection='3d')
            ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=color_list, s=40, edgecolor='gray', lw=0.2)
            ax.set_title(method_titles.get(name, name), fontsize=21)
            ax.set_xlabel(r'$y_1$')
            ax.set_ylabel(r'$y_2$')
            ax.set_zlabel(r'$y_3$')

    elif N_COMPONENTS == 2:
        fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        fig.suptitle('%s — colored by group' % dataset_name, fontsize=20, fontweight='bold')
        axes_flat = np.array(axes_grid).flatten()

        for i, name in enumerate(method_order):
            emb = all_embeddings[name]
            ax = axes_flat[i]
            ax.scatter(emb[:, 0], emb[:, 1], c=color_list, s=40, edgecolor='gray', lw=0.2)
            ax.set_title(method_titles.get(name, name), fontsize=21)
            ax.set_xlabel(r'$y_1$')
            ax.set_ylabel(r'$y_2$')

        for j in range(len(method_order), len(axes_flat)):
            axes_flat[j].set_visible(False)
    else:
        raise ValueError('N_COMPONENTS must be 2 or 3, got %d' % N_COMPONENTS)

    handles = [plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=color_map_groups[g], markersize=10,
                           label='Group %d' % g) for g in range(n_groups)]
    fig.legend(handles=handles, loc='upper center', ncol=n_groups,
               bbox_to_anchor=(0.5, 0.97), fontsize=10)
    plt.tight_layout()
    plt.show()
    save_fig('embeddings_all_methods', fig, path_save)

    # ----------------------------------------------------------------
    # Cost convergence — DIMPO only
    # ----------------------------------------------------------------
    dimpo_models = {k: v for k, v in models.items()}
    if len(dimpo_models) > 0:
        n_dimpo = len(dimpo_models)
        fig, axes = plt.subplots(1, n_dimpo, figsize=(5 * n_dimpo, 4))
        if n_dimpo == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        for ax, (name, model) in zip(axes, dimpo_models.items()):
            ax.plot(model.cost_history_, linewidth=1)
            ax.set_title('%s — Cost' % method_titles.get(name, name))
            ax.set_xlabel('Iteration')
            ax.set_ylabel(r'$W$')
        plt.tight_layout()
        plt.show()
        save_fig('cost_convergence', fig, path_save)

    print('\nDataset %s complete. Results saved to: %s' % (dataset_name, path_save))
    return all_embeddings, labels


# ====================================================================
# MAIN — NONLINEAR STANDARD
# ====================================================================

X_nl, labels_nl = generate_synthetic_data_nonlinear(
    n_conds_per_group=N_CONDS_PER_GROUP_NONLINEAR,
    n_neurons=N_NEURONS_NONLINEAR,
    seed=SEED,
    rate_bias=RATE_BIAS_NONLINEAR,
    rate_peak=RATE_PEAK_NONLINEAR
)

# Include latents_ in path only if N_COMPONENTS != 3
if N_COMPONENTS != 3:
    path_save_nonlinear = os.path.join(
        os.getcwd(), 'latents_%d' % N_COMPONENTS, 'nonlinear_standard',
        'nconds%d_nneurons%d_bias%.2f_peak%.1f_%s' % (N_CONDS_PER_GROUP_NONLINEAR, N_NEURONS_NONLINEAR, RATE_BIAS_NONLINEAR, RATE_PEAK_NONLINEAR, today)
    )
else:
    path_save_nonlinear = os.path.join(
        os.getcwd(), 'nonlinear_standard',
        'nconds%d_nneurons%d_bias%.2f_peak%.1f_%s' % (N_CONDS_PER_GROUP_NONLINEAR, N_NEURONS_NONLINEAR, RATE_BIAS_NONLINEAR, RATE_PEAK_NONLINEAR, today)
    )

params_nl = {
    'dataset': 'nonlinear_standard',
    'n_conds_per_group': N_CONDS_PER_GROUP_NONLINEAR,
    'n_neurons': N_NEURONS_NONLINEAR,
    'rate_bias': RATE_BIAS_NONLINEAR,
    'rate_peak': RATE_PEAK_NONLINEAR,
    'seed': SEED,
    'dimpo_eta': DIMPO_ETA,
    'dimpo_max_iter': DIMPO_MAX_ITER,
    'dimpo_w_low': DIMPO_W_LOW,
    'dimpo_w_mid': DIMPO_W_MID,
    'dimpo_w_high': DIMPO_W_HIGH,
    'dimpo_gamma_sparse': DIMPO_GAMMA_SPARSE,
    'tsne_perplexity': TSNE_PERPLEXITY,
    'n_components': N_COMPONENTS,
    'date': today,
    'script': os.path.abspath(__file__),
}
run_demo_dataset(X_nl, labels_nl, n_groups=3, dataset_name='nonlinear_standard',
                 path_save=path_save_nonlinear, params_dict=params_nl)
# ====================================================================
# MAIN — NONLINEAR XOR
# ====================================================================

X_xor, labels_xor, t_values_xor = generate_synthetic_data_xor(
    n_conds_per_group=N_CONDS_PER_GROUP_XOR,
    n_neurons=N_NEURONS_XOR,
    seed=SEED,
    rate_bias=RATE_BIAS_XOR,
    rate_peak=RATE_PEAK_XOR
)

if N_COMPONENTS != 3:
    path_save_xor = os.path.join(
        os.getcwd(), 'latents_%d' % N_COMPONENTS, 'nonlinear_xor',
        'nconds%d_nneurons%d_bias%.2f_peak%.1f_%s' % (N_CONDS_PER_GROUP_XOR, N_NEURONS_XOR, RATE_BIAS_XOR, RATE_PEAK_XOR, today)
    )
else:
    path_save_xor = os.path.join(
        os.getcwd(), 'nonlinear_xor',
        'nconds%d_nneurons%d_bias%.2f_peak%.1f_%s' % (N_CONDS_PER_GROUP_XOR, N_NEURONS_XOR, RATE_BIAS_XOR, RATE_PEAK_XOR, today)
    )

params_xor = {
    'dataset': 'nonlinear_xor',
    'n_conds_per_group': N_CONDS_PER_GROUP_XOR,
    'n_neurons': N_NEURONS_XOR,
    'rate_bias': RATE_BIAS_XOR,
    'rate_peak': RATE_PEAK_XOR,
    'seed': SEED,
    'dimpo_eta': DIMPO_ETA,
    'dimpo_max_iter': DIMPO_MAX_ITER,
    'dimpo_w_low': DIMPO_W_LOW,
    'dimpo_w_mid': DIMPO_W_MID,
    'dimpo_w_high': DIMPO_W_HIGH,
    'dimpo_gamma_sparse': DIMPO_GAMMA_SPARSE,
    'tsne_perplexity': TSNE_PERPLEXITY,
    'n_components': N_COMPONENTS,
    'date': '2026-03-15',
    'script': os.path.abspath(__file__),
}
run_demo_dataset(X_xor, labels_xor, n_groups=4, dataset_name='nonlinear_xor',
                 path_save=path_save_xor, params_dict=params_xor, t_values=t_values_xor)

print('\nAll demos complete.')