# -*- coding: utf-8 -*-
# author: noga mudrik
"""
PSNE: Poisson Stochastic Neighbor Embedding.

Core module containing all algorithm components:
    - Poisson KL distance matrix D
    - High-dim joint probabilities S (weight_exp or perplexity mode)
    - Low-dim joint probabilities Q (Cauchy kernel)
    - Hellinger distance cost + group lasso penalty
    - Analytical gradient (verified against numerical)
    - Gradient descent optimizer with momentum + early exaggeration
    - PSNE class (sklearn-style: fit / fit_transform)

Usage:
    from psne.psne_core import PSNE
    model = PSNE(n_components=3, s_mode='weight_exp', weight_exp=1.0)
    y = model.fit_transform(X)
"""

import numpy as np
import os
from scipy.stats import t as t_dist

#from psne.psne_utils import assert_finite, assert_shape, assert_nonnegative, assert_probability_matrix
#from psne.psne_config import get_default_optimizer_params
from psne_utils import assert_finite, assert_shape, assert_nonnegative, assert_probability_matrix
from psne_config import get_default_optimizer_params


# ====================================================================
# DISTANCES
# ====================================================================

def poisson_kl_pairwise(x_t1, x_t2, epsilon=1e-2):
    """
    Compute element-wise Poisson KL divergence KL(Pois(x_t2) || Pois(x_t1))
    averaged over features. Used for testing; the vectorized version is compute_distance_matrix.
    """
    assert isinstance(epsilon, float), 'epsilon must be a float, got %s' % type(epsilon)
    assert epsilon > 0, 'epsilon must be positive, got %f' % epsilon
    assert x_t1.ndim == 1 and x_t2.ndim == 1, 'x_t1 and x_t2 must be 1D, got %dD and %dD' % (x_t1.ndim, x_t2.ndim)
    assert x_t1.shape[0] == x_t2.shape[0], 'x_t1 and x_t2 length mismatch: %d vs %d' % (x_t1.shape[0], x_t2.shape[0])

    #log_arg = x_t2 / (x_t1 + epsilon) + epsilon
    log_arg = (x_t1 + epsilon) / (x_t2 + epsilon)
    d_n = x_t1 * np.log(log_arg) + x_t2 - x_t1

    assert np.all(log_arg > 0), 'log argument has non-positive values. Min: %f' % np.min(log_arg)
    #d_n = x_t1 * np.log(log_arg) + x_t2 - x_t1
    assert_finite(d_n, 'd_n (element-wise Poisson KL)')
    return np.mean(d_n)


def compute_distance_matrix(X, epsilon=1e-2):
    """
    Compute T x T pairwise Poisson KL distance matrix (vectorized).

    D[t1, t2] = (1/N) sum_n [ x_{n,t1} log(x_{n,t2}/(x_{n,t1}+eps) + eps) + x_{n,t2} - x_{n,t1} ]

    Parameters
    ----------
    X : np.ndarray, shape (N, T) — features x samples, non-negative.
    epsilon : float — numerical stability constant.

    Returns
    -------
    D : np.ndarray, shape (T, T) — asymmetric distance matrix.
    """
    assert isinstance(X, np.ndarray), 'X must be a numpy array, got %s' % type(X)
    assert X.ndim == 2, 'X must be 2D (N x T), got %dD' % X.ndim
    N, T = X.shape
    assert N > 0, 'X must have at least 1 feature, got N=%d' % N
    assert T > 1, 'X must have at least 2 samples, got T=%d' % T
    assert_nonnegative(X, 'X (input data)')
    assert_finite(X, 'X (input data)')

    X_col = X.T  # (T, N)
    X_t1 = X_col[:, np.newaxis, :]  # (T, 1, N)
    X_t2 = X_col[np.newaxis, :, :]  # (1, T, N)

    log_arg = (X_t1 + epsilon) / (X_t2 + epsilon)
    D = np.mean(X_t1 * np.log(log_arg) + X_t2 - X_t1, axis=2)
                
    assert np.all(log_arg > 0), 'log argument has non-positive values. Min: %f' % np.min(log_arg)

    #D = np.mean(X_t1 * np.log(log_arg) + X_t2 - X_t1, axis=2)  # (T, T)

    assert_shape(D, (T, T), 'D (distance matrix)')
    assert_finite(D, 'D (distance matrix)')
    diag_max = np.max(np.abs(np.diag(D)))
    diag_tol = max(epsilon * np.mean(X) * 10, 1e-4)
    assert diag_max < diag_tol, 'Diagonal of D should be ~0, but max |diag| = %f (tol=%f)' % (diag_max, diag_tol)
    return D


# ====================================================================
# PROBABILITIES — S (high-dim)
# ====================================================================

def _compute_conditional_weight_exp(D, weight_exp=1.0):
    """Conditional probabilities via global weight exponent: p_{t2|t1} = exp(-w*D) / sum."""
    assert weight_exp > 0, 'weight_exp must be positive, got %f' % weight_exp
    assert D.ndim == 2 and D.shape[0] == D.shape[1], 'D must be square 2D, got shape %s' % str(D.shape)
    T = D.shape[0]

    exp_neg_wD = np.exp(-weight_exp * D)
    np.fill_diagonal(exp_neg_wD, 0.0)
    row_sums = np.sum(exp_neg_wD, axis=1, keepdims=True)
    assert np.all(row_sums > 0), 'Row sums of exp(-w*D) are zero. Check D and weight_exp.'

    P_cond = exp_neg_wD / row_sums
    np.fill_diagonal(P_cond, 0.0)
    assert_finite(P_cond, 'P_cond (weight_exp mode)')
    return P_cond


def _compute_perplexity_from_row(p_row):
    """Perplexity = 2^{H(P)} where H = -sum p_j log2(p_j)."""
    p_valid = p_row[p_row > 1e-30]
    H = -np.sum(p_valid * np.log2(p_valid))
    return 2.0 ** H


def _binary_search_sigma(distances_row, target_perplexity, tol=1e-5, max_iter=200):
    """Binary search for sigma giving target perplexity for one row."""
    assert target_perplexity > 0, 'target_perplexity must be positive, got %f' % target_perplexity
    sigma_min, sigma_max, sigma = 1e-20, 1e4, 1.0

    for iteration in range(max_iter):
        exp_vals = np.exp(-distances_row / (2.0 * sigma ** 2))
        exp_sum = np.sum(exp_vals)
        if exp_sum == 0:
            exp_sum = 1e-30
        p_row = exp_vals / exp_sum
        perp = _compute_perplexity_from_row(p_row)

        if np.abs(perp - target_perplexity) < tol:
            break
        if perp > target_perplexity:
            sigma_max = sigma
        else:
            sigma_min = sigma
        sigma = (sigma_min + sigma_max) / 2.0

    return p_row, sigma


def _compute_conditional_perplexity(D, perplexity=30.0):
    """Conditional probabilities via adaptive per-point sigma (standard t-SNE style)."""
    assert D.ndim == 2 and D.shape[0] == D.shape[1], 'D must be square 2D, got shape %s' % str(D.shape)
    T = D.shape[0]
    assert 0 < perplexity < T, 'perplexity must be in (0, %d), got %f' % (T, perplexity)

    P_cond = np.zeros((T, T))
    for t1 in range(T):
        dist_row = D[t1, :].copy()
        dist_row[t1] = np.inf
        P_cond[t1, :], _ = _binary_search_sigma(dist_row, perplexity)
        P_cond[t1, t1] = 0.0

    assert_finite(P_cond, 'P_cond (perplexity mode)')
    return P_cond


def _symmetrize_conditionals(P_cond):
    """S = (P + P.T) / (2T), then renormalize to sum to 1."""
    T = P_cond.shape[0]
    S = (P_cond + P_cond.T) / (2.0 * T)
    np.fill_diagonal(S, 0.0)
    s_sum = np.sum(S)
    assert s_sum > 0, 'S sums to 0 — something is wrong with P_cond'
    S = S / s_sum
    assert_finite(S, 'S (joint probabilities)')
    assert_probability_matrix(S, 'S (joint probabilities)', tol=1e-6)
    return S


def compute_S(D, mode='weight_exp', weight_exp=1.0, perplexity=30.0):
    """
    Compute high-dim joint probability matrix S.

    Parameters
    ----------
    D : np.ndarray, shape (T, T)
    mode : str — 'weight_exp' or 'perplexity'
    weight_exp : float — used if mode='weight_exp'
    perplexity : float — used if mode='perplexity'

    Returns
    -------
    S : np.ndarray, shape (T, T) — symmetric, sums to 1.
    """
    assert mode in ('weight_exp', 'perplexity'), 'mode must be "weight_exp" or "perplexity", got "%s"' % mode
    if mode == 'weight_exp':
        P_cond = _compute_conditional_weight_exp(D, weight_exp=weight_exp)
    else:
        P_cond = _compute_conditional_perplexity(D, perplexity=perplexity)
    return _symmetrize_conditionals(P_cond)


# ====================================================================
# PROBABILITIES — Q (low-dim)
# ====================================================================

def compute_Q(y):
    """
    Compute low-dim joint probabilities Q with Cauchy kernel.

    Q_{ij} = (1 + ||y_i - y_j||^2)^{-1} / Z

    Parameters
    ----------
    y : np.ndarray, shape (p, T)

    Returns
    -------
    Q : np.ndarray, shape (T, T)
    Z : float — normalization constant
    inv_distances : np.ndarray, shape (T, T) — unnormalized kernel (useful for gradient)
    """
    assert isinstance(y, np.ndarray), 'y must be a numpy array, got %s' % type(y)
    assert y.ndim == 2, 'y must be 2D (p x T), got %dD' % y.ndim
    p, T = y.shape
    assert T > 1, 'y must have at least 2 samples, got T=%d' % T
    assert_finite(y, 'y (embedding)')

    y_T = y.T  # (T, p)
    diff = y_T[:, np.newaxis, :] - y_T[np.newaxis, :, :]  # (T, T, p)
    sq_distances = np.sum(diff ** 2, axis=2)  # (T, T)
    assert np.all(sq_distances >= 0), 'sq_distances has negatives, min: %f' % np.min(sq_distances)

    inv_distances = 1.0 / (1.0 + sq_distances)
    np.fill_diagonal(inv_distances, 0.0)

    Z = np.sum(inv_distances)
    assert Z > 0, 'Z (normalization) is <= 0: %f' % Z

    Q = np.maximum(inv_distances / Z, 1e-30)
    assert_finite(Q, 'Q (low-dim joint probabilities)')
    return Q, Z, inv_distances


# ====================================================================
# COST
# ====================================================================

def hellinger_distance(S, Q):
    """H(S, Q) = sqrt(0.5 * sum_{ij} (sqrt(S_{ij}) - sqrt(Q_{ij}))^2)"""
    assert S.shape == Q.shape, 'S and Q shape mismatch: %s vs %s' % (str(S.shape), str(Q.shape))
    assert_nonnegative(S, 'S')
    assert_nonnegative(Q, 'Q')
    assert_finite(S, 'S')
    assert_finite(Q, 'Q')
    H = np.sqrt(0.5 * np.sum((np.sqrt(S) - np.sqrt(Q)) ** 2))
    assert np.isfinite(H), 'Hellinger distance is not finite: %s' % str(H)
    return H


def group_lasso_penalty(y):
    """Group lasso (l_{1,2}): sum_n ||y[n, :]||_2"""
    assert y.ndim == 2, 'y must be 2D, got %dD' % y.ndim
    assert_finite(y, 'y')
    penalty = np.sum(np.sqrt(np.sum(y ** 2, axis=1)))
    assert np.isfinite(penalty), 'Group lasso penalty not finite: %s' % str(penalty)
    return penalty


def compute_cost(S, Q, y, gamma=0.0):
    """
    W = H(S, Q) + gamma * group_lasso(y)

    Returns: (W, H_val, penalty_val)
    """
    assert gamma >= 0, 'gamma must be non-negative, got %f' % gamma
    H_val = hellinger_distance(S, Q)
    penalty_val = group_lasso_penalty(y)
    W = H_val + gamma * penalty_val
    assert np.isfinite(W), 'Total cost W not finite: %s' % str(W)
    return W, H_val, penalty_val


# ====================================================================
# GRADIENTS
# ====================================================================

def compute_gradient(S, Q, y, Z, inv_distances, gamma=0.0):
    """
    Gradient of W = H(S,Q) + gamma*group_lasso(y) w.r.t. y.

    Uses element-wise Hellinger chain rule through Q(y):
        dH/dQ_{ij} = -(sqrt(S_{ij}) - sqrt(Q_{ij})) / (4 H sqrt(Q_{ij}))
        dH/dy_i = (4/Z)[-sum_j dH/dQ_{ij}*(y_i-y_j)*f_{ij}^2 + C_Z*sum_m (y_i-y_m)*f_{im}^2]
    where C_Z = sum_{ab} dH/dQ_{ab}*Q_{ab}

    Parameters
    ----------
    S, Q : np.ndarray, shape (T, T)
    y : np.ndarray, shape (p, T)
    Z : float
    inv_distances : np.ndarray, shape (T, T) — unnormalized Cauchy kernel
    gamma : float

    Returns
    -------
    grad : np.ndarray, shape (p, T)
    """
    p, T = y.shape
    assert S.shape == (T, T), 'S shape mismatch: expected (%d,%d), got %s' % (T, T, str(S.shape))
    assert Q.shape == (T, T), 'Q shape mismatch: expected (%d,%d), got %s' % (T, T, str(Q.shape))
    assert inv_distances.shape == (T, T), 'inv_distances shape mismatch: expected (%d,%d), got %s' % (T, T, str(inv_distances.shape))
    assert Z > 0, 'Z must be positive, got %f' % Z
    assert gamma >= 0, 'gamma must be non-negative, got %f' % gamma

    sqrt_S, sqrt_Q = np.sqrt(S), np.sqrt(Q)
    H = np.sqrt(0.5 * np.sum((sqrt_S - sqrt_Q) ** 2))

    if H < 1e-30:
        return np.zeros_like(y)

    dH_dQ = -(sqrt_S - sqrt_Q) / (4.0 * H * sqrt_Q)  # (T, T)
    assert_finite(dH_dQ, 'dH_dQ')

    inv_dist_sq = inv_distances ** 2
    y_T = y.T  # (T, p)
    diff = y_T[:, np.newaxis, :] - y_T[np.newaxis, :, :]  # (T, T, p)

    C_Z = np.sum(dH_dQ * Q)
    term_direct = np.einsum('ij,ijk->ik', dH_dQ * inv_dist_sq, diff)  # (T, p)
    repulsion = np.einsum('ij,ijk->ik', inv_dist_sq, diff)  # (T, p)
    #grad_hellinger = (4.0 / Z) * (term_direct - C_Z * repulsion)
    grad_hellinger = (4.0 / Z) * (-term_direct + C_Z * repulsion)  # (T, p)
    grad_hellinger = grad_hellinger.T  # (p, T)

    assert_finite(grad_hellinger, 'grad_hellinger')

    # Group lasso subgradient
    grad_lasso = np.zeros_like(y)
    if gamma > 0:
        row_norms = np.sqrt(np.sum(y ** 2, axis=1))  # (p,)
        for n in range(p):
            if row_norms[n] > 1e-30:
                grad_lasso[n, :] = gamma * y[n, :] / row_norms[n]

    grad = grad_hellinger + grad_lasso
    assert_finite(grad, 'grad (total)')
    assert_shape(grad, (p, T), 'grad (total)')
    return grad


# ====================================================================
# OPTIMIZER
# ====================================================================

def optimize(S, y_init, optimizer_params):
    """
    Gradient descent with optional momentum and early exaggeration.

    Parameters
    ----------
    S : np.ndarray, shape (T, T) — high-dim joint probabilities.
    y_init : np.ndarray, shape (p, T) — initial embedding.
    optimizer_params : dict — see psne_config.get_default_optimizer_params()

    Returns
    -------
    result : dict with keys: y, cost_history, hellinger_history, n_iter, final_cost, final_hellinger, Q
    """
    eta = optimizer_params['eta']
    max_iter = optimizer_params['max_iter']
    tol = optimizer_params['tol']
    gamma = optimizer_params['gamma']
    use_momentum = optimizer_params['use_momentum']
    momentum_alpha = optimizer_params['momentum_alpha']
    momentum_switch_iter = optimizer_params['momentum_switch_iter']
    momentum_alpha_final = optimizer_params['momentum_alpha_final']
    use_early_exaggeration = optimizer_params['use_early_exaggeration']
    exaggeration_factor = optimizer_params['exaggeration_factor']
    exaggeration_iters = optimizer_params['exaggeration_iters']
    save_every = optimizer_params['save_every']
    path_save = optimizer_params['path_save']
    verbose = optimizer_params['verbose']

    assert eta > 0, 'eta must be positive, got %s' % str(eta)
    assert max_iter > 0, 'max_iter must be positive, got %s' % str(max_iter)
    T = S.shape[0]
    p = y_init.shape[0]
    assert y_init.shape == (p, T), 'y_init shape %s inconsistent with S shape %s' % (str(y_init.shape), str(S.shape))
    assert_finite(y_init, 'y_init')

    y = y_init.copy()
    velocity = np.zeros_like(y)
    cost_history, hellinger_history = [], []

    if save_every > 0 and path_save and not os.path.exists(path_save):
        os.makedirs(path_save)

    prev_cost = np.inf
    for iteration in range(max_iter):
        if use_early_exaggeration and iteration < exaggeration_iters:
            S_eff = S * exaggeration_factor
            S_eff = S_eff / np.sum(S_eff)
        else:
            S_eff = S

        Q, Z, inv_distances = compute_Q(y)
        W, H_val, penalty_val = compute_cost(S_eff, Q, y, gamma=gamma)
        cost_history.append(W)
        hellinger_history.append(H_val)

        cost_change = np.abs(prev_cost - W)
        if iteration > 0 and cost_change < tol:
            if verbose:
                print('Converged at iteration %d, cost=%.6f, change=%.2e' % (iteration, W, cost_change))
            break

        grad = compute_gradient(S_eff, Q, y, Z, inv_distances, gamma=gamma)

        if use_momentum:
            alpha = momentum_alpha if iteration < momentum_switch_iter else momentum_alpha_final
            velocity = alpha * velocity - eta * grad
            y = y + velocity
        else:
            y = y - eta * grad

        assert_finite(y, 'y (after update at iteration %d)' % iteration)
        prev_cost = W

        if verbose and (iteration % max(1, max_iter // 20) == 0 or iteration == max_iter - 1):
            print('iter %d/%d | cost=%.6f | H=%.6f | penalty=%.4f | grad_norm=%.4e' % (
                iteration, max_iter, W, H_val, penalty_val, np.linalg.norm(grad)))

        if save_every > 0 and path_save and iteration > 0 and iteration % save_every == 0:
            np.save(os.path.join(path_save, 'checkpoint_%d.npy' % iteration),
                    {'y': y, 'cost': W, 'iteration': iteration})

    Q, Z, _ = compute_Q(y)
    W_final, H_final, _ = compute_cost(S, Q, y, gamma=gamma)

    return {'y': y, 'cost_history': cost_history, 'hellinger_history': hellinger_history,
            'n_iter': len(cost_history), 'final_cost': W_final, 'final_hellinger': H_final, 'Q': Q}


# ====================================================================
# PSNE CLASS
# ====================================================================

class PSNE:
    """
    PSNE: Poisson Stochastic Neighbor Embedding.

    Combines Poisson KL divergence, Hellinger cost, and optional group lasso sparsity.
    sklearn-style API.

    Parameters
    ----------
    n_components : int — embedding dimension (default 3)
    s_mode : str — 'weight_exp' or 'perplexity' (default 'weight_exp')
    weight_exp : float — weight exponent for S (default 1.0)
    perplexity : float — target perplexity for S (default 30.0)
    epsilon : float — stability constant for Poisson KL (default 1e-2)
    gamma : float — group lasso weight (default 0.0)
    eta : float — learning rate (default 200.0)
    max_iter : int — max optimization iterations (default 1000)
    tol : float — convergence tolerance (default 1e-8)
    df : int — t-distribution df for initialization (default 3)
    use_momentum : bool — enable momentum (default True)
    momentum_alpha : float — initial momentum (default 0.5)
    momentum_alpha_final : float — final momentum (default 0.8)
    momentum_switch_iter : int — iteration to switch momentum (default 250)
    use_early_exaggeration : bool — enable early exaggeration (default True)
    exaggeration_factor : float — S multiplier during exaggeration (default 12.0)
    exaggeration_iters : int — number of exaggeration iterations (default 250)
    random_state : int — random seed (default 42)
    verbose : bool — print progress (default True)
    save_every : int — checkpoint interval, 0=disabled (default 0)
    path_save : str — checkpoint directory (default '')
    """

    def __init__(self, n_components=3, s_mode='weight_exp', weight_exp=1.0, perplexity=30.0,
                 epsilon=1e-2, gamma=0.0, eta=200.0, max_iter=1000, tol=1e-8, df=3,
                 use_momentum=True, momentum_alpha=0.5, momentum_alpha_final=0.8,
                 momentum_switch_iter=250, use_early_exaggeration=True, exaggeration_factor=12.0,
                 exaggeration_iters=250, random_state=42, verbose=True, save_every=0, path_save=''):

        assert isinstance(n_components, int) and n_components > 0, 'n_components must be positive int, got %s' % str(n_components)
        assert s_mode in ('weight_exp', 'perplexity'), 's_mode must be "weight_exp" or "perplexity", got "%s"' % s_mode
        assert epsilon > 0, 'epsilon must be positive, got %f' % epsilon
        assert gamma >= 0, 'gamma must be non-negative, got %f' % gamma
        assert eta > 0, 'eta must be positive, got %f' % eta
        assert max_iter > 0, 'max_iter must be positive, got %d' % max_iter

        self.n_components = n_components
        self.s_mode = s_mode
        self.weight_exp = weight_exp
        self.perplexity = perplexity
        self.epsilon = epsilon
        self.gamma = gamma
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol
        self.df = df
        self.use_momentum = use_momentum
        self.momentum_alpha = momentum_alpha
        self.momentum_alpha_final = momentum_alpha_final
        self.momentum_switch_iter = momentum_switch_iter
        self.use_early_exaggeration = use_early_exaggeration
        self.exaggeration_factor = exaggeration_factor
        self.exaggeration_iters = exaggeration_iters
        self.random_state = random_state
        self.verbose = verbose
        self.save_every = save_every
        self.path_save = path_save

        # Set after fitting
        self.embedding_ = []
        self.cost_history_ = []
        self.hellinger_history_ = []
        self.n_iter_ = 0
        self.D_ = []
        self.S_ = []
        self.Q_ = []

    def fit(self, X):
        """
        Fit PSNE to data X.

        Parameters
        ----------
        X : np.ndarray, shape (N, T) — features x samples, non-negative counts/rates.

        Returns
        -------
        self
        """
        assert isinstance(X, np.ndarray), 'X must be a numpy array, got %s' % type(X)
        assert X.ndim == 2, 'X must be 2D (N x T), got %dD' % X.ndim
        N, T = X.shape
        assert T > 1, 'X must have at least 2 samples, got T=%d' % T
        assert N > 0, 'X must have at least 1 feature, got N=%d' % N
        assert_nonnegative(X, 'X (input data)')
        assert_finite(X, 'X (input data)')
        if self.s_mode == 'perplexity':
            assert self.perplexity < T, 'perplexity (%.1f) must be < T (%d)' % (self.perplexity, T)

        if self.verbose:
            print('PSNE: N=%d features, T=%d samples, p=%d components' % (N, T, self.n_components))
            print('  s_mode=%s, gamma=%.4f, eta=%.1f, max_iter=%d' % (self.s_mode, self.gamma, self.eta, self.max_iter))
            print('  momentum=%s, early_exaggeration=%s' % (self.use_momentum, self.use_early_exaggeration))

        # Step 1: Distance matrix
        if self.verbose:
            print('Step 1/3: Computing Poisson KL distance matrix D...')
        self.D_ = compute_distance_matrix(X, epsilon=self.epsilon)

        # Step 2: Joint probabilities S
        if self.verbose:
            print('Step 2/3: Computing joint probabilities S (mode=%s)...' % self.s_mode)
        self.S_ = compute_S(self.D_, mode=self.s_mode, weight_exp=self.weight_exp, perplexity=self.perplexity)

        # Step 3: Initialize + optimize
        if self.random_state is not None:
            np.random.seed(self.random_state)
        y_init = t_dist.rvs(self.df, size=(self.n_components, T))
        assert_finite(y_init, 'y_init')

        if self.verbose:
            print('Step 3/3: Optimizing embedding...')

        opt_params = {
            'eta': self.eta, 'max_iter': self.max_iter, 'tol': self.tol, 'gamma': self.gamma,
            'use_momentum': self.use_momentum, 'momentum_alpha': self.momentum_alpha,
            'momentum_switch_iter': self.momentum_switch_iter, 'momentum_alpha_final': self.momentum_alpha_final,
            'use_early_exaggeration': self.use_early_exaggeration,
            'exaggeration_factor': self.exaggeration_factor, 'exaggeration_iters': self.exaggeration_iters,
            'save_every': self.save_every, 'path_save': self.path_save, 'verbose': self.verbose,
        }

        result = optimize(self.S_, y_init, opt_params)
        self.embedding_ = result['y']
        self.cost_history_ = result['cost_history']
        self.hellinger_history_ = result['hellinger_history']
        self.n_iter_ = result['n_iter']
        self.Q_ = result['Q']

        if self.verbose:
            print('Done. Final cost=%.6f, Hellinger=%.6f, iterations=%d' % (
                result['final_cost'], result['final_hellinger'], self.n_iter_))
        return self

    def fit_transform(self, X):
        """
        Fit and return embedding as (T, n_components).

        Parameters
        ----------
        X : np.ndarray, shape (N, T)

        Returns
        -------
        embedding : np.ndarray, shape (T, n_components) — samples as rows.
        """
        self.fit(X)
        return self.embedding_.T
