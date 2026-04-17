# -*- coding: utf-8 -*-
# author: noga mudrik
"""
PSNE — all tests in one file.

Covers: distances, probabilities, cost, gradients (numerical check), optimizer, PSNE class.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from psne.psne_core import (poisson_kl_pairwise, compute_distance_matrix, compute_S, compute_Q,
                               hellinger_distance, group_lasso_penalty, compute_cost,
                               compute_gradient, PSNE)


# ====================================================================
# Synthetic data generator
# ====================================================================

def _generate_synthetic_poisson(n_groups=3, n_conds_per_group=20, n_neurons=15, max_lambda=5, seed=42):
    np.random.seed(seed)
    data_groups = []
    label_list = []
    for g in range(n_groups):
        lambdas = np.random.choice(np.arange(1, max_lambda + 1), size=n_neurons)
        group_data = np.vstack([np.random.poisson(lam, size=(1, n_conds_per_group)) for lam in lambdas])
        data_groups.append(group_data)
        label_list.extend([g] * n_conds_per_group)
    X = np.hstack(data_groups).astype(float)
    labels = np.array(label_list)
    return X, labels


# ====================================================================
# DISTANCES
# ====================================================================

def test_kl_pairwise_identical():
    np.random.seed(42)
    x = np.random.poisson(5, size=100).astype(float)
    d = poisson_kl_pairwise(x, x, epsilon=1e-2)
    assert np.abs(d) < 0.1, 'KL(x, x) should be ~0, got %f' % d


def test_kl_pairwise_asymmetric():
    np.random.seed(42)
    x1 = np.random.poisson(3, size=100).astype(float)
    x2 = np.random.poisson(7, size=100).astype(float)
    d12, d21 = poisson_kl_pairwise(x1, x2), poisson_kl_pairwise(x2, x1)
    assert not np.isclose(d12, d21, atol=1e-6), 'KL should be asymmetric, got d12=%f, d21=%f' % (d12, d21)


def test_kl_pairwise_with_zeros():
    x1 = np.array([0.0, 0.0, 5.0, 3.0])
    x2 = np.array([3.0, 0.0, 0.0, 7.0])
    d = poisson_kl_pairwise(x1, x2, epsilon=1e-2)
    assert np.isfinite(d), 'KL with zeros should be finite, got %f' % d


def test_distance_matrix_shape_and_diag():
    np.random.seed(42)
    X = np.random.poisson(5, size=(20, 10)).astype(float)
    D = compute_distance_matrix(X)
    assert D.shape == (10, 10), 'D should be (10,10), got %s' % str(D.shape)
    assert np.max(np.abs(np.diag(D))) < 0.1, 'Diagonal should be ~0'


def test_distance_matrix_asymmetric():
    np.random.seed(42)
    X = np.random.poisson(4, size=(50, 10)).astype(float)
    D = compute_distance_matrix(X)
    assert np.max(np.abs(D - D.T)) > 1e-6, 'D should be asymmetric'


def test_distance_matrix_matches_loop():
    np.random.seed(42)
    N, T, eps = 15, 8, 1e-2
    X = np.random.poisson(4, size=(N, T)).astype(float)
    D_vec = compute_distance_matrix(X, epsilon=eps)
    D_loop = np.zeros((T, T))
    for t1 in range(T):
        for t2 in range(T):
            D_loop[t1, t2] = poisson_kl_pairwise(X[:, t1], X[:, t2], epsilon=eps)
    max_diff = np.max(np.abs(D_vec - D_loop))
    assert max_diff < 1e-10, 'Vectorized vs loop differ by %e' % max_diff


def test_distance_matrix_rejects_negative():
    X = np.array([[1.0, -2.0], [3.0, 4.0]])
    try:
        compute_distance_matrix(X)
        raise RuntimeError('Should have raised AssertionError for negative values')
    except AssertionError:
        pass


# ====================================================================
# PROBABILITIES
# ====================================================================

def test_S_weight_exp():
    np.random.seed(42)
    D = np.random.rand(12, 12)
    S = compute_S(D, mode='weight_exp', weight_exp=1.5)
    assert np.abs(np.sum(S) - 1.0) < 1e-6, 'S sums to %f' % np.sum(S)
    assert np.all(S >= 0), 'S has negative values'
    assert np.max(np.abs(S - S.T)) < 1e-10, 'S is not symmetric'


def test_S_perplexity():
    np.random.seed(42)
    D = np.random.rand(15, 15) * 3
    S = compute_S(D, mode='perplexity', perplexity=5.0)
    assert np.abs(np.sum(S) - 1.0) < 1e-6, 'S sums to %f' % np.sum(S)
    assert np.all(S >= 0), 'S has negative values'


def test_S_rejects_bad_mode():
    D = np.random.rand(5, 5)
    try:
        compute_S(D, mode='bad_mode')
        raise RuntimeError('Should have raised AssertionError')
    except AssertionError:
        pass


def test_Q_properties():
    np.random.seed(42)
    y = np.random.randn(3, 10)
    Q, Z, inv_dist = compute_Q(y)
    assert Q.shape == (10, 10), 'Q shape should be (10,10), got %s' % str(Q.shape)
    assert np.abs(np.sum(Q) - 1.0) < 1e-3, 'Q sums to %f' % np.sum(Q)
    assert np.max(np.abs(Q - Q.T)) < 1e-10, 'Q is not symmetric'
    assert np.all(Q >= 0), 'Q has negative values'
    assert Z > 0, 'Z should be positive, got %f' % Z


# ====================================================================
# COST
# ====================================================================

def test_hellinger_identical():
    S = np.array([[0.0, 0.3], [0.3, 0.0]]) / 0.6
    assert hellinger_distance(S, S) < 1e-10, 'H(S,S) should be 0'


def test_hellinger_range_and_symmetry():
    np.random.seed(42)
    S = np.random.rand(10, 10); S = S / np.sum(S)
    Q = np.random.rand(10, 10); Q = Q / np.sum(Q)
    H_sq = hellinger_distance(S, Q)
    H_qs = hellinger_distance(Q, S)
    assert 0 <= H_sq <= 1.0, 'H should be in [0,1], got %f' % H_sq
    assert np.abs(H_sq - H_qs) < 1e-10, 'Hellinger should be symmetric'


def test_group_lasso():
    assert group_lasso_penalty(np.zeros((3, 10))) == 0.0, 'Penalty of zeros should be 0'
    np.random.seed(42)
    assert group_lasso_penalty(np.random.randn(3, 10)) > 0, 'Penalty of nonzero should be > 0'


def test_cost_gamma():
    np.random.seed(42)
    S = np.random.rand(8, 8); S = S / np.sum(S)
    Q = np.random.rand(8, 8); Q = Q / np.sum(Q)
    y = np.random.randn(3, 8)
    W0, H0, _ = compute_cost(S, Q, y, gamma=0.0)
    W1, H1, _ = compute_cost(S, Q, y, gamma=0.1)
    assert np.abs(W0 - H0) < 1e-10, 'gamma=0: W should equal H'
    assert W1 > H1, 'gamma>0: W should be > H'


# ====================================================================
# GRADIENTS (numerical check)
# ====================================================================

def _numerical_gradient(S, y, gamma=0.0, h=1e-5):
    grad_num = np.zeros_like(y)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y_p, y_m = y.copy(), y.copy()
            y_p[i, j] += h
            y_m[i, j] -= h
            Q_p, _, _ = compute_Q(y_p)
            Q_m, _, _ = compute_Q(y_m)
            W_p, _, _ = compute_cost(S, Q_p, y_p, gamma=gamma)
            W_m, _, _ = compute_cost(S, Q_m, y_m, gamma=gamma)
            grad_num[i, j] = (W_p - W_m) / (2.0 * h)
    return grad_num


def test_gradient_numerical_no_gamma():
    np.random.seed(123)
    T, p = 6, 2
    S = np.random.rand(T, T); S = (S + S.T) / 2.0; np.fill_diagonal(S, 0); S = S / np.sum(S)
    y = np.random.randn(p, T) * 0.5
    Q, Z, inv_dist = compute_Q(y)
    grad_a = compute_gradient(S, Q, y, Z, inv_dist, gamma=0.0)
    grad_n = _numerical_gradient(S, y, gamma=0.0, h=1e-5)
    rel_err = np.max(np.abs(grad_a - grad_n)) / (np.max(np.abs(grad_n)) + 1e-30)
    assert rel_err < 0.05, 'Gradient mismatch (gamma=0): rel_err=%e' % rel_err


def test_gradient_numerical_with_gamma():
    np.random.seed(456)
    T, p, gamma = 6, 2, 0.1
    S = np.random.rand(T, T); S = (S + S.T) / 2.0; np.fill_diagonal(S, 0); S = S / np.sum(S)
    y = np.random.randn(p, T) * 0.5
    Q, Z, inv_dist = compute_Q(y)
    grad_a = compute_gradient(S, Q, y, Z, inv_dist, gamma=gamma)
    grad_n = _numerical_gradient(S, y, gamma=gamma, h=1e-5)
    rel_err = np.max(np.abs(grad_a - grad_n)) / (np.max(np.abs(grad_n)) + 1e-30)
    assert rel_err < 0.05, 'Gradient mismatch (gamma=%.2f): rel_err=%e' % (gamma, rel_err)


def test_gradient_zero_at_convergence():
    np.random.seed(42)
    y = np.random.randn(2, 6) * 0.1
    Q, Z, inv_dist = compute_Q(y)
    grad = compute_gradient(Q.copy(), Q, y, Z, inv_dist, gamma=0.0)
    assert np.linalg.norm(grad) < 1e-6, 'Grad should be ~0 when S==Q, norm=%e' % np.linalg.norm(grad)


# ====================================================================
# PSNE CLASS (end-to-end)
# ====================================================================

def test_psne_fit_transform_shape():
    X, _ = _generate_synthetic_poisson(n_groups=2, n_conds_per_group=15, n_neurons=10)
    model = PSNE(n_components=3, max_iter=50, verbose=False, eta=50.0,
                  use_momentum=False, use_early_exaggeration=False)
    emb = model.fit_transform(X)
    T = X.shape[1]
    assert emb.shape == (T, 3), 'Expected (%d,3), got %s' % (T, str(emb.shape))


def test_psne_cost_decreases():
    X, _ = _generate_synthetic_poisson(n_groups=2, n_conds_per_group=15, n_neurons=10)
    model = PSNE(n_components=2, max_iter=100, verbose=False, eta=50.0,
                  use_momentum=False, use_early_exaggeration=False)
    model.fit(X)
    costs = model.cost_history_
    n_early = max(1, len(costs) // 10)
    assert np.mean(costs[-n_early:]) < np.mean(costs[:n_early]), 'Cost should decrease over iterations'


def test_psne_with_momentum():
    X, _ = _generate_synthetic_poisson(n_groups=2, n_conds_per_group=10, n_neurons=8)
    emb = PSNE(n_components=2, max_iter=50, verbose=False, eta=50.0,
                use_momentum=True, use_early_exaggeration=False).fit_transform(X)
    assert np.all(np.isfinite(emb)), 'Embedding contains NaN/Inf with momentum'


def test_psne_with_early_exaggeration():
    X, _ = _generate_synthetic_poisson(n_groups=2, n_conds_per_group=10, n_neurons=8)
    emb = PSNE(n_components=2, max_iter=50, verbose=False, eta=50.0,
                use_momentum=False, use_early_exaggeration=True,
                exaggeration_factor=4.0, exaggeration_iters=20).fit_transform(X)
    assert np.all(np.isfinite(emb)), 'Embedding contains NaN/Inf with early exaggeration'


def test_psne_with_gamma():
    X, _ = _generate_synthetic_poisson(n_groups=2, n_conds_per_group=10, n_neurons=8)
    emb = PSNE(n_components=2, max_iter=50, verbose=False, eta=50.0, gamma=0.01,
                use_momentum=False, use_early_exaggeration=False).fit_transform(X)
    assert np.all(np.isfinite(emb)), 'Embedding contains NaN/Inf with gamma>0'


def test_psne_perplexity_mode():
    X, _ = _generate_synthetic_poisson(n_groups=2, n_conds_per_group=15, n_neurons=8)
    emb = PSNE(n_components=2, s_mode='perplexity', perplexity=5.0,
                max_iter=50, verbose=False, eta=50.0,
                use_momentum=False, use_early_exaggeration=False).fit_transform(X)
    assert np.all(np.isfinite(emb)), 'Embedding contains NaN/Inf in perplexity mode'


def test_psne_rejects_negative():
    try:
        PSNE(n_components=1, max_iter=10, verbose=False).fit(np.array([[1.0, -1.0], [2.0, 3.0]]))
        raise RuntimeError('Should have raised AssertionError for negative data')
    except AssertionError:
        pass


# ====================================================================
# RUN ALL
# ====================================================================

if __name__ == '__main__':
    all_tests = [
        # distances
        test_kl_pairwise_identical, test_kl_pairwise_asymmetric, test_kl_pairwise_with_zeros,
        test_distance_matrix_shape_and_diag, test_distance_matrix_asymmetric,
        test_distance_matrix_matches_loop, test_distance_matrix_rejects_negative,
        # probabilities
        test_S_weight_exp, test_S_perplexity, test_S_rejects_bad_mode, test_Q_properties,
        # cost
        test_hellinger_identical, test_hellinger_range_and_symmetry, test_group_lasso, test_cost_gamma,
        # gradients
        test_gradient_numerical_no_gamma, test_gradient_numerical_with_gamma, test_gradient_zero_at_convergence,
        # end-to-end
        test_psne_fit_transform_shape, test_psne_cost_decreases,
        test_psne_with_momentum, test_psne_with_early_exaggeration,
        test_psne_with_gamma, test_psne_perplexity_mode, test_psne_rejects_negative,
    ]
    for t in all_tests:
        t()
        print('PASS: %s' % t.__name__)
    print('\n=== ALL %d TESTS PASSED ===' % len(all_tests))
