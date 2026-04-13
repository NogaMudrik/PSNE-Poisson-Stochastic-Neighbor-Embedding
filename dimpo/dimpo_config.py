# -*- coding: utf-8 -*-
"""
DIMPO default hyperparameters.
"""


def get_default_optimizer_params():
    """Return default optimizer parameters dict."""
    return {
        'eta': 200.0,
        'max_iter': 1000,
        'tol': 1e-8,
        'gamma': 0.0,
        'use_momentum': True,
        'momentum_alpha': 0.5,
        'momentum_switch_iter': 250,
        'momentum_alpha_final': 0.8,
        'use_early_exaggeration': True,
        'exaggeration_factor': 12.0,
        'exaggeration_iters': 250,
        'save_every': 0,
        'path_save': '',
        'verbose': True,
    }


def get_default_dimpo_params():
    """Return default DIMPO model parameters dict."""
    return {
        'n_components': 3,
        's_mode': 'weight_exp',
        'weight_exp': 1.0,
        'perplexity': 30.0,
        'epsilon': 1e-2,
        'df': 3,
        'random_state': 42,
    }
