"""Tests des modèles à σ(x) : données synthétiques où le bruit dépend d'une feature connue."""
import numpy as np
import pytest

import scipy.stats
from src.models.hetero import MODELS, Prediction, mlp_hetero, ridge_const, ridge_scaled

N_WL, N_FEAT = 283, 12


def make_data(n, seed):
    """y = signal linéaire + bruit dont l'amplitude dépend de la feature 0 (planète facile/difficile)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, N_FEAT))
    W = np.random.default_rng(123).normal(size=(N_FEAT, N_WL)) * 1e-3     # même loi X→y en train et en test
    noise_scale = 1e-4 * np.exp(1.2 * X[:, 0])                 # varie d'un facteur ~10 entre planètes
    y = 0.02 + X @ W + rng.normal(size=(n, N_WL)) * noise_scale[:, None]
    return X, y, noise_scale


@pytest.fixture(scope="module")
def data():
    Xtr, ytr, _ = make_data(100, 1)
    Xte, yte, s_te = make_data(40, 2)
    return Xtr, ytr, Xte, yte, s_te


@pytest.mark.parametrize("name", list(MODELS))
def test_forme_et_positivite(name, data):
    pytest.importorskip("torch") if name == "mlp-hetero" else None
    Xtr, ytr, Xte, *_ = data
    p = MODELS[name](Xtr, ytr, Xte)
    assert isinstance(p, Prediction)
    assert p.mu.shape == p.sigma.shape == (len(Xte), N_WL)
    assert np.isfinite(p.mu).all() and np.isfinite(p.sigma).all() and (p.sigma > 0).all()


def test_ridge_const_sigma_identique_pour_toutes_les_planetes(data):
    Xtr, ytr, Xte, *_ = data
    p = ridge_const(Xtr, ytr, Xte)
    assert np.allclose(p.sigma, p.sigma[0])


def test_echelon_A_sigma_suit_le_bruit_reel(data):
    Xtr, ytr, Xte, _, s_te = data
    p = ridge_scaled(Xtr, ytr, Xte)
    corr = np.corrcoef(np.log(p.sigma.mean(axis=1)), np.log(s_te))[0, 1]
    assert corr > 0.7                                          # σ(x) retrouve les planètes bruitées


def test_echelon_A_ne_degrade_pas_la_baseline(data):
    # A ne change pas μ : il ne peut gagner que par la calibration de σ (gain faible ici, car l'erreur
    # de μ domine le bruit) ; on vérifie surtout qu'il ne dégrade pas.
    Xtr, ytr, Xte, yte, _ = data
    assert mean_ll(yte, ridge_scaled(Xtr, ytr, Xte)) >= mean_ll(yte, ridge_const(Xtr, ytr, Xte)) - 0.05


def test_echelon_B_bat_la_baseline_et_reste_borne(data):
    pytest.importorskip("torch")
    Xtr, ytr, Xte, yte, _ = data
    base = ridge_const(Xtr, ytr, Xte)
    p = mlp_hetero(Xtr, ytr, Xte, n_models=2, epochs=150)
    # le MLP pondère implicitement les planètes bruyantes (effet moindres carrés pondérés) : gain net
    assert mean_ll(yte, p) > mean_ll(yte, base) + 0.3
    assert p.sigma.max() / p.sigma.min() < 1e4                 # σ borné, pas d'effondrement


def test_mlp_deterministe(data):
    pytest.importorskip("torch")
    Xtr, ytr, Xte, *_ = data
    p1 = mlp_hetero(Xtr, ytr, Xte, n_models=1, epochs=30, seed=3)
    p2 = mlp_hetero(Xtr, ytr, Xte, n_models=1, epochs=30, seed=3)
    assert np.allclose(p1.mu, p2.mu) and np.allclose(p1.sigma, p2.sigma)


def mean_ll(y, p):
    """Log-vraisemblance gaussienne moyenne (la métrique officielle est écrêtée à 0 sur ces données)."""
    return float(scipy.stats.norm.logpdf(y, loc=p.mu, scale=p.sigma).mean())
