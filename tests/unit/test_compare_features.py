"""Tests des variantes de features : profondeur ≈ Rp², donc l'entrée Rp² doit aider un modèle linéaire."""
import numpy as np

from scripts.compare_features import (N_WL, SIGMA_MODES, channel_linear, level_stats, ridge_variant,
                                      scaled_identity, variants, with_oof_sigma, with_sigma)


def make(n, seed, k=1.05, noise=1e-4):
    """wl = k · Rp² + bruit ; profondeurs de 0,0004 à 0,06 (Rp de 0,02 à 0,25), avec quelques extrêmes."""
    rng = np.random.default_rng(seed)
    level = rng.uniform(0.02, 0.25, size=(n, 1))
    rp = level * (1 + 0.05 * rng.normal(size=(n, N_WL)))
    astro = rng.normal(size=(n, 6))
    y = k * rp ** 2 + noise * rng.normal(size=(n, N_WL))
    return np.hstack([rp, astro]), y


def test_k_rp2_retrouve_le_facteur():
    Xtr, ytr = make(80, 0)
    Xte, yte = make(30, 1)
    mu = scaled_identity(Xtr, ytr, Xte)
    assert abs(np.median(mu / yte) - 1) < 0.02


def test_affine_par_canal_retrouve_la_relation():
    Xtr, ytr = make(80, 0)
    Xte, yte = make(30, 1)
    assert np.sqrt(((channel_linear(Xtr, ytr, Xte) - yte) ** 2).mean()) < 5e-4


def test_entree_rp2_bat_entree_rp_pour_un_modele_lineaire():
    Xtr, ytr = make(80, 0)
    Xte, yte = make(40, 1)
    rmse = lambda mu: np.sqrt(((mu - yte) ** 2).mean())
    e_rp = rmse(ridge_variant(10, power=1, astro=False)(Xtr, ytr, Xte))
    e_rp2 = rmse(ridge_variant(10, power=2, astro=False)(Xtr, ytr, Xte))
    assert e_rp2 < e_rp


def test_sigma_hors_echantillon_positif_et_forme():
    Xtr, ytr = make(40, 0)
    Xte, _ = make(10, 1)
    p = with_oof_sigma(ridge_variant(10, 2, True), Xtr, ytr, Xte)
    assert p.mu.shape == p.sigma.shape == (10, N_WL) and (p.sigma > 0).all()


def test_astro_ecretees_ne_plante_pas_sur_valeur_extreme():
    Xtr, ytr = make(40, 0)
    Xte, _ = make(5, 1)
    Xte[0, N_WL:] = 1e6                                    # planète aux features astro aberrantes
    mu = ridge_variant(10, 2, True, clip=True)(Xtr, ytr, Xte)
    assert np.isfinite(mu).all() and np.abs(mu).max() < 1       # l'écrêtage borne l'effet


def test_liste_des_variantes_et_stats_de_niveau():
    assert len(variants([1, 10])) == 2 * 5 + 2
    assert SIGMA_MODES == ("insample", "kfold5", "loo")
    E = np.zeros((4, N_WL)); E[0] += 0.01
    mu = -E
    ids = np.array([7, 8, 9, 10])
    rmse, lvl = level_stats(np.zeros((4, N_WL)), mu, ids, [7])
    assert abs(lvl - 0.01) < 1e-12 and rmse > 0


def test_modes_sigma_forme_et_ordre():
    Xtr, ytr = make(40, 0, noise=1e-3)
    Xte, _ = make(10, 1, noise=1e-3)
    fn = ridge_variant(1, 1, True)                          # peu régularisé : surajuste en échantillon
    sig = {m: with_sigma(fn, Xtr, ytr, Xte, m).sigma for m in SIGMA_MODES}
    for m in SIGMA_MODES:
        assert sig[m].shape == (10, N_WL) and (sig[m] > 0).all()
    assert np.median(sig["insample"]) < np.median(sig["loo"])   # l'erreur hors-échantillon est plus grande
