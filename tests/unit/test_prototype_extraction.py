"""Tests de l'évaluation : dispersion du niveau, rescalage k hors-échantillon, erreur de forme."""
import numpy as np

from scripts.prototype_extraction import evaluate


def make(ratios, n_ch=40, seed=0):
    rng = np.random.default_rng(seed)
    base = 0.01 * (1 + 0.1 * rng.normal(size=n_ch))
    truth = np.stack([base * (1 + 0.3 * i) for i in range(len(ratios))])
    return truth * np.asarray(ratios)[:, None], truth


def test_niveau_toujours_juste_a_un_facteur_pres():
    est, truth = make([1.06] * 6)
    r = evaluate(est, truth)
    assert r["disp_niveau"] < 1e-9 and abs(r["biais_k"] - 1 / 1.06) < 1e-9
    assert r["rmse_k_loo"] < 1e-12 and r["rmse_forme"] < 1e-12


def test_dispersion_et_erreur_croissent_avec_les_ecarts_de_niveau():
    e1, t1 = make([1.05, 1.05, 1.06, 1.04, 1.05, 1.06])
    e2, t2 = make([0.82, 1.10, 1.05, 0.90, 1.16, 1.00])
    a, b = evaluate(e1, t1), evaluate(e2, t2)
    assert a["disp_niveau"] < 0.02 < 0.08 < b["disp_niveau"]
    assert a["rmse_k_loo"] < 0.2 * b["rmse_k_loo"]


def test_dispersion_egale_ecart_type_des_log():
    ratios = np.array([0.8, 1.0, 1.2, 1.1])
    est, truth = make(ratios)
    assert abs(evaluate(est, truth)["disp_niveau"] - np.log(ratios).std()) < 1e-9


def test_forme_mesuree_apres_retrait_du_niveau():
    est, truth = make([1.0] * 5)
    est = est * (1 + 0.1 * np.linspace(-1, 1, est.shape[1]))[None, :]      # on déforme le spectre
    r = evaluate(est, truth)
    assert r["rmse_forme"] > 1e-4
    est2, truth2 = make([0.8, 1.2, 1.0, 0.9, 1.1])                          # niveaux faux mais formes justes
    assert evaluate(est2, truth2)["rmse_forme"] < 1e-9


def test_k_par_canal_absorbe_un_motif_commun_a_toutes_les_planetes():
    est, truth = make([1.0] * 6)
    est = est * np.linspace(2.0, 1.0, est.shape[1])[None, :]               # assombrissement : plus fort dans le bleu
    r = evaluate(est, truth)
    assert r["rmse_kj_loo"] < 1e-9 < r["rmse_k_loo"]


def test_niveau_median_robuste_aux_canaux_aberrants():
    est, truth = make([1.1] * 6)
    est[:, 0] = truth[:, 0] * 60                                            # un canal de bord explose
    r = evaluate(est, truth)
    assert r["disp_niveau_med"] < 1e-9
    assert abs(r["rmse_kj_loo_sans5"]) < 1e-9
