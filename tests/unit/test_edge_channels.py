"""Tests : ajustement du décalage additif, correctifs des canaux de bord, k par canal leave-one-out."""
import numpy as np

from scripts.check_edge_channels import cap_edges, kj_errors, offset_fit, replace_edges


def synth(c, n=60):
    j = np.arange(n)
    flux = 0.02 + 0.98 * np.clip(j / 12.0, 0, 1) ** 1.5            # flux faible dans les premiers canaux
    ld = 1.0 + 0.2 * np.exp(-j / 15.0)                              # assombrissement : ratio qui décroît en douceur
    ratio = ld / (1.0 + c / flux)                                   # décalage additif c (relatif au flux médian)
    return ratio, flux


def test_decalage_additif_retrouve_et_ameliore_l_ajustement():
    ratio, flux = synth(c=-0.01)
    fit = offset_fit(ratio, flux)
    assert fit["c"] < 0 and abs(fit["c"] - (-0.01)) < 0.006
    assert fit["r2_avec"] > fit["r2_sans"] + 0.2


def test_sans_decalage_c_proche_de_zero():
    ratio, flux = synth(c=0.0)
    assert abs(offset_fit(ratio, flux)["c"]) < 2e-3


def test_plafond_et_remplacement_ne_touchent_que_les_bords():
    est = np.r_[np.full(5, 0.5), np.full(10, 0.01), np.full(20, 0.012)]
    capped, repl = cap_edges(est), replace_edges(est)
    assert np.allclose(capped[:5], 0.02) and np.array_equal(capped[5:], est[5:])
    assert np.allclose(repl[:5], 0.01) and np.array_equal(repl[5:], est[5:])


def test_k_par_canal_absorbe_un_motif_commun():
    rng = np.random.default_rng(0)
    truth = 0.01 * (1 + 0.1 * rng.normal(size=(6, 30)))
    est = truth * np.linspace(2.0, 1.0, 30)[None, :]
    assert np.abs(kj_errors(est, truth)).max() < 1e-12
