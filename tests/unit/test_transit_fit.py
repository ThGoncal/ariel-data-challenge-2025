"""Tests de l'extraction de profondeur : transit décalé, ligne de base qui dérive, spectre préservé."""
import numpy as np
import pytest

from src.signal_processing.transit_fit import (METHODS, depth_fixed_window, detrended, extract_depths,
                                               oot_mask, transit_window)

N_WL = 30


def synth(n=1100, dur=0.40, center=0.50, ingress=0.04, depth=0.01, drift=0.0, noise=3e-5, seed=0):
    """Flux (n_time, N_WL) = 1e6 · base(t) · (1 − d_j · forme(t)) ; profondeur d_j variable avec j."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / n
    shape = np.clip((dur / 2 - np.abs(t - center)) / ingress, 0, 1)
    d_j = depth * (1 + 0.05 * np.sin(np.linspace(0, 3, N_WL)))
    base = 1 + drift * (t - 0.5)
    F = 1e6 * base[:, None] * (1 - d_j[None, :] * shape[:, None]) * (1 + noise * rng.normal(size=(n, N_WL)))
    fgs = 1e6 * base * (1 - depth * shape) * (1 + noise * rng.normal(size=n))
    return F, fgs, d_j


def rel_level_error(est_airs, d_true):
    return abs(est_airs.mean() / d_true.mean() - 1)


def test_sortie_et_methodes():
    F, fgs, d_j = synth()
    for m in METHODS:
        out = extract_depths(F, fgs, m)
        assert out.shape == (N_WL + 1,) and np.isfinite(out).all()
    with pytest.raises(ValueError):
        extract_depths(F, fgs, "inconnue")


def test_fixe_reproduit_le_calcul_historique_en_profondeur():
    F, fgs, _ = synth(dur=0.30)
    n = F.shape[0]
    mid, win = n // 2, n // 10
    expected = np.maximum(1 - np.median(F[mid - win: mid + win], axis=0)
                          / np.median(np.concatenate([F[: mid - win], F[mid + win:]]), axis=0), 0)
    assert np.allclose(depth_fixed_window(F), expected)


@pytest.mark.parametrize("method", ["adaptee", "adaptee_detrend", "gabarit"])
def test_transit_decale_et_long_retrouve_la_profondeur(method):
    F, fgs, d_j = synth(dur=0.30, center=0.72)                 # la fenêtre fixe tombe sur la descente
    est = extract_depths(F, fgs, method)[1:]
    assert rel_level_error(est, d_j) < 0.02
    assert rel_level_error(extract_depths(F, fgs, "fixe")[1:], d_j) > 0.15      # la méthode actuelle échoue


def test_ligne_de_base_qui_derive_exige_le_detrending():
    F, fgs, d_j = synth(dur=0.30, center=0.65, drift=0.012)    # dérive de 1,2 % sur la série, transit décalé
    err = {m: rel_level_error(extract_depths(F, fgs, m)[1:], d_j) for m in ("adaptee", "adaptee_detrend", "gabarit")}
    assert err["adaptee_detrend"] < 0.5 * err["adaptee"]
    assert err["gabarit"] < 0.03


def test_le_gabarit_conserve_la_forme_du_spectre():
    F, fgs, d_j = synth(dur=0.35, center=0.6, drift=0.006)
    est = extract_depths(F, fgs, "gabarit")[1:]
    assert np.corrcoef(est, d_j)[0, 1] > 0.95


def test_detrended_vaut_un_hors_transit_et_corrige_la_derive():
    F, _, _ = synth(dur=0.20, center=0.5, drift=0.01)
    st = transit_window(F.sum(axis=1))
    mask = oot_mask(F.shape[0], st)
    R = detrended(F, mask, 1)
    assert np.abs(R[mask] - 1).max() < 2e-3                    # la dérive de 1 % a disparu hors transit


def test_repli_si_transit_non_detecte():
    rng = np.random.default_rng(0)
    F = 1e6 * (1 + 1e-4 * rng.normal(size=(800, N_WL)))
    st = transit_window(F.sum(axis=1))
    assert not st["detected"]
    mask = oot_mask(800, st)
    assert mask[:80].all() and mask[-80:].all() and not mask[200:600].any()
    out = extract_depths(F, F.sum(axis=1), "gabarit")           # ne plante pas, repli sur la fenêtre fixe
    assert np.isfinite(out).all()
