"""Tests du module d'extraction partagé (entraînement et inférence)."""
import numpy as np
import pytest

from spectra_factory import synth_curve
from src.signal_processing.spectra import (ExtractionParams, curves_from_cubes, depth_to_rp, extract_spectrum,
                                           replace_edge_channels, spectrum_from_cubes)
from src.signal_processing.transit_fit import extract_depths


def test_params_defauts_surcharges_et_validation():
    assert ExtractionParams.from_params(None) == ExtractionParams("adaptee", 1, 10)
    p = ExtractionParams.from_params({"extraction": {"method": "gabarit", "edge_channels": 5}})
    assert p == ExtractionParams("gabarit", 1, 5)
    with pytest.raises(ValueError):
        ExtractionParams.from_params({"extraction": {"method": "magique"}})
    with pytest.raises(ValueError):
        ExtractionParams(edge_channels=-1)


def test_replace_edge_channels():
    d = np.linspace(1.0, 2.0, 282)
    assert np.array_equal(replace_edge_channels(d, 0), d)
    out = replace_edge_channels(d, 5)
    assert np.allclose(out[:5], d[5:15].mean()) and np.array_equal(out[5:], d[5:])
    assert out is not d and d[0] == 1.0                              # pas de modification en place
    with pytest.raises(ValueError):
        replace_edge_channels(d, 280)


def test_extract_spectrum_forme_et_bord():
    F, fgs, _ = synth_curve(0)
    s0 = extract_spectrum(F, fgs, ExtractionParams("adaptee", 1, 0))
    s10 = extract_spectrum(F, fgs, ExtractionParams("adaptee", 1, 10))
    assert s0.shape == (283,) and np.isfinite(s0).all()
    assert s0[0] == s10[0] and np.array_equal(s0[11:], s10[11:])      # FGS1 et canaux au-delà du bord intacts
    assert np.allclose(s10[1:11], s0[11:21].mean())
    assert np.allclose(extract_spectrum(F, fgs, ExtractionParams("fixe", 1, 0)), extract_depths(F, fgs, "fixe"))


def test_spectre_depuis_les_cubes_identique_aux_courbes():
    F, fgs, _ = synth_curve(1)
    n = len(fgs)
    pix = np.random.default_rng(0).dirichlet(np.ones(4), size=(n, 282)).transpose(0, 2, 1)   # (n, 4 pixels, 282), somme 1
    cubes = {"airs": F[:, None, :] * pix, "fgs1": fgs[:, None, None] * np.full((1, 2, 2), 0.25)}
    airs, fg = curves_from_cubes(cubes)
    assert np.allclose(airs, F, rtol=1e-9) and np.allclose(fg, fgs, rtol=1e-9)
    assert np.allclose(spectrum_from_cubes(cubes), extract_spectrum(F, fgs))


def test_depth_to_rp_borne_a_zero():
    assert np.allclose(depth_to_rp(np.array([-0.1, 0.0, 0.04])), [0.0, 0.0, 0.2])
