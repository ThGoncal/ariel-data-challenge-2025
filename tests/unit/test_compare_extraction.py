"""Tests : variantes « bord », remplacement des Rp, comparaison appariée, plis identiques."""
import numpy as np
import pandas as pd

from scripts.compare_extraction import (apply_edges, build_features, depth_vector, fold_splits, models, nonneg,
                                        paired_summary)
from src.models.hetero import Prediction


def test_apply_edges_zero_ne_change_rien_et_ne_touche_pas_fgs1():
    d = np.r_[0.5, np.linspace(1.0, 2.0, 282)]
    assert np.array_equal(apply_edges(d, 0), d)
    out = apply_edges(d, 5)
    assert out[0] == d[0]                                       # FGS1 intact
    assert np.allclose(out[1:6], d[1 + 5: 1 + 15].mean())       # 5 premiers canaux AIRS = moyenne des 10 suivants
    assert np.array_equal(out[6:], d[6:])


def test_depth_vector_actuelle_est_rp_au_carre():
    rp = np.linspace(0.05, 0.2, 283)
    assert np.allclose(depth_vector({"rp": rp}, None, 1), rp ** 2)


def test_build_features_remplace_rp_et_garde_le_reste():
    df = pd.DataFrame({"planet_id": [1, 2, 3], "Ms": [1.0, 2.0, 3.0], **{f"Rp_{i}": 9.0 for i in range(1, 284)}})
    out = build_features(df, {"1": np.full(283, 0.04), "3": np.r_[np.full(282, -1.0), 0.09]})
    assert list(out["planet_id"]) == [1, 3] and list(out["Ms"]) == [1.0, 3.0]
    assert np.allclose(out.loc[out.planet_id == 1, "Rp_5"], 0.2)
    assert out.loc[out.planet_id == 3, "Rp_5"].iloc[0] == 0.0 and np.isclose(out.loc[out.planet_id == 3, "Rp_283"].iloc[0], 0.3)


def test_comparaison_appariee():
    s = paired_summary(np.array([0.02, 0.03, 0.01, 0.04, 0.02, 0.03]))
    assert s["plis_gagnes"] == 1.0 and s["diff_ci05"] > 0 and abs(s["diff_moy"] - 0.025) < 1e-9
    s0 = paired_summary(np.array([0.02, -0.02, 0.01, -0.01, 0.03, -0.03]))
    assert s0["diff_ci05"] < 0 < s0["diff_ci95"] and s0["plis_gagnes"] == 0.5


def test_plis_deterministes_et_complets():
    a, b = fold_splits(30, 5, 2, 0), fold_splits(30, 5, 2, 0)
    assert len(a) == 10 and all(np.array_equal(x[1], y[1]) for x, y in zip(a, b))
    for _, te in a[:5]:
        assert len(te) == 6
    assert sorted(np.concatenate([te for _, te in a[:5]])) == list(range(30))


def test_nonneg_borne_mu_a_zero():
    fn = nonneg(lambda a, b, c: Prediction(np.array([[-1.0, 2.0]]), np.array([[1.0, 1.0]])))
    assert (fn(None, None, None).mu >= 0).all()


def test_les_deux_modeles_existent():
    assert len(models()) == 2


def test_egalite_numerique_n_est_pas_une_victoire():
    assert paired_summary(np.array([1e-15, -1e-15, 0.0, 2e-16]))["plis_gagnes"] == 0.0
