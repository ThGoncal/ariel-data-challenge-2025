"""Tests du diagnostic des planètes aberrantes."""
import numpy as np
import pandas as pd

from scripts.diagnose_outliers import correlations, planet_table, without_worst
from spectra_factory import make_world
from src.pipelines.build_spectra import build_layout
from src.pipelines.train_champion import load_layout
from src.signal_processing.spectra import ExtractionParams


def all_planets(tmp_path):
    split, curves, raw = make_world(tmp_path)
    build_layout(split, curves, raw, tmp_path / "s", ExtractionParams())
    train, val = load_layout(tmp_path / "s")
    return pd.concat([val, train], ignore_index=True)


def test_une_ligne_par_planete_triee_du_pire_au_meilleur(tmp_path):
    df = all_planets(tmp_path)
    t = planet_table(df, folds=3, repeats=2)
    assert len(t) == len(df) and set(t["planet_id"]) == set(df["planet_id"])
    assert t["score"].is_monotonic_increasing
    assert {"frac_z3", "niveau_modele", "niveau_extraction", "profondeur_med", "astro_zmax", "score_FGS1"} <= set(t.columns)
    assert np.isfinite(t[["score", "niveau_modele", "niveau_extraction"]].to_numpy()).all()


def test_deterministe(tmp_path):
    df = all_planets(tmp_path)
    a, b = planet_table(df, folds=3, repeats=1), planet_table(df, folds=3, repeats=1)
    assert np.allclose(a["score"], b["score"]) and list(a["planet_id"]) == list(b["planet_id"])


def test_retirer_les_pires_ne_baisse_jamais_le_score():
    t = pd.DataFrame({"score": [-5.0, -1.0, 0.2, 0.4, 0.5, 0.6]})
    w = without_worst(t, ks=(0, 1, 2, 4, 99))
    assert list(w["k_pires_retirees"]) == [0, 1, 2, 4]                       # k >= n ignoré
    assert w["score"].is_monotonic_increasing
    assert w["score"].iloc[0] == 0.0                                          # moyenne négative clippée comme la métrique


def test_correlations_detectent_un_lien_monotone():
    rng = np.random.default_rng(0)
    d = rng.uniform(0.005, 0.05, 60)
    t = pd.DataFrame({"score": -d * 10, "profondeur_med": d, "niveau_extraction": 1 + rng.normal(0, 0.01, 60),
                      "niveau_modele": 1 + rng.normal(0, 0.01, 60), "astro_zmax": rng.uniform(0, 3, 60)})
    c = correlations(t).set_index("indicateur")
    assert c.loc["profondeur_med", "rho_spearman"] < -0.99 and c.loc["profondeur_med", "p"] < 1e-6
