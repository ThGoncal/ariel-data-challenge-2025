"""Tests de la courbe d'apprentissage : plis communs, sous-ensembles imbriqués, résumé apparié."""
import numpy as np
import pandas as pd

from scripts.learning_curve import learning_curve, summarize
from spectra_factory import make_world
from src.pipelines.build_spectra import build_layout
from src.pipelines.train_champion import load_layout
from src.signal_processing.spectra import ExtractionParams


def all_planets(tmp_path):
    split, curves, raw = make_world(tmp_path)
    build_layout(split, curves, raw, tmp_path / "s", ExtractionParams())
    train, val = load_layout(tmp_path / "s")
    return pd.concat([val, train], ignore_index=True)


def test_une_ligne_par_pli_et_taille_avec_tailles_bornees(tmp_path):
    df = all_planets(tmp_path)
    r = learning_curve(df, [5, 10, 999], folds=3, repeats=2)
    assert len(r) == 3 * 2 * 3                                   # plis × répétitions × tailles
    assert set(r["n"]) == {5, 10, 20}                            # 999 borné à la taille du jeu d'entraînement du pli (30*2/3)
    assert r["score"].between(0, 1).all()


def test_deterministe(tmp_path):
    df = all_planets(tmp_path)
    a = learning_curve(df, [8, 16], folds=3, repeats=1)
    b = learning_curve(df, [8, 16], folds=3, repeats=1)
    assert np.allclose(a["score"], b["score"])


def test_resume_gain_apparie(tmp_path):
    df = pd.DataFrame({"repeat": [0, 0, 0, 0], "fold": [0, 1, 0, 1], "n": [10, 10, 20, 20],
                       "score": [0.1, 0.2, 0.3, 0.2]})
    s = summarize(df).set_index("n")
    assert np.isclose(s.loc[10, "score_moyen"], 0.15) and np.isclose(s.loc[20, "score_moyen"], 0.25)
    assert np.isclose(s.loc[20, "gain_vs_precedent"], 0.1) and s.loc[20, "plis_gagnes"] == 0.5   # un pli gagne, un est à égalité
