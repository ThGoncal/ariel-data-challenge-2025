"""Tests de la courbe d'apprentissage : plis communs, sous-ensembles imbriqués, résumé apparié."""
import numpy as np
import pandas as pd

from scripts.learning_curve import default_sizes, learning_curve, plot_curve, summarize
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


def test_tailles_communes_a_tous_les_plis_meme_si_plis_inegaux(tmp_path):
    df = all_planets(tmp_path)[:29]                              # 29 planètes, 5 plis de tailles inégales (5 ou 6 en test)
    r = learning_curve(df, [10, 23, 24, 100], folds=5, repeats=2)
    assert r.groupby("n").size().nunique() == 1                  # même nombre de plis pour chaque taille
    assert r["n"].max() == 29 - 6                                # plafond = plus petit entraînement d'un pli, pas de doublon


def test_tailles_par_defaut():
    s = default_sizes(1000, folds=5)
    assert s[0] == 20 and s[-1] == 800 and s == sorted(set(s))
    assert all(b > a for a, b in zip(s, s[1:]))
    assert default_sizes(30, folds=5)[-1] == 30 - 6


def test_trace_ecrit_un_png(tmp_path):
    s = pd.DataFrame({"n": [20, 40, 80], "score_moyen": [0.25, 0.3, 0.33], "ecart_type_plis": [0.1, 0.08, 0.06]})
    out = plot_curve(s, tmp_path / "docs" / "c.png", n_total=100)
    assert out.exists() and out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
