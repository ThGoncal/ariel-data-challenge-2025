"""Tests de la comparaison de seuils : test commun, mêmes plis, tailles d'entraînement."""
import numpy as np
import pandas as pd

from scripts.compare_extraction import fold_splits
from scripts.compare_filter import in_range, range_name, run_cv, summarize
from scripts.compare_models import to_arrays
from scripts.compare_pca import level_of
from spectra_factory import make_world
from src.pipelines.build_spectra import build_layout
from src.pipelines.train_champion import load_layout
from src.signal_processing.spectra import ExtractionParams


def planets(tmp_path):
    split, curves, raw = make_world(tmp_path)
    build_layout(split, curves, raw, tmp_path / "s", ExtractionParams())
    train, val = load_layout(tmp_path / "s")
    return to_arrays(pd.concat([val, train], ignore_index=True))


def test_plages_noms_et_masques():
    lev = np.array([0.4, 0.8, 1.0, 1.6, 3.0])
    assert in_range(lev, None).all()
    assert list(in_range(lev, (0.5, 2.0))) == [False, True, True, True, False]
    assert range_name(None) == "sans filtre" and range_name((0.5, 2.0)) == "[0.5 ; 2]"


def test_test_commun_et_tailles_d_entrainement(tmp_path):
    X, y, ids = planets(tmp_path)
    lev = level_of(X, y)
    ranges = [None, (0.5, 2.0), (0.95, 1.1)]
    splits = fold_splits(len(X), 3, 1, 0)
    scores, sizes = run_cv(X, y, ids, splits, ranges, eval_range=(0.5, 2.0))
    assert set(scores) == {"sans filtre", "[0.5 ; 2]", "[0.95 ; 1.1]"}
    assert all(len(d["commun"]) == 3 and len(d["toutes"]) == 3 for d in scores.values())
    assert not ((lev < 0.5) | (lev > 2)).any()                                       # synthétique : aucun échec d'extraction
    assert sizes["sans filtre"] == sizes["[0.5 ; 2]"] > sizes["[0.95 ; 1.1]"]        # plage plus étroite : moins de planètes
    t = summarize(scores, sizes, "[0.5 ; 2]")
    assert t.loc[t.plage == "[0.5 ; 2]", "diff_moy"].iloc[0] == 0.0


def test_un_echec_d_extraction_dans_le_test_n_affecte_que_cv_toutes(tmp_path):
    X, y, ids = planets(tmp_path)
    X = X.copy()
    X[5, :283] = X[5, :283] * 3.4                                   # échec d'extraction : amplitude ×3,4 → hors --eval-range
    assert level_of(X, y)[5] > 2
    scores, sizes = run_cv(X, y, ids, fold_splits(len(X), 5, 2, 0), [(0.5, 2.0)], (0.5, 2.0))
    d = scores["[0.5 ; 2]"]
    assert d["toutes"].min() < d["commun"].min()                    # le pli contenant la planète ratée est pénalisé sans elle exclue
