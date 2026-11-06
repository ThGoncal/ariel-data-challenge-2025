"""Tests du sous-espace PCA : projection, détection hors distribution sans label, plis sans fuite."""
import numpy as np
import pandas as pd

from scripts.compare_pca import (REFERENCE, auc, config_names, fold_predictions, level_of, ood_distances, pca_basis,
                                 project, run_cv, summarize)
from scripts.compare_extraction import fold_splits
from scripts.compare_models import to_arrays
from spectra_factory import make_world
from src.pipelines.build_spectra import build_layout
from src.pipelines.train_champion import load_layout
from src.signal_processing.spectra import ExtractionParams


def planets(tmp_path):
    split, curves, raw = make_world(tmp_path)
    build_layout(split, curves, raw, tmp_path / "s", ExtractionParams())
    train, val = load_layout(tmp_path / "s")
    return to_arrays(pd.concat([val, train], ignore_index=True))


def test_projection_exacte_dans_le_sous_espace_et_distance_hors_sous_espace():
    rng = np.random.default_rng(0)
    comps = rng.normal(size=(2, 283))
    comps -= comps.mean(axis=1, keepdims=True)
    Y = 0.01 + rng.normal(size=(40, 2)) * 1e-3 @ comps
    V = pca_basis(Y, 2)
    rec, d = project(Y[:5], V)
    assert np.allclose(rec, Y[:5], atol=1e-9) and d.max() < 1e-9
    noisy = Y[:5] + rng.normal(size=(5, 283)) * 1e-3
    assert project(noisy, V)[1].min() > 1e-3                      # du bruit hors sous-espace est mesuré
    assert np.allclose(project(noisy, V)[0].mean(axis=1), noisy.mean(axis=1))      # le niveau est conservé


def test_niveau_d_extraction_proche_de_un(tmp_path):
    X, y, _ = planets(tmp_path)
    lev = level_of(X, y)
    assert 0.9 < np.median(lev) < 1.2 and ((lev > 0.5) & (lev < 2.0)).all()


def test_variantes_et_plis_apparies(tmp_path):
    X, y, ids = planets(tmp_path)
    splits = fold_splits(len(X), 3, 1, 0)
    out = fold_predictions(X[splits[0][0]], y[splits[0][0]], X[splits[0][1]], ks=(3,))
    assert set(out) == set(config_names((3,)))
    assert all(p.mu.shape == (len(splits[0][1]), 283) and (p.mu >= 0).all() for p in out.values())
    t = summarize(run_cv(X, y, ids, splits, ks=(2,)))
    assert REFERENCE in set(t["variante"]) and t.loc[t.variante == REFERENCE, "diff_moy"].iloc[0] == 0.0
    assert t["cv_moy"].between(0, 1).all()


def test_une_planete_au_spectre_embrouille_est_la_plus_eloignee(tmp_path):
    X, y, ids = planets(tmp_path)
    X = X.copy()
    X[3, :283] = X[3, :283] * (1 + 0.5 * np.sin(np.arange(283) / 5))      # forme déformée
    d = ood_distances(X, y, ids, fold_splits(len(X), 5, 2, 0), k=2)
    assert d.loc[d.dist_rel.idxmax(), "planet_id"] == ids[3]
    flag = (d.planet_id == ids[3]).to_numpy()
    assert auc(d.dist_rel.to_numpy(), flag) == 1.0


def test_auc_cas_limites():
    s = np.array([0.1, 0.2, 0.3, 0.4])
    assert auc(s, np.array([False, False, True, True])) == 1.0
    assert auc(s, np.array([True, True, False, False])) == 0.0
    assert np.isnan(auc(s, np.zeros(4, bool)))
