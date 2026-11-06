"""Tests de la comparaison de σ en validation croisée appariée."""
import numpy as np
import pandas as pd

from scripts.compare_models import to_arrays
from scripts.compare_sigma_cv import CONFIGS, REFERENCE, fold_predictions, run_cv, summarize
from scripts.compare_extraction import fold_splits
from spectra_factory import make_world
from src.models.champion import fit_champion, predict_champion
from src.pipelines.build_spectra import build_layout
from src.pipelines.train_champion import load_layout
from src.signal_processing.spectra import ExtractionParams


def world(tmp_path):
    split, curves, raw = make_world(tmp_path)
    build_layout(split, curves, raw, tmp_path / "s", ExtractionParams())
    train, val = load_layout(tmp_path / "s")
    df = pd.concat([val, train], ignore_index=True)
    X, y, ids = to_arrays(df)
    return df, X, y, ids


def test_mu_commun_et_facteurs_exacts(tmp_path):
    _, X, y, _ = world(tmp_path)
    a, b = np.arange(0, 22), np.arange(22, 30)
    p = fold_predictions(X[a], y[a], X[b])
    mus = [v.mu for v in p.values()]
    assert all(np.array_equal(mus[0], m) for m in mus)                       # seule σ change
    assert np.allclose(p["const · insample × 2"].sigma, 2 * p[REFERENCE].sigma)
    assert all((v.sigma > 0).all() and v.sigma.shape == v.mu.shape for v in p.values())
    assert (p["const · loo"].sigma >= p[REFERENCE].sigma * 0.5).all()


def test_reference_equivalente_au_champion(tmp_path):
    df, X, y, _ = world(tmp_path)
    a, b = np.arange(0, 22), np.arange(22, 30)
    ref = fold_predictions(X[a], y[a], X[b])[REFERENCE]
    champ = predict_champion(fit_champion(df.iloc[a]), df.iloc[b])
    assert np.allclose(ref.mu, champ.mu, atol=1e-9)
    assert np.allclose(ref.sigma, champ.sigma, rtol=0.02)                    # écart-type vs rms des résidus (moyenne ≈ 0)


def test_cv_appariee_et_resume(tmp_path):
    _, X, y, ids = world(tmp_path)
    splits = fold_splits(len(X), 3, 1, 0)
    s = run_cv(X, y, ids, splits)
    assert set(s) == {c[0] for c in CONFIGS} and all(len(v) == 3 for v in s.values())
    t = summarize(s).set_index("σ")
    assert abs(t.loc[REFERENCE, "diff_moy"]) < 1e-12 and t.loc[REFERENCE, "plis_gagnes"] == 0.0
    assert np.isfinite(t.to_numpy(dtype=float)).all()
