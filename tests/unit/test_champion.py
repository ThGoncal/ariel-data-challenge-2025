"""Tests du modèle champion et de son entraînement/enregistrement."""
import numpy as np
import pandas as pd
import pytest

from spectra_factory import make_world
from src.models.champion import feature_matrix, fit_champion, predict_champion, score_champion
from src.pipelines.build_spectra import build_layout
from src.pipelines.train_champion import load_layout, run
from src.signal_processing.spectra import ExtractionParams

PARAMS = {"extraction": {"method": "adaptee", "deg": 1, "edge_channels": 10},
          "champion": {"alpha": 10.0, "power": 2, "cv_folds": 3, "cv_repeats": 1}}


@pytest.fixture
def layout(tmp_path):
    split, curves, raw = make_world(tmp_path)
    out = tmp_path / "spectra"
    build_layout(split, curves, raw, out, ExtractionParams())
    return out


def test_features_en_profondeur(layout):
    train, _ = load_layout(layout)
    X1, _ = feature_matrix(train, 1)
    X2, _ = feature_matrix(train, 2)
    assert np.allclose(X2[:, :283], X1[:, :283] ** 2) and np.array_equal(X1[:, 283:], X2[:, 283:])


def test_ajustement_prediction_et_score(layout):
    train, val = load_layout(layout)
    b = fit_champion(train)
    p = predict_champion(b, val)
    assert p.mu.shape == (len(val), 283) and (p.mu >= 0).all() and (p.sigma > 0).all()
    assert 0.0 <= score_champion(b, val) <= 1.0
    assert b["power"] == 2 and b["n_train"] == len(train)


def test_prediction_refuse_des_features_manquantes(layout):
    train, val = load_layout(layout)
    b = fit_champion(train)
    bad = val.copy()
    bad.loc[bad.index[0], "Rp_10"] = np.nan
    with pytest.raises(ValueError):
        predict_champion(b, bad)


def test_run_local_ecrit_modele_meta_et_metriques(layout, tmp_path):
    m = run(PARAMS, layout, tmp_path / "models", tmp_path / "reports")
    assert {"gll_val", "gll_val_ci05", "gll_val_ci95", "gll_cv_mean", "gll_cv_sd", "n_train"} <= set(m)
    assert m["gll_val_ci05"] <= m["gll_val_ci95"]
    assert (tmp_path / "models" / "bundle.joblib").exists()
    meta = pd.read_json(tmp_path / "models" / "meta.json", typ="series")
    assert meta["extraction_method"] == "adaptee" and meta["power"] == 2 and "curves_sha256" in meta.index
    assert (tmp_path / "reports" / "metrics.json").exists()


@pytest.mark.slow
def test_enregistrement_mlflow_et_rechargement(layout, tmp_path):
    mlflow = pytest.importorskip("mlflow")
    from pathlib import Path
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    exp = mlflow.create_experiment("t", artifact_location=str(tmp_path / "artifacts"))
    mlflow.set_experiment(experiment_id=exp)
    import src.pipelines.train_champion as tc
    orig = mlflow.set_experiment
    mlflow.set_experiment = lambda *a, **k: orig(experiment_id=exp)       # le test impose son expérience locale
    try:
        m = run(PARAMS, layout, tmp_path / "models", tmp_path / "reports", use_mlflow=True, register="test-champion",
                cv=False, src_dir=Path(tc.__file__).resolve().parents[1])
    finally:
        mlflow.set_experiment = orig
    assert m["registered_version"] == "1"
    loaded = mlflow.pyfunc.load_model("models:/test-champion@champion")
    _, val = load_layout(layout)
    out = loaded.predict(val)
    assert out.shape == (len(val), 567) and list(out["planet_id"]) == list(val["planet_id"])
    assert (out.filter(like="_std") > 0).all().all()


def test_sigma_constant_par_defaut_et_compatibilite_des_anciens_modeles(layout):
    train, val = load_layout(layout)
    b = fit_champion(train)
    assert b["sigma_mode"] == "const" and b["sigma_fit"] is None
    p = predict_champion(b, val)
    assert np.allclose(p.sigma, p.sigma[0])                                     # identique pour toutes les planètes
    old = {k: v for k, v in b.items() if k not in ("sigma_mode", "sigma_fit")}   # bundle écrit par la v0.2.0
    assert np.array_equal(predict_champion(old, val).sigma, p.sigma)


def test_sigma_proportionnel_a_la_profondeur(layout):
    from src.models.champion import depth_of
    train, val = load_layout(layout)
    b = fit_champion(train, sigma="depth", gamma=1.0)
    p = predict_champion(b, val)
    d = depth_of(feature_matrix(val, 2)[0], 2)
    i, j = int(np.argmin(d)), int(np.argmax(d))
    assert d[j] > d[i] and np.allclose(p.sigma[j] / p.sigma[i], d[j] / d[i])   # γ=1 : σ ∝ profondeur
    assert (p.sigma > 0).all() and b["sigma_fit"]["gamma"] == 1.0
    g0 = predict_champion(fit_champion(train, sigma="depth", gamma=0.0), val)   # γ=0 : redonne un σ constant
    assert np.allclose(g0.sigma, g0.sigma[0])
    with pytest.raises(ValueError):
        fit_champion(train, sigma="magique")


def test_sigma_profondeur_identique_a_l_etude_comparative(layout):
    from scripts.compare_sigma import fit_depth_sigma as ref_fit
    from scripts.compare_sigma import sigma_matrix as ref_sigma
    from src.models.champion import depth_sigma, fit_depth_sigma
    rng = np.random.default_rng(0)
    resid, d = rng.normal(0, 1e-3, (40, 283)), rng.uniform(0.004, 0.04, 40)
    for gamma in (None, 0.0, 1.0):
        a, b = fit_depth_sigma(resid, d, gamma), ref_fit(resid, d, gamma)
        assert np.isclose(a["a"], b["a"]) and np.isclose(a["gamma"], b["gamma"]) and np.allclose(a["q"], b["q"])
        assert np.allclose(depth_sigma(a, d[:5]), ref_sigma(b, d[:5]))


def test_run_avec_sigma_profondeur_ecrit_la_meta(layout, tmp_path):
    params = {**PARAMS, "champion": {**PARAMS["champion"], "sigma": "depth"}}
    m = run(params, layout, tmp_path / "models", tmp_path / "reports")
    meta = pd.read_json(tmp_path / "models" / "meta.json", typ="series")
    assert meta["sigma"] == "depth" and 0.0 <= meta["gamma"] <= 2.0 and np.isfinite(m["gll_cv_mean"])
