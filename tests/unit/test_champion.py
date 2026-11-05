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
