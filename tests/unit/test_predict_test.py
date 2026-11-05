"""Tests de l'inférence : mise en forme, format de soumission, contrôle de cohérence brut -> entraînement."""
import numpy as np
import pandas as pd
import pytest

from spectra_factory import make_world
from src.models.champion import fit_champion
from src.pipelines.build_spectra import build_layout
from src.pipelines.predict_test import (align_to_sample, check_submission, consistency, gll_per_planet, inference_frame,
                                        predict_submission, read_reference)
from src.pipelines.train_champion import load_layout
from src.signal_processing.spectra import ExtractionParams, extract_spectrum

P = ExtractionParams()


@pytest.fixture
def world(tmp_path):
    split, curves, raw = make_world(tmp_path)
    out = tmp_path / "spectra"
    build_layout(split, curves, raw, out, P)
    train, _ = load_layout(out)
    return {"split": split, "curves": curves, "raw": raw, "out": out, "bundle": fit_champion(train)}


def depths_from_curves(world, ids):
    d = {}
    for pid in ids:
        z = np.load(world["curves"] / f"{pid}.npz")
        d[pid] = extract_spectrum(z["airs"], z["fgs"], P)
    return d


def test_chaine_depuis_les_courbes_identique_a_l_entrainement(world):
    ids = [world["split"]["val"][0], world["split"]["train"][0]]
    stars = pd.read_csv(world["raw"] / "train_star_info.csv")
    fresh = inference_frame(depths_from_curves(world, ids), stars)
    ref = read_reference(world["out"], ids)
    assert list(ref["where"]) == ["val", "train"]
    c = consistency(fresh, ref, world["bundle"])
    assert c["max_rel_rp"] < 1e-9 and c["max_abs_dmu"] < 1e-12


def test_le_controle_detecte_une_extraction_differente(world):
    pid = world["split"]["val"][0]
    stars = pd.read_csv(world["raw"] / "train_star_info.csv")
    z = np.load(world["curves"] / f"{pid}.npz")
    other = extract_spectrum(z["airs"], z["fgs"], ExtractionParams("fixe", 1, 0))       # autre extraction
    fresh = inference_frame({pid: other}, stars)
    c = consistency(fresh, read_reference(world["out"], [pid]), world["bundle"])
    assert c["max_rel_rp"] > 1e-3


def test_planete_sans_astro_ou_colonne_manquante(world):
    stars = pd.read_csv(world["raw"] / "train_star_info.csv")
    d = depths_from_curves(world, [world["split"]["val"][0]])
    with pytest.raises(ValueError, match="absente"):
        inference_frame({999999: next(iter(d.values()))}, stars)
    with pytest.raises(ValueError, match="colonnes absentes"):
        inference_frame(d, stars.drop(columns=["Ms"]))


def test_reference_refuse_une_planete_inconnue(world):
    with pytest.raises(ValueError, match="absentes"):
        read_reference(world["out"], [424242])


def test_soumission_conforme_et_verifications(world):
    ids = world["split"]["val"][:2]
    stars = pd.read_csv(world["raw"] / "train_star_info.csv")
    sub = predict_submission(world["bundle"], inference_frame(depths_from_curves(world, ids), stars))
    sample = sub.copy()
    assert sub.shape == (2, 567) and check_submission(sub, sample) == []
    bad = sub.copy(); bad.loc[0, "wl_5"] = -1.0
    assert any("négatives" in p for p in check_submission(bad, sample))
    bad = sub.copy(); bad.loc[0, "wl_5"] = np.nan
    assert any("manquantes" in p for p in check_submission(bad, sample))
    bad = sub.copy(); bad["planet_id"] = [1, 2]
    assert any("planet_id" in p for p in check_submission(bad, sample))
    assert any("colonnes" in p for p in check_submission(sub.drop(columns="wl_9"), sample))
    bad = sub.copy(); bad.loc[0, "wl_3_std"] = 0.0
    assert any("écart-type" in p for p in check_submission(bad, sample))


def test_gll_par_planete_avec_labels(world):
    ids = world["split"]["val"][:2]
    ref = read_reference(world["out"], ids)
    labels = pd.read_csv(world["raw"] / "train.csv")
    g = gll_per_planet(world["bundle"], ref, labels)
    assert set(g) == set(ids) and all(0.0 <= v <= 1.0 for v in g.values())


def test_alignement_sur_l_entete_officielle(world):
    ids = world["split"]["val"][:1]
    stars = pd.read_csv(world["raw"] / "train_star_info.csv")
    sub = predict_submission(world["bundle"], inference_frame(depths_from_curves(world, ids), stars))
    official = ["planet_id"] + [f"wl_{i}" for i in range(1, 284)] + [f"sigma_{i}" for i in range(1, 284)]
    sample = pd.DataFrame([[ids[0]] + [0.456] * 566], columns=official)
    out = align_to_sample(sub, sample)
    assert list(out.columns) == official and check_submission(out, sample) == []
    assert np.array_equal(out["sigma_7"].to_numpy(), sub["wl_7_std"].to_numpy())      # mêmes valeurs, autre nom
    assert align_to_sample(out, sample) is out                                       # déjà conforme : inchangé
    with pytest.raises(ValueError, match="incompatibles"):
        align_to_sample(sub.drop(columns="wl_283_std"), sample)                       # mauvais nombre de colonnes
    with pytest.raises(ValueError, match="incompatibles"):
        align_to_sample(sub.rename(columns={"wl_5": "x"}), sample)                    # bloc des moyennes différent
