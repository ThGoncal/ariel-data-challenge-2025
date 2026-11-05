"""Tests : découpage figé, reconstruction des spectres depuis les courbes, traçabilité."""
import json

import numpy as np
import pandas as pd
import pytest

from spectra_factory import make_world
from src.pipelines.build_spectra import build_layout, curves_digest, export_split, load_split
from src.signal_processing.spectra import ExtractionParams, depth_to_rp, extract_spectrum

P = ExtractionParams("adaptee", 1, 10)


def test_export_et_chargement_du_decoupage(tmp_path):
    (tmp_path / "batches").mkdir()
    pd.DataFrame({"planet_id": [5, 6]}).to_csv(tmp_path / "val.csv", index=False)
    pd.DataFrame({"planet_id": [1, 2, 6]}).to_csv(tmp_path / "batches" / "batch_000.csv", index=False)
    pd.DataFrame({"planet_id": [3, 1]}).to_csv(tmp_path / "batches" / "batch_001.csv", index=False)
    s = export_split(tmp_path)
    assert s["val"] == [5, 6] and s["train"] == [1, 2, 3]            # validation exclue de l'entraînement, ordre gardé
    (tmp_path / "s.json").write_text(json.dumps({"val": [1], "train": [1, 2]}))
    with pytest.raises(ValueError):
        load_split(tmp_path / "s.json")


def test_spectres_reconstruits_depuis_les_courbes(tmp_path):
    split, curves, raw = make_world(tmp_path)
    out = tmp_path / "out"
    m = build_layout(split, curves, raw, out, P)
    assert m["n_val"] == 8 and m["n_train"] == 22 and m["missing"] == []
    val = pd.read_csv(out / "val.csv")
    assert list(val["planet_id"]) == split["val"]
    assert len(list((out / "batches").glob("batch_*.csv"))) == 3     # 22 planètes par lots de 10
    pid = split["val"][0]
    z = np.load(curves / f"{pid}.npz")
    expected = depth_to_rp(extract_spectrum(z["airs"], z["fgs"], P))
    got = val.loc[val.planet_id == pid, [f"Rp_{i}" for i in range(1, 284)]].to_numpy()[0]
    assert np.allclose(got, expected)
    assert {"wl_1", "wl_283", "Ms", "Mp", "P", "sma", "i", "Rs", "Ts"} <= set(val.columns)


def test_planetes_sans_courbes_ignorees_ou_erreur_en_mode_strict(tmp_path):
    split, curves, raw = make_world(tmp_path, skip_curves=(1003, 1015))
    m = build_layout(split, curves, raw, tmp_path / "out", P)
    assert sorted(m["missing"]) == [1003, 1015] and m["n_val"] + m["n_train"] == 28
    with pytest.raises(RuntimeError):
        build_layout(split, curves, raw, tmp_path / "out2", P, strict=True)


def test_lots_perimes_supprimes(tmp_path):
    split, curves, raw = make_world(tmp_path)
    out = tmp_path / "out"
    build_layout(split, curves, raw, out, P)
    (out / "batches" / "batch_099.csv").write_text("planet_id\n1\n")
    build_layout(split, curves, raw, out, P)
    assert not (out / "batches" / "batch_099.csv").exists()


def test_empreinte_des_courbes_change_si_une_courbe_change(tmp_path):
    split, curves, raw = make_world(tmp_path)
    ids = split["val"] + split["train"]
    d1 = curves_digest(curves, ids)
    assert d1 == curves_digest(curves, ids)
    z = dict(np.load(curves / "1000.npz"))
    z["airs"] = z["airs"] * 1.001
    np.savez(curves / "1000.npz", **z)
    assert curves_digest(curves, ids) != d1
