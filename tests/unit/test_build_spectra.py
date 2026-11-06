"""Tests : découpage figé, reconstruction des spectres depuis les courbes, traçabilité."""
import json

import numpy as np
import pandas as pd
import pytest

from spectra_factory import make_world
from src.pipelines.build_spectra import (build_layout, curves_digest, export_split, extraction_level, load_split,
                                         quality_filter)
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


def test_extension_du_decoupage_garde_validation_et_ordre(tmp_path):
    from src.pipelines.build_spectra import extend_split
    split, curves, raw = make_world(tmp_path, skip_curves=(1005,))
    old = {"val": split["val"], "train": split["train"][:10], "batch_size": 10, "source": "v1"}
    labels = [1000 + i for i in range(30)] + [9999]                           # 9999 : pas de courbes
    new = extend_split(old, curves, labels)
    assert new["val"] == old["val"] and new["train"][:10] == old["train"]       # début inchangé
    assert new["train"][10:] == [p for p in split["train"][10:] if p != 1005]   # ajout dans l'ordre des labels, sans doublon
    assert 1005 not in new["train"] and 9999 not in new["train"]                # sans courbes : ignorées
    assert new["n_added"] == len(new["train"]) - 10 and not set(new["val"]) & set(new["train"])
    assert extend_split(new, curves, labels)["n_added"] == 0                    # idempotent


def test_extension_refuse_d_ecraser_la_source(tmp_path, monkeypatch):
    import sys

    from src.pipelines import build_spectra
    split, curves, raw = make_world(tmp_path)
    src = tmp_path / "split_v1.json"
    src.write_text(json.dumps(split))
    monkeypatch.setattr(sys, "argv", ["x", "--extend-split", str(src), "--split", str(src), "--curves", str(curves),
                                      "--raw", str(raw)])
    with pytest.raises(SystemExit, match="NOUVEAU"):
        build_spectra.main()
    out = tmp_path / "split_v2.json"
    monkeypatch.setattr(sys, "argv", ["x", "--extend-split", str(src), "--split", str(out), "--curves", str(curves),
                                      "--raw", str(raw)])
    build_spectra.main()
    assert json.loads(out.read_text())["val"] == split["val"] and json.loads(src.read_text()) == split


def _monde_avec_labels_faux(tmp_path, bad_train, bad_val=None):
    split, curves, raw = make_world(tmp_path)
    lab = pd.read_csv(raw / "train.csv")
    cols = [f"wl_{i}" for i in range(1, 284)]
    for pid, k in [(bad_train, 4.0)] + ([(bad_val, 0.1)] if bad_val else []):
        lab.loc[lab.planet_id == pid, cols] *= 1 / k             # label k fois plus petit => niveau = k
    lab.to_csv(raw / "train.csv", index=False)
    return split, curves, raw


def test_niveau_d_extraction_proche_de_un_sur_donnees_saines(tmp_path):
    split, curves, raw = make_world(tmp_path)
    df = pd.read_csv(build_layout(split, curves, raw, tmp_path / "o", P) and tmp_path / "o" / "val.csv")
    lev = extraction_level(df)
    assert ((lev > 0.8) & (lev < 1.3)).all()


def test_filtre_qualite_exclut_de_l_entrainement_mais_pas_de_la_validation(tmp_path):
    bad_tr, bad_va = 1020, 1001
    split, curves, raw = _monde_avec_labels_faux(tmp_path, bad_tr, bad_va)
    out = tmp_path / "o"
    m0 = build_layout(split, curves, raw, tmp_path / "o0", P)
    m = build_layout(split, curves, raw, out, P, level_range=(0.5, 2.0))
    assert set(m["excluded_train"]) == {bad_tr} and m["excluded_train"][bad_tr] > 3
    assert m["n_train"] == m0["n_train"] - 1
    assert m["n_val"] == m0["n_val"]                               # validation intacte
    assert set(m["val_out_of_range"]) == {bad_va}                  # mais signalée
    ids = pd.concat([pd.read_csv(f) for f in (out / "batches").glob("batch_*.csv")])["planet_id"]
    assert bad_tr not in set(ids) and bad_va in set(pd.read_csv(out / "val.csv")["planet_id"])
    assert m0["excluded_train"] == {} and m0["level_range"] is None   # sans option : comportement inchangé


def test_filtre_qualite_cas_limites():
    df = pd.DataFrame({"planet_id": [], **{f"Rp_{i}": [] for i in range(1, 284)}})
    assert quality_filter(df, (0.5, 2.0))[1] == {}
