"""Tests du retraitement : liste de planètes, écriture atomique, boucle reprenable et tolérante aux échecs."""
import numpy as np
import pandas as pd
import pytest

from scripts.reprocess_curves import N_WL, planet_list, run, save_curves, validate_curves


def curves(n=200):
    return np.ones((n, N_WL)), np.ones(n), np.ones(N_WL + 1)


def test_liste_validation_d_abord_sans_doublon(tmp_path):
    (tmp_path / "batches").mkdir()
    pd.DataFrame({"planet_id": [5, 6]}).to_csv(tmp_path / "val.csv", index=False)
    pd.DataFrame({"planet_id": [1, 2, 6]}).to_csv(tmp_path / "batches" / "batch_000.csv", index=False)
    pd.DataFrame({"planet_id": [3, 1]}).to_csv(tmp_path / "batches" / "batch_001.csv", index=False)
    assert planet_list(tmp_path) == ["5", "6", "1", "2", "3"]
    assert planet_list(tmp_path, ids=["9", "8"]) == ["9", "8"]


def test_repli_sur_les_labels_si_pas_de_csv(tmp_path):
    pd.DataFrame({"planet_id": [10, 11]}).to_csv(tmp_path / "train.csv", index=False)
    assert planet_list(tmp_path / "absent", tmp_path / "train.csv") == ["10", "11"]


def test_sauvegarde_atomique_et_relisible(tmp_path):
    airs, fgs, rp = curves()
    save_curves(tmp_path / "1.npz", airs, fgs, rp)
    z = np.load(tmp_path / "1.npz")
    assert z["airs"].shape == (200, N_WL) and z["rp"].shape == (N_WL + 1,)
    assert not list(tmp_path.glob("*.tmp.npz"))


@pytest.mark.parametrize("bad", ["forme", "nan", "courte"])
def test_validation_refuse_les_courbes_suspectes(bad):
    airs, fgs, rp = curves()
    if bad == "forme":
        airs = airs[:, :100]
    elif bad == "nan":
        airs[3, 3] = np.nan
    else:
        airs, fgs = airs[:10], fgs[:10]
    with pytest.raises(ValueError):
        validate_curves(airs, fgs, rp)


def make_run(tmp_path, fail=(), no_download=(), **kw):
    saved, cleaned = [], []

    def download(p):
        return p not in no_download

    def process(p):
        if p in fail:
            raise RuntimeError("boom")
        return {"airs": 1, "fgs": 1, "rp": 1}

    stats = run(["a", "b", "c", "d"], download, process, lambda p, r: saved.append(p), lambda p: cleaned.append(p), **kw)
    return stats, saved, cleaned


@pytest.mark.parametrize("prefetch", [True, False])
def test_un_echec_n_arrete_pas_la_nuit_et_le_brut_est_nettoye(tmp_path, prefetch):
    stats, saved, cleaned = make_run(tmp_path, fail={"b"}, no_download={"c"}, prefetch=prefetch)
    assert saved == ["a", "d"] and stats["failed"] == ["b", "c"] and stats["stopped"] is None
    assert set(cleaned) >= {"a", "b", "c", "d"}


def test_arret_apres_trop_d_echecs_consecutifs(tmp_path):
    stats, saved, _ = make_run(tmp_path, fail={"a", "b", "c", "d"}, max_consecutive_failures=2)
    assert saved == [] and "échecs consécutifs" in stats["stopped"] and stats["failed"] == ["a", "b"]


def test_arret_si_disque_plein(tmp_path):
    stats, saved, _ = make_run(tmp_path, free_gb=lambda: 1.0, min_free_gb=5.0)
    assert saved == [] and "disque" in stats["stopped"]


FILES = ["a.parquet", "cal/dark.parquet"]


def put(d, names=FILES, content=b"x"):
    for n in names:
        (d / n).parent.mkdir(parents=True, exist_ok=True)
        (d / n).write_bytes(content)


def test_is_complete(tmp_path):
    from scripts.reprocess_curves import is_complete
    put(tmp_path / "p1")
    assert is_complete(tmp_path / "p1", FILES)
    put(tmp_path / "p2", FILES[:1])
    assert not is_complete(tmp_path / "p2", FILES)                       # fichier manquant
    put(tmp_path / "p3", content=b"")
    assert not is_complete(tmp_path / "p3", FILES)                       # fichiers vides (copie interrompue)
    assert not is_complete(tmp_path / "absent", FILES)


def test_locate_local_accepte_les_deux_dispositions(tmp_path):
    from scripts.reprocess_curves import locate_local
    put(tmp_path / "disque" / "train" / "111")                           # racine/train/<id>
    put(tmp_path / "plat" / "222")                                       # racine/<id>
    assert locate_local(tmp_path / "disque", "111", FILES) == (tmp_path / "disque", "train")
    assert locate_local(tmp_path / "plat", "222", FILES) == (tmp_path, "plat")
    assert locate_local(tmp_path / "plat", "111", FILES) is None
    put(tmp_path / "disque" / "train" / "333", FILES[:1])
    assert locate_local(tmp_path / "disque", "333", FILES) is None       # incomplet : jamais utilisé
