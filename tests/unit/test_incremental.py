import numpy as np
import pandas as pd

from src.pipelines.incremental import (IncrementalConfig, IncrementalRunner, IncrementalState,
                                       split_ids, update_stopping)


def fake_fetch_factory(calls):
    def fetch(ids):
        calls.append(list(ids))
        return pd.DataFrame({"planet_id": [int(i) for i in ids], "x": 1.0})
    return fetch


def make_runner(tmp_path, scores, n=100, **kw):
    it = iter(scores)
    calls = []
    cfg = IncrementalConfig(batch_size=10, val_size=20, min_train_for_fit=10, patience=2,
                            min_delta=0.01, smooth_window=1, work_dir=str(tmp_path), **kw)
    r = IncrementalRunner(cfg, list(range(1, n + 1)), fake_fetch_factory(calls), lambda t, v: next(it))
    return r, calls


def test_split_deterministe_et_disjoint():
    v1, t1 = split_ids(range(1, 101), 20, seed=1)
    v2, t2 = split_ids(range(1, 101), 20, seed=1)
    assert (v1, t1) == (v2, t2)
    assert not set(v1) & set(t1) and len(v1) == 20 and len(t1) == 80
    assert t1 != sorted(t1, key=int)         # ordre mélangé


def test_arret_sur_patience(tmp_path):
    r, _ = make_runner(tmp_path, [0.1, 0.2, 0.2, 0.2, 0.9])
    st = r.run()
    assert st.stopped and st.batches_done == 4      # 0.1, 0.2 (mieux), 0.2, 0.2 (2 sans progrès)
    assert st.best_smoothed == 0.2


def test_validation_fixe_telechargee_une_fois(tmp_path):
    r, calls = make_runner(tmp_path, [0.1, 0.2, 0.2, 0.2])
    r.run()
    val_calls = [c for c in calls if len(c) == 20]
    assert len(val_calls) == 1
    assert sorted(val_calls[0]) == sorted(r.state.val_ids)


def test_reprise_idempotente(tmp_path):
    r, calls = make_runner(tmp_path, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], max_batches=3)
    r.step(); r.step()
    r2, calls2 = make_runner(tmp_path, [0.3, 0.4, 0.5, 0.6])      # nouveau process, même dossier
    assert r2.state.batches_done == 2 and r2.state.val_ids == r.state.val_ids
    r2.step()
    assert r2.state.batches_done == 3
    assert all(len(c) == 10 for c in calls2)       # ni re-fetch de la validation ni des batchs faits


def test_batch_rejoue_n_ajoute_pas_de_doublons(tmp_path):
    r, _ = make_runner(tmp_path, [0.1, 0.2, 0.3])
    r.step()
    r.state.batches_done = 0                       # simule un crash avant sauvegarde
    r.step()
    assert len(r._load_train(0)) == 10


def test_pas_de_score_avant_min_train(tmp_path):
    r, _ = make_runner(tmp_path, [0.5], n=100)
    r.cfg.min_train_for_fit = 25
    r.step()
    assert r.state.history[0]["score"] is None and r.state.stalled == 0


def test_donnees_epuisees(tmp_path):
    r, _ = make_runner(tmp_path, [0.1 * i for i in range(1, 20)], n=50)   # 30 train -> 3 batchs
    st = r.run()
    assert st.stopped and "épuisées" in st.stop_reason


def test_update_stopping_unit():
    cfg = IncrementalConfig(patience=1, min_delta=0.0, smooth_window=2)
    st = IncrementalState(history=[{"score": 0.5}])
    update_stopping(st, cfg, 0.5)
    assert st.best_smoothed == 0.5 and not st.stopped
    st.history.append({"score": 0.4}); update_stopping(st, cfg, 0.4)
    assert st.stopped


def test_lissage_fenetre_3(tmp_path):
    cfg = IncrementalConfig(patience=5, min_delta=0.0, smooth_window=3)
    st = IncrementalState()
    for sc in [0.0, 0.3, 0.3]:
        st.history.append({"score": sc, "smoothed": None}); update_stopping(st, cfg, sc)
    assert abs(st.history[-1]["smoothed"] - 0.2) < 1e-12     # moyenne de 3, sans doublon


def test_baseline_fit_predict_synthetique():
    from src.pipelines.incremental import baseline_fit_predict
    rng = np.random.default_rng(0)
    def df(n, off):
        d = {"planet_id": np.arange(off, off + n)}
        for i in range(1, 284):
            d[f"Rp_{i}"] = 0.1 + 0.01 * rng.standard_normal(n)
            d[f"wl_{i}"] = 0.01 + 1e-4 * rng.standard_normal(n)
        for c, v in dict(Ms=1.0, Mp=1.0, P=5.0, sma=0.05, i=88.0, Rs=1.0, Ts=5800.0).items():
            d[c] = v * (1 + 0.05 * rng.standard_normal(n))
        return pd.DataFrame(d)
    res = baseline_fit_predict(df(60, 0), df(20, 1000))
    sc = res.score
    assert {'scaler', 'model', 'sigma', 'feat_cols'} <= set(res.model)
    assert np.isfinite(sc) and 0.0 <= sc <= 1.0


def test_max_batches_reprenable(tmp_path):
    r, _ = make_runner(tmp_path, [0.1, 0.2, 0.3, 0.4, 0.5], max_batches=2)
    st = r.run()
    assert st.batches_done == 2 and not st.stopped          # budget, pas un arrêt définitif
    r2, calls2 = make_runner(tmp_path, [0.5, 0.6, 0.7], max_batches=4)
    r2.run()
    assert r2.state.batches_done >= 3
    assert all(len(c) == 10 for c in calls2)                 # rien de refetché


def test_reprise_refuse_batch_size_different(tmp_path):
    import pytest
    make_runner(tmp_path, [0.1])[0].step()
    cfg = IncrementalConfig(batch_size=25, val_size=20, work_dir=str(tmp_path))
    with pytest.raises(ValueError, match="incompatible"):
        IncrementalRunner(cfg, list(range(1, 101)), lambda ids: None, lambda a, b: 0.0)


def test_modele_sauvegarde_seulement_sur_nouveau_record(tmp_path):
    from src.pipelines.incremental import FitResult
    scores = iter([0.1, 0.3, 0.2, 0.25, 0.4])
    saved = []
    cfg = IncrementalConfig(batch_size=10, val_size=20, min_train_for_fit=10, patience=9,
                            min_delta=0.0, smooth_window=1, work_dir=str(tmp_path))
    r = IncrementalRunner(cfg, list(range(1, 101)), fake_fetch_factory([]),
                          lambda t, v: FitResult(next(scores), {"n": len(t)}),
                          save_fn=lambda m, rec: saved.append((m["n"], rec["score"])))
    for _ in range(5):
        r.step()
    assert saved == [(10, 0.1), (20, 0.3), (50, 0.4)]     # 0.2 et 0.25 ne sont pas des records


def test_reprise_refuse_binning_different(tmp_path):
    import pytest
    make_runner(tmp_path, [0.1])[0].step()
    cfg = IncrementalConfig(batch_size=10, val_size=20, work_dir=str(tmp_path), binning_airs=9)
    with pytest.raises(ValueError, match="spectres"):
        IncrementalRunner(cfg, list(range(1, 101)), lambda ids: None, lambda a, b: 0.0)


def test_write_reports(tmp_path):
    from src.pipelines.incremental import write_reports
    st = IncrementalState(history=[{"batch": 0, "n_train": 10, "score": 0.2, "smoothed": 0.2}], best_smoothed=0.2)
    write_reports(st, str(tmp_path))
    import json
    assert json.loads((tmp_path / "metrics.json").read_text())["best_smoothed_gll"] == 0.2
    assert (tmp_path / "history.csv").exists()


def test_make_model_saver_ecrit_et_relit(tmp_path):
    import joblib, json
    from src.pipelines.incremental import make_model_saver
    cfg = IncrementalConfig(work_dir=str(tmp_path / "w"))
    save = make_model_saver(cfg, models_dir=str(tmp_path / "champ"))
    save({"a": 1}, {"n_train": 30, "score": 0.2, "smoothed": 0.2, "batch": 2})
    assert joblib.load(tmp_path / "champ" / "bundle.joblib") == {"a": 1}
    meta = json.loads((tmp_path / "champ" / "meta.json").read_text())
    assert meta["n_train"] == 30 and "git_commit" in meta and "spectra_sha256" in meta


def test_migration_ancien_etat_max_batches(tmp_path):
    r, _ = make_runner(tmp_path, [0.1, 0.2, 0.3, 0.4, 0.5], max_batches=1)
    r.run()
    r.state.stopped, r.state.stop_reason = True, "max_batches atteint"   # ancien format persisté
    r.state.save(r.state_path)
    r2, _ = make_runner(tmp_path, [0.2, 0.3, 0.4], max_batches=3)
    assert not r2.state.stopped
    r2.run()
    assert r2.state.batches_done == 3
