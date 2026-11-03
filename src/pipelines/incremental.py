"""
incremental.py — Apprentissage incrémental par batchs (Ariel Data Challenge 2025)
===================================================================================
Idée : le dataset fait ~264 Go, impossible à stocker. On boucle :

    télécharger un batch → traiter le signal → supprimer le brut → réentraîner
    sur TOUT ce qui a été traité → scorer sur une validation FIXE → décider

et on s'arrête quand le score (lissé) ne progresse plus pendant `patience` batchs.

Choix de conception (corrigent les défauts de la première version) :
  * validation FIXE : tirée une fois (seed), traitée en premier, jamais réentraînée
    -> les scores de deux batchs successifs sont comparables ;
  * ordre des planètes MÉLANGÉ (seed) -> pas de biais lié à l'ordre des ids ;
  * 1 fichier CSV par batch -> idempotent : relancer un batch écrase, ne duplique pas ;
  * arrêt sur score LISSÉ (moyenne glissante) -> robuste au bruit batch à batch ;
  * état JSON -> reprise après interruption ;
  * le cœur est pur : fetch_fn et fit_predict_fn sont injectés (testable sans Kaggle).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# fetch_fn(planet_ids) -> DataFrame (planet_id, Rp_*, wl_*, astro...) ; télécharge, traite, nettoie
FetchFn = Callable[[list], pd.DataFrame]
@dataclass
class FitResult:
    """Retour optionnel de fit_predict_fn : le score ET l'objet à versionner (bundle du modèle)."""
    score: float
    model: object = None


# fit_predict_fn(train_df, val_df) -> float | FitResult
FitPredictFn = Callable[[pd.DataFrame, pd.DataFrame], "float | FitResult"]
LogFn = Callable[[dict], None]
# save_fn(model_bundle, record) : appelé uniquement quand le score lissé atteint un nouveau record
SaveFn = Callable[[object, dict], None]


@dataclass
class IncrementalConfig:
    batch_size: int = 20
    val_size: int = 40
    max_batches: int = 50
    patience: int = 3             # batchs consécutifs sans amélioration du score lissé
    min_delta: float = 1e-3       # amélioration minimale du score lissé
    smooth_window: int = 3        # taille de la moyenne glissante
    min_train_for_fit: int = 20   # pas d'entraînement en dessous
    seed: int = 42
    work_dir: str = "data/processed/incremental"
    binning_airs: int = 5         # paramètres de traitement du signal (figés dans l'état)
    binning_fgs1: int = 20


@dataclass
class IncrementalState:
    order_train: list = field(default_factory=list)   # ids train, déjà mélangés
    val_ids: list = field(default_factory=list)
    batches_done: int = 0
    history: list = field(default_factory=list)       # [{batch, n_train, score, smoothed}]
    best_smoothed: Optional[float] = None
    stalled: int = 0
    stopped: bool = False
    stop_reason: str = ""
    signal: Optional[dict] = None      # binning utilisés : changer = spectres incomparables
    batch_size: Optional[int] = None   # figé au premier lancement : changer le découpage casserait la reprise

    @classmethod
    def load(cls, path: Path) -> "IncrementalState":
        return cls(**json.loads(path.read_text())) if path.exists() else cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        tmp.replace(path)   # écriture atomique


# --------------------------------------------------------------------------- utilitaires purs

def split_ids(all_ids: list, val_size: int, seed: int) -> tuple[list, list]:
    """Mélange déterministe puis (validation fixe, flux d'entraînement)."""
    ids = [str(int(i)) for i in all_ids]
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(ids))
    if val_size >= len(perm):
        raise ValueError("val_size >= nombre de planètes")
    return [str(i) for i in perm[:val_size]], [str(i) for i in perm[val_size:]]


def smoothed(scores: list, window: int) -> float:
    return float(np.mean(scores[-window:]))


def update_stopping(state: IncrementalState, cfg: IncrementalConfig, score: float) -> None:
    """Met à jour best_smoothed / stalled / stopped. history[-1] contient déjà `score`."""
    assert state.history and state.history[-1]["score"] == score
    scores = [h["score"] for h in state.history if h["score"] is not None]
    sm = smoothed(scores, cfg.smooth_window)
    if state.best_smoothed is None or sm > state.best_smoothed + cfg.min_delta:
        state.best_smoothed = sm
        state.stalled = 0
    else:
        state.stalled += 1
    state.history[-1]["smoothed"] = sm
    if state.stalled >= cfg.patience:
        state.stopped, state.stop_reason = True, (
            f"score lissé sans amélioration > {cfg.min_delta} pendant {cfg.patience} batchs"
        )


# --------------------------------------------------------------------------- boucle

class IncrementalRunner:
    def __init__(self, cfg: IncrementalConfig, all_ids: list, fetch_fn: FetchFn,
                 fit_predict_fn: FitPredictFn, log_fn: Optional[LogFn] = None,
                 save_fn: Optional[SaveFn] = None):
        self.cfg, self.fetch_fn, self.fit_fn, self.log_fn = cfg, fetch_fn, fit_predict_fn, log_fn
        self.save_fn = save_fn
        self.dir = Path(cfg.work_dir)
        self.state_path = self.dir / "state.json"
        self.state = IncrementalState.load(self.state_path)
        if not self.state.order_train:                      # première exécution
            self.state.val_ids, self.state.order_train = split_ids(all_ids, cfg.val_size, cfg.seed)
        if self.state.batch_size is None:
            self.state.batch_size = cfg.batch_size
        sig = {"binning_airs": cfg.binning_airs, "binning_fgs1": cfg.binning_fgs1}
        if self.state.signal is None:
            self.state.signal = sig
        if self.state.signal != sig:
            raise ValueError(f"Reprise incompatible : spectres calculés avec {self.state.signal}, "
                             f"demandé {sig}. Repars d'un dossier work_dir vide.")
        if self.state.batch_size != cfg.batch_size or len(self.state.val_ids) != cfg.val_size:
            raise ValueError(
                f"Reprise incompatible : l'état a batch_size={self.state.batch_size}, "
                f"val_size={len(self.state.val_ids)} ; configuration demandée "
                f"{cfg.batch_size}/{cfg.val_size}. Relance avec les mêmes valeurs.")
        self.state.save(self.state_path)

    # --- fichiers
    def _val_path(self) -> Path:
        return self.dir / "val.csv"

    def _batch_path(self, k: int) -> Path:
        return self.dir / "batches" / f"batch_{k:04d}.csv"

    def _write(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(path)                                   # idempotent : écrase, jamais d'append

    def _batch_ids(self, k: int) -> list:
        s = k * self.cfg.batch_size
        return self.state.order_train[s: s + self.cfg.batch_size]

    def _load_train(self, upto: int) -> pd.DataFrame:
        parts = [pd.read_csv(self._batch_path(k)) for k in range(upto + 1)]
        return pd.concat(parts, ignore_index=True).drop_duplicates("planet_id")

    # --- une itération
    def step(self) -> bool:
        """Traite le prochain batch. Retourne False si la boucle doit s'arrêter."""
        st, cfg = self.state, self.cfg
        if st.stopped:
            return False
        k = st.batches_done
        ids = self._batch_ids(k)
        if not ids:                       # arrêt définitif
            st.stopped, st.stop_reason = True, "données épuisées"
            st.save(self.state_path)
            return False
        if k >= cfg.max_batches:          # budget atteint : NON persistant, on peut relever --max-batches
            logger.info("max_batches=%d atteint (reprise possible avec une valeur plus grande)", cfg.max_batches)
            return False

        if not self._val_path().exists():
            self._write(self.fetch_fn(st.val_ids), self._val_path())
        if not self._batch_path(k).exists():
            self._write(self.fetch_fn(ids), self._batch_path(k))

        train = self._load_train(k)
        rec = {"batch": k, "n_train": int(len(train)), "score": None, "smoothed": None}
        st.history.append(rec)
        if len(train) >= cfg.min_train_for_fit:
            res = self.fit_fn(train, pd.read_csv(self._val_path()))
            score = float(res.score if isinstance(res, FitResult) else res)
            rec["score"] = score
            before = st.best_smoothed
            update_stopping(st, cfg, score)
            if self.save_fn and isinstance(res, FitResult) and st.best_smoothed != before:
                rec["saved_model"] = True               # nouveau record -> on versionne ce modèle
                self.save_fn(res.model, rec)
        st.batches_done = k + 1
        st.save(self.state_path)
        logger.info("batch %d | n_train=%d | score=%s | lissé=%s | stalled=%d",
                    k, rec["n_train"], rec["score"], rec["smoothed"], st.stalled)
        if self.log_fn:
            self.log_fn(rec)
        return not st.stopped

    def run(self) -> IncrementalState:
        while self.step():
            pass
        return self.state


# --------------------------------------------------------------------------- câblage réel

def baseline_fit_predict(train: pd.DataFrame, val: pd.DataFrame) -> float:
    """Ridge sur [Rp_*, astro] ; σ = écart-type des résidus d'entraînement (par longueur d'onde)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from src.ml.features import build_feature_matrix, build_target_matrix, compute_naive_stats
    from src.ml.metrics import evaluate_on_validation

    Xtr, cols = build_feature_matrix(train)
    ytr, _ = build_target_matrix(train)
    Xva, _ = build_feature_matrix(val)
    yva, _ = build_target_matrix(val)
    ok_tr, ok_va = np.isfinite(Xtr).all(1), np.isfinite(Xva).all(1)
    Xtr, ytr = Xtr[ok_tr], ytr[ok_tr]
    Xva, yva, ids_va = Xva[ok_va], yva[ok_va], val["planet_id"].to_numpy()[ok_va]
    sc = StandardScaler().fit(Xtr)                     # fit sur le train uniquement
    model = Ridge(alpha=10.0).fit(sc.transform(Xtr), ytr)
    sigma = np.maximum((ytr - model.predict(sc.transform(Xtr))).std(axis=0), 1e-6)
    pred = model.predict(sc.transform(Xva))
    nm, ns = compute_naive_stats(ytr)
    score = float(evaluate_on_validation(yva, pred, np.tile(sigma, (len(pred), 1)), ids_va, nm, ns))
    bundle = {"scaler": sc, "model": model, "sigma": sigma, "feat_cols": cols,
              "naive_mean": nm, "naive_sigma": ns}
    return FitResult(score=score, model=bundle)


def make_kaggle_fetch(raw_dir: str = "data/raw", split: str = "train", keep_raw: bool = False,
                      binning_airs: int = 5, binning_fgs1: int = 20) -> FetchFn:
    """fetch_fn réel : Kaggle → process_one_planet → suppression du brut → merge labels + astro."""
    import shutil
    from src.data import collect
    from src.signal_processing.io import wavelength_index
    from src.signal_processing.pipeline import process_one_planet

    raw = Path(raw_dir)
    api = collect.get_api()
    collect.download_meta(api)
    wl = wavelength_index(raw / "wavelengths.csv", raw / "axis_info.parquet")
    cut_inf, cut_sup = int(wl[0]), int(wl[-1]) + 1
    adc = pd.read_csv(raw / "adc_info.csv")
    gain, offset = float(adc["AIRS-CH0_adc_gain"].iloc[0]), float(adc["AIRS-CH0_adc_offset"].iloc[0])
    axis = pd.read_parquet(raw / "axis_info.parquet")
    labels = pd.read_csv(raw / "train.csv")
    stars = pd.read_csv(raw / f"{split}_star_info.csv")

    def fetch(ids: list) -> pd.DataFrame:
        rows = []
        for pid in ids:
            try:
                if collect.download_planet(api, pid, split):
                    r = process_one_planet(pid, raw, split, axis, gain, offset, cut_inf, cut_sup,
                                           binning_airs=binning_airs, binning_fgs1=binning_fgs1)
                    if r is not None:
                        rows.append(r)
            finally:
                if not keep_raw:
                    shutil.rmtree(raw / split / pid, ignore_errors=True)
        df = pd.DataFrame(rows)
        df["planet_id"] = df["planet_id"].astype(int)
        return df.merge(labels, on="planet_id").merge(stars, on="planet_id")
    return fetch


def traceability_tags(cfg: IncrementalConfig) -> dict:
    """Git + version des spectres (hash du contenu des CSV) + paramètres : de quoi relier un modèle à ses données."""
    import hashlib
    import subprocess
    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             timeout=5).stdout.strip() or "unknown"
    except Exception:
        git = "unknown"
    h = hashlib.sha256()
    d = Path(cfg.work_dir)
    for f in sorted([d / "val.csv", *sorted((d / "batches").glob("batch_*.csv"))]):
        if f.exists():
            h.update(f.read_bytes())
    return {"git_commit": git, "spectra_sha256": h.hexdigest()[:12],
            "binning_airs": str(cfg.binning_airs), "binning_fgs1": str(cfg.binning_fgs1),
            "seed": str(cfg.seed)}


def make_model_saver(cfg: IncrementalConfig, models_dir: str = "models/champion", use_mlflow: bool = False) -> SaveFn:
    """Sauvegarde le bundle du modèle à chaque nouveau record : copie locale « champion » + artefact MLflow."""
    import joblib

    def save(bundle: object, rec: dict) -> None:
        out = Path(models_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, out / "bundle.joblib")
        meta = {**traceability_tags(cfg), "n_train": rec["n_train"], "score": rec["score"],
                "smoothed": rec["smoothed"], "batch": rec["batch"]}
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        if use_mlflow:
            import mlflow
            mlflow.set_tags({k: str(v) for k, v in meta.items() if k not in ("score", "smoothed")})
            mlflow.log_artifact(str(out / "bundle.joblib"), artifact_path=f"models/n{rec['n_train']}")
            mlflow.log_artifact(str(out / "meta.json"), artifact_path=f"models/n{rec['n_train']}")
        logger.info("modèle versionné (n_train=%d, score lissé=%.4f)", rec["n_train"], rec["smoothed"])
    return save


def register_champion(models_dir: str = "models/champion", name: str = "ariel-champion") -> None:
    """Enregistre le dernier modèle record dans le Model Registry MLflow (si le backend le permet)."""
    try:
        import mlflow
        run = mlflow.active_run()
        if run is None:
            return
        mlflow.register_model(f"runs:/{run.info.run_id}/models", name)
    except Exception as e:                 # un backend fichier n'a pas de registry : non bloquant
        logger.warning("Model Registry indisponible (%s) — le modèle reste dans models/champion/", e)


def write_reports(state: IncrementalState, work_dir: str) -> None:
    """metrics.json (dvc metrics) + history.csv (dvc plots)."""
    d = Path(work_dir)
    d.mkdir(parents=True, exist_ok=True)
    scored = [h for h in state.history if h["score"] is not None]
    (d / "metrics.json").write_text(json.dumps({
        "best_smoothed_gll": state.best_smoothed,
        "last_gll": scored[-1]["score"] if scored else None,
        "n_train": state.history[-1]["n_train"] if state.history else 0,
        "batches_done": state.batches_done,
        "stopped": state.stopped, "stop_reason": state.stop_reason}, indent=2))
    pd.DataFrame(state.history).to_csv(d / "history.csv", index=False)


def load_params(path: str = "params.yaml") -> dict:
    import yaml
    p = Path(path)
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def mlflow_logger(experiment: str = "ariel-incremental") -> LogFn:
    import mlflow
    mlflow.set_experiment(experiment)
    run = mlflow.start_run()

    def log(rec: dict) -> None:
        if rec["score"] is not None:
            mlflow.log_metric("gll_score", rec["score"], step=rec["n_train"])
            mlflow.log_metric("gll_score_smoothed", rec["smoothed"], step=rec["n_train"])
    return log


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--val-size", type=int)
    ap.add_argument("--max-batches", type=int)
    ap.add_argument("--patience", type=int)
    ap.add_argument("--mlflow", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    prm = load_params(a.params)
    kw = {**prm.get("incremental", {})}
    kw.update({k: v for k, v in dict(batch_size=a.batch_size, val_size=a.val_size,
                                     max_batches=a.max_batches, patience=a.patience).items() if v is not None})
    sig = prm.get("signal", {})
    kw.setdefault("binning_airs", sig.get("binning_airs", 5))
    kw.setdefault("binning_fgs1", sig.get("binning_fgs1", 20))
    cfg = IncrementalConfig(**kw)
    stars = pd.read_csv("data/raw/train_star_info.csv")
    runner = IncrementalRunner(
        cfg, stars["planet_id"].tolist(),
        make_kaggle_fetch(binning_airs=cfg.binning_airs, binning_fgs1=cfg.binning_fgs1),
        baseline_fit_predict, mlflow_logger() if a.mlflow else None,
        make_model_saver(cfg, use_mlflow=a.mlflow))
    try:
        st = runner.run()
    finally:
        write_reports(runner.state, "reports/incremental")
    if a.mlflow:
        register_champion()
    logger.info("Arrêt : %s | meilleur score lissé = %s", st.stop_reason or "budget", st.best_smoothed)


if __name__ == "__main__":
    main()
