"""
reprocess_curves.py — Retraitement unique des planètes : sauvegarde des COURBES de lumière
=========================================================================================
Pour chaque planète du jeu de travail (validation fixe + planètes d'entraînement des batches), télécharge le
brut, traite le signal, sauvegarde la courbe (flux sommé sur l'espace, calibré, binné) dans
data/processed/transit_curves/{planet_id}.npz (même format que check_transit_window.py), puis SUPPRIME le brut.
Ensuite toute l'extraction (fenêtre adaptée, canaux de bord, ...) se refait hors ligne, sans retélécharger.

  - REPRENABLE : une planète dont le .npz existe est sautée ; relancer la commande reprend où on s'est arrêté
    et retente les échecs. Écriture atomique (pas de fichier tronqué si le processus est tué).
  - Le téléchargement de la planète suivante se fait pendant le traitement de la courbe courante.
  - Contrôle de non-régression : le Rp recalculé est comparé à celui déjà présent dans les CSV de spectres.
  - S'arrête proprement si l'espace disque est trop faible ou après trop d'échecs consécutifs (quota, réseau).

    uv run python -m scripts.reprocess_curves --limit 2          # essai sur 2 planètes
    nohup uv run python -m scripts.reprocess_curves > logs/reprocess.out 2>&1 &    # la nuit, en tâche de fond
    tail -f logs/reprocess_curves.log                              # suivre l'avancement

Puis : dvc add data/processed/transit_curves && git add data/processed/transit_curves.dvc && dvc push
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
CURVES = Path("data/processed/transit_curves")
INCREMENTAL = Path("data/processed/incremental")
N_WL = 282
log = logging.getLogger("reprocess")


# --------------------------------------------------------------------------- briques pures (testées)

def planet_list(root: Path, labels_csv: Path | None = None, ids: list[str] | None = None) -> list[str]:
    """Planètes à traiter : validation d'abord, puis entraînement dans l'ordre des batches, sans doublon.
    `ids` impose une liste ; sinon, si `root` n'existe pas, toutes les planètes de labels_csv."""
    if ids:
        return [str(i) for i in ids]
    out: list[str] = []
    files = ([root / "val.csv"] if (root / "val.csv").exists() else []) + sorted((root / "batches").glob("batch_*.csv"))
    for f in files:
        out += [str(int(p)) for p in pd.read_csv(f, usecols=["planet_id"])["planet_id"]]
    if not out and labels_csv is not None:
        out = [str(int(p)) for p in pd.read_csv(labels_csv)["planet_id"]]
    return list(dict.fromkeys(out))


def validate_curves(airs: np.ndarray, fgs: np.ndarray, rp: np.ndarray) -> None:
    if airs.ndim != 2 or airs.shape[1] != N_WL:
        raise ValueError(f"airs de forme inattendue : {airs.shape}")
    if fgs.ndim != 1 or len(fgs) < 50 or airs.shape[0] < 50:
        raise ValueError(f"série trop courte : airs {airs.shape}, fgs {fgs.shape}")
    if rp.shape != (N_WL + 1,):
        raise ValueError(f"rp de forme inattendue : {rp.shape}")
    if not (np.isfinite(airs).all() and np.isfinite(fgs).all()):
        raise ValueError("valeurs non finies dans les courbes")


def save_curves(path: Path, airs: np.ndarray, fgs: np.ndarray, rp: np.ndarray) -> None:
    """Écriture atomique : fichier temporaire puis renommage."""
    validate_curves(airs, fgs, rp)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, airs=airs.astype(np.float32), fgs=fgs.astype(np.float32), rp=np.asarray(rp, float))
    os.replace(tmp, path)


def fmt_eta(seconds: float) -> str:
    h, m = int(seconds // 3600), int(seconds % 3600 // 60)
    return f"{h} h {m:02d} min"


def is_complete(planet_dir: Path, files: list[str]) -> bool:
    """Tous les fichiers de la planète sont présents et non vides."""
    return all((planet_dir / f).is_file() and (planet_dir / f).stat().st_size > 0 for f in files)


def locate_local(root: Path, pid: str, files: list[str]) -> tuple[Path, str] | None:
    """(base_path, base_folder) tels que base_path/base_folder/pid contienne tous les fichiers, sinon None.
    Accepte root/train/pid (comme data/raw) et root/pid (dossier de planètes à plat)."""
    root = Path(root)
    for base, folder in ((root, "train"), (root.parent, root.name)):
        if is_complete(base / folder / pid, files):
            return base, folder
    return None


def run(todo: list[str], download, process, save, cleanup, free_gb=lambda: 1e9, min_free_gb: float = 5.0,
        max_consecutive_failures: int = 8, prefetch: bool = True) -> dict:
    """Boucle principale. download(pid)->bool, process(pid)->dict|None, save(pid, result), cleanup(pid).
    Le téléchargement de la planète suivante est lancé pendant le traitement de la courante."""
    stats = {"ok": [], "failed": [], "stopped": None}
    t0, consecutive = time.time(), 0
    pool = ThreadPoolExecutor(max_workers=1) if prefetch else None
    futures: dict[str, object] = {}

    def start(pid):
        if pool and pid not in futures:
            futures[pid] = pool.submit(download, pid)

    try:
        if todo:
            start(todo[0])
        for i, pid in enumerate(todo):
            if free_gb() < min_free_gb:
                stats["stopped"] = f"espace disque insuffisant ({free_gb():.1f} Go libres)"
                break
            if consecutive >= max_consecutive_failures:
                stats["stopped"] = f"{consecutive} échecs consécutifs (quota Kaggle, réseau ?)"
                break
            try:
                ok = futures.pop(pid).result() if pid in futures else download(pid)
                if i + 1 < len(todo):
                    start(todo[i + 1])
                if not ok:
                    raise RuntimeError("téléchargement incomplet")
                res = process(pid)
                if res is None:
                    raise RuntimeError("échec du traitement")
                save(pid, res)
                stats["ok"].append(pid)
                consecutive = 0
                done = len(stats["ok"]) + len(stats["failed"])
                rate = (time.time() - t0) / done
                log.info("[%s] ok  (%d/%d, %.1f min/planète, reste ~%s)", pid, i + 1, len(todo), rate / 60,
                         fmt_eta(rate * (len(todo) - i - 1)))
            except KeyboardInterrupt:
                stats["stopped"] = "interrompu par l'utilisateur"
                break
            except Exception as e:  # une planète en échec ne doit pas arrêter la nuit
                stats["failed"].append(pid)
                consecutive += 1
                log.warning("[%s] ÉCHEC : %s", pid, e)
            finally:
                cleanup(pid)
                if i + 1 < len(todo):
                    start(todo[i + 1])
    finally:
        if pool:
            pool.shutdown(wait=True)
        for pid in list(futures):
            cleanup(pid)
    return stats


# --------------------------------------------------------------------------- branchement réel (réseau)

def known_rp(root: Path) -> dict[str, np.ndarray]:
    """Rp déjà calculés (CSV de spectres) : sert au contrôle de non-régression."""
    out: dict[str, np.ndarray] = {}
    files = ([root / "val.csv"] if (root / "val.csv").exists() else []) + sorted((root / "batches").glob("batch_*.csv"))
    cols = [f"Rp_{i}" for i in range(1, N_WL + 2)]
    for f in files:
        df = pd.read_csv(f)
        if set(cols) <= set(df.columns):
            for pid, row in zip(df["planet_id"], df[cols].to_numpy(float)):
                out[str(int(pid))] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="n'en traiter que N (essai)")
    ap.add_argument("--root", default=str(INCREMENTAL))
    ap.add_argument("--curves", default=str(CURVES))
    ap.add_argument("--min-free-gb", type=float, default=5.0)
    ap.add_argument("--no-prefetch", action="store_true")
    ap.add_argument("--local-raw", default=None, metavar="DOSSIER",
                    help="traite des planètes déjà téléchargées (DOSSIER/train/<id>/ ou DOSSIER/<id>/) : "
                         "aucun téléchargement, et le brut n'est JAMAIS supprimé")
    a = ap.parse_args()

    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%d/%m %H:%M:%S",
                        handlers=[logging.StreamHandler(), logging.FileHandler("logs/reprocess_curves.log")])
    curves = Path(a.curves)
    curves.mkdir(parents=True, exist_ok=True)
    for tmp in curves.glob("*.tmp.npz"):          # restes d'un processus tué
        tmp.unlink()

    planets = planet_list(Path(a.root), RAW / "train.csv", a.ids)
    todo = [p for p in planets if not (curves / f"{p}.npz").exists()]
    if a.limit:
        todo = todo[: a.limit]
    log.info("%d planètes au total, %d déjà faites, %d à traiter", len(planets), len(planets) - len(
        [p for p in planets if not (curves / f"{p}.npz").exists()]), len(todo))
    if not todo:
        log.info("Rien à faire.")
        return

    import yaml

    from src.data import collect
    from src.signal_processing.io import wavelength_index
    from src.signal_processing.pipeline import process_one_planet

    local = Path(a.local_raw) if a.local_raw else None
    meta = ("wavelengths.csv", "axis_info.parquet", "adc_info.csv", "train.csv", "train_star_info.csv")
    sp = yaml.safe_load(open("params.yaml"))["signal"]
    api = None
    if local is None:
        api = collect.get_api()
        if not all((RAW / f).exists() for f in meta):
            collect.download_meta(api)
    elif not all((RAW / f).exists() for f in meta):
        raise SystemExit("Fichiers de métadonnées absents de data/raw : lance `dvc pull` ou une fois sans --local-raw.")
    wl = wavelength_index(RAW / "wavelengths.csv", RAW / "axis_info.parquet")
    cut_inf, cut_sup = int(wl[0]), int(wl[-1]) + 1
    adc = pd.read_csv(RAW / "adc_info.csv")
    gain, offset = float(adc["AIRS-CH0_adc_gain"].iloc[0]), float(adc["AIRS-CH0_adc_offset"].iloc[0])
    axis = pd.read_parquet(RAW / "axis_info.parquet")
    ref = known_rp(Path(a.root))
    diffs: list[float] = []
    where: dict[str, tuple[Path, str]] = {}

    def download(pid: str) -> bool:
        if local is None:
            return bool(collect.download_planet(api, pid, "train"))
        loc = locate_local(local, pid, collect.PLANET_FILES)
        if loc is None:
            log.warning("[%s] fichiers absents ou incomplets sous %s", pid, local)
            return False
        where[pid] = loc
        return True

    def process(pid: str):
        cubes: dict = {}
        base, folder = where.get(pid, (RAW, "train"))
        row = process_one_planet(pid, base, folder, axis, gain, offset, cut_inf, cut_sup,
                                 binning_airs=sp["binning_airs"], binning_fgs1=sp["binning_fgs1"], _cubes_out=cubes)
        if row is None:
            return None
        rp = np.array([row[f"Rp_{i}"] for i in range(1, N_WL + 2)])
        if pid in ref:
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.abs(rp / ref[pid] - 1)
            d = float(np.nanmax(np.where(np.isclose(rp, ref[pid]), 0.0, r)))     # 0/0 (profondeur nulle) = identique
            diffs.append(d)
            if d > 1e-3:
                log.warning("[%s] Rp recalculé ≠ Rp des CSV (écart relatif max %.2g) : paramètres différents ?", pid, d)
        return {"airs": cubes["airs"].sum(axis=1), "fgs": cubes["fgs1"].sum(axis=(1, 2)), "rp": rp}

    def save(pid: str, r: dict) -> None:
        save_curves(curves / f"{pid}.npz", r["airs"], r["fgs"], r["rp"])

    def cleanup(pid: str) -> None:
        if local is None:                                  # en mode local, le brut appartient à l'utilisateur
            shutil.rmtree(RAW / "train" / pid, ignore_errors=True)

    t0 = time.time()
    stats = run(todo, download, process, save, cleanup,
                free_gb=lambda: shutil.disk_usage(curves).free / 1e9, min_free_gb=a.min_free_gb,
                prefetch=not a.no_prefetch)
    log.info("Terminé en %s : %d ok, %d échecs%s", fmt_eta(time.time() - t0), len(stats["ok"]), len(stats["failed"]),
             f", ARRÊT : {stats['stopped']}" if stats["stopped"] else "")
    if diffs:
        log.info("Contrôle Rp vs CSV existants : écart relatif max %.2g, médian %.2g sur %d planètes",
                 max(diffs), float(np.median(diffs)), len(diffs))
    if stats["failed"]:
        log.info("Échecs (relancer la commande les retentera) : %s", " ".join(stats["failed"]))


if __name__ == "__main__":
    main()
