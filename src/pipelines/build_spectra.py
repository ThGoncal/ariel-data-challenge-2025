"""
build_spectra.py — Courbes de lumière sauvegardées -> spectres (mise en page val.csv + batches/), reproductible
==============================================================================================================
Étape DVC `spectra` : aucun téléchargement. Pour chaque planète du découpage figé (configs/split_v1.json) :
    Rp_1..283 = sqrt(max(profondeur, 0)) calculé par `extract_spectrum` (params.yaml : extraction)
    + labels (data/raw/train.csv) + astro (data/raw/train_star_info.csv)
Même mise en page que la boucle incrémentale (val.csv, batches/batch_XXX.csv) : les scripts de comparaison
s'appliquent tels quels avec --root data/processed/spectra_v2.

    # une fois : figer le découpage de la boucle incrémentale v0.1.0 (validation fixe + ordre d'entraînement)
    python -m src.pipelines.build_spectra --export-split data/processed/incremental
    # puis (ou via dvc repro spectra)
    python -m src.pipelines.build_spectra --params params.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.signal_processing.spectra import RP_COLS, ExtractionParams, depth_to_rp, extract_spectrum

logger = logging.getLogger(__name__)
SPLIT = Path("configs/split_v1.json")
CURVES = Path("data/processed/transit_curves")
RAW = Path("data/raw")
OUT = Path("data/processed/spectra_v2")


# --------------------------------------------------------------------------- découpage figé

def export_split(incremental_root: Path, batch_size: int = 10) -> dict:
    """Relit val.csv + batches/*.csv de la boucle incrémentale : ids de validation et ordre d'entraînement."""
    root = Path(incremental_root)
    val = [int(p) for p in pd.read_csv(root / "val.csv", usecols=["planet_id"])["planet_id"]]
    train: list[int] = []
    for f in sorted((root / "batches").glob("batch_*.csv")):
        train += [int(p) for p in pd.read_csv(f, usecols=["planet_id"])["planet_id"]]
    train = list(dict.fromkeys(p for p in train if p not in set(val)))
    return {"val": val, "train": train, "batch_size": batch_size, "source": "incremental v0.1.0"}


def extend_split(split: dict, curves_dir: Path, candidates: list[int]) -> dict:
    """Nouveau découpage : MÊME validation, MÊME début d'entraînement, puis les planètes qui ont des courbes
    (dans l'ordre de `candidates`) et ne figurent pas déjà dans le découpage."""
    known = set(split["val"]) | set(split["train"])
    added = [int(p) for p in candidates if int(p) not in known and (Path(curves_dir) / f"{int(p)}.npz").exists()]
    added = list(dict.fromkeys(added))
    return {"val": list(split["val"]), "train": list(split["train"]) + added, "batch_size": split.get("batch_size", 10),
            "source": f"{split.get('source', 'split')} + {len(added)} planètes avec courbes", "n_added": len(added)}


def load_split(path: Path) -> dict:
    s = json.loads(Path(path).read_text())
    if set(s["val"]) & set(s["train"]):
        raise ValueError("planètes communes à la validation et à l'entraînement")
    return s


# --------------------------------------------------------------------------- construction

def curves_digest(curves_dir: Path, ids: list[int]) -> str:
    """Empreinte du contenu des courbes utilisées : relie un modèle aux données exactes qui l'ont produit."""
    h = hashlib.sha256()
    for pid in sorted(ids):
        f = Path(curves_dir) / f"{pid}.npz"
        if f.exists():
            h.update(str(pid).encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:12]


def build_rows(ids: list[int], curves_dir: Path, labels: pd.DataFrame, stars: pd.DataFrame,
               p: ExtractionParams) -> tuple[pd.DataFrame, list[int]]:
    """Une ligne par planète disposant de courbes ; retourne aussi les ids sans courbes."""
    rows, missing = [], []
    for pid in ids:
        f = Path(curves_dir) / f"{pid}.npz"
        if not f.exists():
            missing.append(pid)
            continue
        z = np.load(f)
        rp = depth_to_rp(extract_spectrum(z["airs"], z["fgs"], p))
        rows.append({"planet_id": pid, **dict(zip(RP_COLS, rp))})
    if not rows:
        return pd.DataFrame(columns=["planet_id", *RP_COLS]), missing
    df = pd.DataFrame(rows)
    return df.merge(labels, on="planet_id").merge(stars, on="planet_id"), missing


def extraction_level(df: pd.DataFrame) -> pd.Series:
    """Niveau d'extraction par planète = médiane(Rp² / label). ~1 si le spectre extrait est cohérent avec le label
    (plage normale observée : 0,8 à 1,25). Utilise les LABELS : réservé au contrôle qualité de l'entraînement."""
    rp2 = df[RP_COLS].to_numpy(float) ** 2
    y = df[[c.replace("Rp_", "wl_") for c in RP_COLS]].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(y > 0, rp2 / y, np.nan)
    return pd.Series(np.nanmedian(ratio, axis=1), index=df["planet_id"].to_numpy())


def quality_filter(df: pd.DataFrame, level_range: tuple[float, float]) -> tuple[pd.DataFrame, dict[int, float]]:
    """Écarte les planètes dont le niveau d'extraction sort de [lo, hi] (NaN compris) ; retourne les écartées."""
    if not len(df):
        return df, {}
    lo, hi = level_range
    lev = extraction_level(df)
    ok = ((lev >= lo) & (lev <= hi)).to_numpy()
    return df[ok], {int(k): round(float(v), 4) for k, v in lev[~ok].items()}


def build_layout(split: dict, curves_dir: Path, raw_dir: Path, out_dir: Path, p: ExtractionParams,
                 strict: bool = False, level_range: tuple[float, float] | None = None) -> dict:
    labels = pd.read_csv(Path(raw_dir) / "train.csv")
    stars = pd.read_csv(Path(raw_dir) / "train_star_info.csv")
    out = Path(out_dir)
    (out / "batches").mkdir(parents=True, exist_ok=True)
    for old in (out / "batches").glob("batch_*.csv"):
        old.unlink()                                   # pas de lots périmés d'une exécution précédente

    val_df, missing_val = build_rows(split["val"], curves_dir, labels, stars, p)
    bs = int(split.get("batch_size", 10))
    n_written, missing_train, excluded = 0, [], {}
    for k in range(0, len(split["train"]), bs):
        df, miss = build_rows(split["train"][k: k + bs], curves_dir, labels, stars, p)
        missing_train += miss
        if level_range is not None:
            df, bad = quality_filter(df, level_range)
            excluded.update(bad)
        if len(df):
            df.to_csv(out / "batches" / f"batch_{k // bs:03d}.csv", index=False)
            n_written += len(df)
    val_df.to_csv(out / "val.csv", index=False)

    missing = missing_val + missing_train
    if missing and strict:
        raise RuntimeError(f"{len(missing)} planètes sans courbes : {missing}")
    if missing:
        logger.warning("%d planètes sans courbes, ignorées : %s", len(missing), missing)
    manifest = {"n_val": len(val_df), "n_train": n_written, "missing": missing, "extraction": p.as_dict(),
                "curves_sha256": curves_digest(curves_dir, split["val"] + split["train"]),
                "level_range": list(level_range) if level_range else None,
                "excluded_train": excluded}                # {planet_id: niveau d'extraction}, tracé pour audit
    if level_range is not None:                            # la validation n'est JAMAIS filtrée : on la signale seulement
        manifest["val_out_of_range"] = quality_filter(val_df, level_range)[1]
    if excluded:
        logger.warning("%d planètes d'entraînement écartées (niveau hors %s) : %s", len(excluded), level_range, excluded)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--split", default=str(SPLIT))
    ap.add_argument("--curves", default=str(CURVES))
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--strict", action="store_true", help="échoue si une planète du découpage n'a pas de courbes")
    ap.add_argument("--level-range", nargs=2, type=float, metavar=("MIN", "MAX"), default=None,
                    help="écarte de l'ENTRAÎNEMENT les planètes dont médiane(Rp²/label) sort de [MIN, MAX] (ex. 0.5 2)")
    ap.add_argument("--extend-split", metavar="DECOUPAGE_SOURCE", default=None,
                    help="écrit dans --split un découpage étendu aux planètes qui ont des courbes (validation inchangée)")
    ap.add_argument("--export-split", metavar="INCREMENTAL_ROOT", default=None,
                    help="écrit le découpage figé depuis la boucle incrémentale puis s'arrête")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if a.extend_split:
        if Path(a.split).resolve() == Path(a.extend_split).resolve():
            raise SystemExit("--split doit être un NOUVEAU fichier (ne jamais écraser le découpage source).")
        labels = pd.read_csv(Path(a.raw) / "train.csv", usecols=["planet_id"])["planet_id"].astype(int).tolist()
        new = extend_split(load_split(Path(a.extend_split)), Path(a.curves), labels)
        Path(a.split).parent.mkdir(parents=True, exist_ok=True)
        Path(a.split).write_text(json.dumps(new))
        print(f"découpage étendu écrit : {a.split} ({len(new['val'])} validation, {len(new['train'])} entraînement, "
              f"dont {new['n_added']} ajoutées)")
        return
    if a.export_split:
        split = export_split(Path(a.export_split))
        Path(a.split).parent.mkdir(parents=True, exist_ok=True)
        Path(a.split).write_text(json.dumps(split))
        print(f"découpage écrit : {a.split} ({len(split['val'])} validation, {len(split['train'])} entraînement)")
        return

    import yaml
    params = yaml.safe_load(Path(a.params).read_text())
    p = ExtractionParams.from_params(params)
    m = build_layout(load_split(Path(a.split)), Path(a.curves), Path(a.raw), Path(a.out), p, strict=a.strict,
                     level_range=tuple(a.level_range) if a.level_range else None)
    print(f"spectres écrits dans {a.out} : {m['n_val']} validation, {m['n_train']} entraînement, "
          f"{len(m['missing'])} planètes sans courbes | extraction {p.as_dict()} | courbes {m['curves_sha256']}"
          + (f" | filtre qualité {m['level_range']} : {len(m['excluded_train'])} écartées, "
             f"{len(m['val_out_of_range'])} en validation hors plage (non filtrées)" if m["level_range"] else ""))


if __name__ == "__main__":
    main()
