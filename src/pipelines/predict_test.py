"""
predict_test.py — Inférence : données brutes d'une planète -> spectre -> modèle champion -> soumission
=====================================================================================================
Même chaîne que l'entraînement, sans réécriture : `process_one_planet` (calibration, binning) puis
`spectrum_from_cubes` (extraction, src/signal_processing/spectra.py) puis le modèle champion.

1. CONTRÔLE (à faire d'abord) : planètes d'ENTRAÎNEMENT dont on a le brut. On refait tout depuis le brut et on
   vérifie que le spectre et les prédictions sont identiques à ceux de data/processed/spectra_v2 (écart ≈ 0).
   Il n'y a pas de labels pour le jeu de test : c'est la seule façon de prouver que la chaîne d'inférence est
   la même que celle de l'entraînement.

   python -m src.pipelines.predict_test --check --raw-root data/raw --folder train --ids 34983 1873185

2. SOUMISSION : planètes du jeu de test -> fichier au format de sample_submission.csv, validé.

   python -m src.pipelines.predict_test --raw-root /mnt/e/ariel_data_2025/downloads --folder test \\
       --out submissions/submission_v0.2.0.csv
   (--model models:/ariel-ridge-adaptive-depth@champion pour passer par le registre MLflow)
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.ml.metrics import build_submission_dataframe
from src.models.champion import predict_champion, score_champion
from src.signal_processing.spectra import N_WL, RP_COLS, ExtractionParams, depth_to_rp, spectrum_from_cubes

logger = logging.getLogger(__name__)
META = Path("data/raw")
BUNDLE = Path("models/champion_v2/bundle.joblib")
SPECTRA = Path("data/processed/spectra_v2")
STAR_COLS = ["Rs", "Ts", "Ms", "Mp", "P", "sma", "i"]
WL_COLS = [f"wl_{i}" for i in range(1, N_WL + 1)]


# --------------------------------------------------------------------------- spectre depuis le brut

def load_context(meta_dir: Path, params: dict) -> dict:
    """Constantes d'instrument communes à toutes les planètes (mêmes fichiers que le retraitement)."""
    from src.signal_processing.io import wavelength_index
    meta = Path(meta_dir)
    wl = wavelength_index(meta / "wavelengths.csv", meta / "axis_info.parquet")
    adc = pd.read_csv(meta / "adc_info.csv")
    sp = params["signal"]
    return {"axis": pd.read_parquet(meta / "axis_info.parquet"),
            "gain": float(adc["AIRS-CH0_adc_gain"].iloc[0]), "offset": float(adc["AIRS-CH0_adc_offset"].iloc[0]),
            "cut_inf": int(wl[0]), "cut_sup": int(wl[-1]) + 1,
            "binning_airs": sp["binning_airs"], "binning_fgs1": sp["binning_fgs1"]}


def planet_depth(pid: int | str, base: Path, folder: str, ctx: dict, p: ExtractionParams) -> np.ndarray:
    """Profondeurs [FGS1, AIRS×282] d'une planète, depuis ses fichiers bruts base/folder/<pid>/."""
    from src.signal_processing.pipeline import process_one_planet
    cubes: dict = {}
    row = process_one_planet(str(pid), Path(base), folder, ctx["axis"], ctx["gain"], ctx["offset"], ctx["cut_inf"],
                             ctx["cut_sup"], binning_airs=ctx["binning_airs"], binning_fgs1=ctx["binning_fgs1"],
                             _cubes_out=cubes)
    if row is None:
        raise RuntimeError(f"planète {pid} : le traitement du signal a échoué (fichiers incomplets ?)")
    return spectrum_from_cubes(cubes, p)


def inference_frame(depths: dict, stars: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par planète : planet_id, Rp_1..283, paramètres stellaires/orbitaux (comme les spectres d'entraînement)."""
    missing_cols = [c for c in STAR_COLS if c not in stars.columns]
    if missing_cols:
        raise ValueError(f"colonnes absentes du fichier astro : {missing_cols} (présentes : {list(stars.columns)})")
    st = stars.assign(planet_id=stars["planet_id"].astype(int)).set_index("planet_id")
    rows = []
    for pid, depth in depths.items():
        if int(pid) not in st.index:
            raise ValueError(f"planète {pid} absente du fichier astro")
        row = {"planet_id": int(pid), **dict(zip(RP_COLS, depth_to_rp(depth)))}
        row.update({c: float(st.loc[int(pid), c]) for c in STAR_COLS})
        rows.append(row)
    df = pd.DataFrame(rows)
    feats = df[RP_COLS + STAR_COLS].to_numpy(dtype=float)
    if not np.isfinite(feats).all():
        bad = df.loc[~np.isfinite(feats).all(axis=1), "planet_id"].tolist()
        raise ValueError(f"valeurs non finies pour les planètes {bad}")
    return df


# --------------------------------------------------------------------------- modèle et soumission

def load_model(spec: str):
    """Chemin d'un bundle joblib ou URI du registre MLflow (models:/nom@alias)."""
    if str(spec).startswith("models:/"):
        import mlflow.pyfunc
        return mlflow.pyfunc.load_model(spec)
    return joblib.load(spec)


def predict_submission(model, df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame au format de soumission (planet_id, wl_1..283, wl_1_std..283_std)."""
    if isinstance(model, dict):
        pred = predict_champion(model, df)
        return build_submission_dataframe(df["planet_id"].to_numpy(), pred.mu, pred.sigma)
    return model.predict(df)


def align_to_sample(sub: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    """Nomme les colonnes comme sample_submission.csv (le code hérité écrit wl_i_std ; l'officiel écrit sigma_i).

    Les colonnes sont positionnelles pour la métrique (283 moyennes puis 283 incertitudes) : on ne renomme que si
    le nombre de colonnes et le bloc planet_id + wl_1..wl_283 sont déjà identiques, sinon on refuse.
    """
    if list(sub.columns) == list(sample.columns):
        return sub
    k = N_WL + 1
    if len(sub.columns) != len(sample.columns) or list(sub.columns[:k]) != list(sample.columns[:k]):
        raise ValueError(f"colonnes incompatibles avec sample_submission.csv ({len(sub.columns)} contre "
                         f"{len(sample.columns)} ; premières colonnes {list(sub.columns[:3])} contre {list(sample.columns[:3])})")
    out = sub.copy()
    out.columns = list(sample.columns)
    return out


def check_submission(sub: pd.DataFrame, sample: pd.DataFrame) -> list[str]:
    """Liste des écarts au format attendu (vide = conforme)."""
    problems = []
    if list(sub.columns) != list(sample.columns):
        problems.append("colonnes différentes de sample_submission.csv")
        return problems
    if sorted(sub["planet_id"].astype(int)) != sorted(sample["planet_id"].astype(int)):
        problems.append("planet_id différents de ceux de sample_submission.csv")
    values = sub.drop(columns="planet_id")
    if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        problems.append("valeurs manquantes ou non finies")
    if (values.to_numpy(dtype=float) < 0).any():
        problems.append("valeurs négatives (refusées par la métrique officielle)")
    if (values.iloc[:, N_WL:].to_numpy(dtype=float) <= 0).any():
        problems.append("écart-type nul ou négatif")
    return problems


# --------------------------------------------------------------------------- contrôle de cohérence

def read_reference(root: Path, ids: list) -> pd.DataFrame:
    """Lignes des spectres d'entraînement (val.csv + batches/) pour ces planètes ; colonne `where` = val | train."""
    root = Path(root)
    files = [("val", root / "val.csv")] + [("train", f) for f in sorted((root / "batches").glob("batch_*.csv"))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)       # tables très larges (580 colonnes)
        want = {int(i) for i in ids}
        parts = []
        for where, f in files:
            d = pd.read_csv(f)
            d = d[d["planet_id"].isin(want)].copy()
            d["where"] = where
            parts.append(d)
        df = pd.concat(parts, ignore_index=True).drop_duplicates("planet_id")
        missing = want - set(df["planet_id"])
        if missing:
            raise ValueError(f"planètes absentes des spectres d'entraînement ({root}) : {sorted(missing)} — "
                             "choisir des planètes du découpage (configs/split_v1.json)")
        return df.set_index("planet_id").loc[[int(i) for i in ids]].reset_index().copy()


def consistency(fresh: pd.DataFrame, reference: pd.DataFrame, model) -> dict:
    """Écarts entre ce que la chaîne d'inférence calcule depuis le brut et ce qui a servi à l'entraînement."""
    f = fresh.set_index("planet_id").loc[reference["planet_id"]]
    a, b = f[RP_COLS].to_numpy(float), reference[RP_COLS].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(a / b - 1)
    rel = np.where(np.isclose(a, b), 0.0, rel)
    pf = predict_submission(model, fresh.set_index("planet_id").loc[reference["planet_id"]].reset_index())
    pr = predict_submission(model, reference)
    dmu = np.abs(pf[WL_COLS].to_numpy(float) - pr[WL_COLS].to_numpy(float))
    return {"max_rel_rp": float(np.nanmax(rel)), "max_abs_dmu": float(dmu.max()),
            "max_rel_dmu": float(np.nanmax(dmu / np.maximum(np.abs(pr[WL_COLS].to_numpy(float)), 1e-12)))}


def gll_per_planet(bundle: dict, df: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Score GLL normalisé de chaque planète ayant des labels (les stats naïves sont celles du bundle)."""
    out = {}
    df = df.drop(columns=[c for c in WL_COLS if c in df.columns])      # les spectres d'entraînement portent déjà des labels
    for pid in df["planet_id"]:
        row = df[df["planet_id"] == pid].merge(labels, on="planet_id")
        out[int(pid)] = float(score_champion(bundle, row)) if len(row) else None
    return out


# --------------------------------------------------------------------------- CLI

def main() -> None:
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True, help="dossier contenant <folder>/<id>/ (ex. data/raw ou le disque externe)")
    ap.add_argument("--folder", default="test", help="sous-dossier : test ou train")
    ap.add_argument("--ids", nargs="*", default=None, help="par défaut : les planètes de sample_submission.csv")
    ap.add_argument("--stars", default=None, help="CSV astro (défaut : <raw-root>/test_star_info.csv)")
    ap.add_argument("--sample", default=None, help="sample_submission.csv (défaut : <raw-root>/sample_submission.csv)")
    ap.add_argument("--meta", default=str(META), help="dossier de wavelengths.csv, axis_info.parquet, adc_info.csv")
    ap.add_argument("--model", default=str(BUNDLE))
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--out", default="submissions/submission_v0.2.0.csv")
    ap.add_argument("--check", action="store_true", help="contrôle de cohérence sur des planètes d'entraînement")
    ap.add_argument("--spectra", default=str(SPECTRA))
    ap.add_argument("--labels", default=str(META / "train.csv"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

    params = yaml.safe_load(Path(a.params).read_text())
    ext = ExtractionParams.from_params(params)
    meta_json = Path(a.model).parent / "meta.json"
    if not str(a.model).startswith("models:/") and meta_json.exists():
        m = json.loads(meta_json.read_text())
        trained = ExtractionParams(m["extraction_method"], int(m["extraction_deg"]), int(m["extraction_edge_channels"]))
        if trained != ext:
            raise SystemExit(f"params.yaml (extraction {ext.as_dict()}) ≠ extraction du modèle {trained.as_dict()} : "
                             "le modèle doit être réentraîné (dvc repro) avant toute inférence.")

    raw = Path(a.raw_root)
    sample_path = Path(a.sample) if a.sample else raw / "sample_submission.csv"
    stars_path = Path(a.stars) if a.stars else raw / ("test_star_info.csv" if a.folder == "test" else "train_star_info.csv")
    if a.ids:
        ids = [int(i) for i in a.ids]
    elif a.check:
        raise SystemExit("--check demande --ids avec des planètes d'entraînement dont le brut est disponible")
    else:
        ids = [int(i) for i in pd.read_csv(sample_path)["planet_id"]]
    stars = pd.read_csv(stars_path)

    ctx = load_context(Path(a.meta), params)
    depths = {}
    for pid in ids:
        depths[pid] = planet_depth(pid, raw, a.folder, ctx, ext)
        print(f"[{pid}] spectre calculé depuis {raw / a.folder / str(pid)}")
    frame = inference_frame(depths, stars)
    model = load_model(a.model)

    if a.check:
        ref = read_reference(Path(a.spectra), ids)
        c = consistency(frame, ref, model)
        print(json.dumps({**c, "planètes": {int(p): w for p, w in zip(ref["planet_id"], ref["where"])}}, indent=2))
        if isinstance(model, dict):
            labels = pd.read_csv(a.labels)
            print("GLL (normalisé) depuis le brut :", gll_per_planet(model, frame, labels))
            print("GLL (normalisé) spectres d'entraînement :", gll_per_planet(model, ref, labels))
            print("NB : une planète 'train' est évaluée sur ses propres données (optimiste) ; seule 'val' est hors échantillon.")
        ok = c["max_rel_rp"] < 1e-3 and c["max_rel_dmu"] < 1e-3
        print("CONTRÔLE :", "OK, chaîne identique à l'entraînement" if ok else "ÉCART — ne pas utiliser pour la soumission")
        raise SystemExit(0 if ok else 1)

    sample = pd.read_csv(sample_path)
    try:
        sub = align_to_sample(predict_submission(model, frame), sample)
    except ValueError as e:
        raise SystemExit(f"soumission non conforme : {e}")
    problems = check_submission(sub, sample)
    if problems:
        raise SystemExit("soumission non conforme : " + " ; ".join(problems))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    print(f"soumission écrite : {out} ({len(sub)} planète(s) × {sub.shape[1]} colonnes), format conforme à {sample_path.name}")
    print("profondeur prédite (wl_1, wl_100, wl_283) :", [float(sub.iloc[0][f"wl_{k}"]) for k in (1, 100, 283)])


if __name__ == "__main__":
    main()
