"""
compare_extraction.py — Quelle extraction du signal donne le meilleur score GLL ? (hors ligne, courbes sauvegardées)
====================================================================================================================
Refait l'extraction de la profondeur depuis data/processed/transit_curves/*.npz (aucun téléchargement) pour
plusieurs configurations, remplace les colonnes Rp_1..283 des spectres, puis mesure le score GLL :

  extraction : actuelle (pipeline v0.1.0) | adaptée | adaptée + droite | gabarit + droite | gabarit + parabole
  bord       : nombre de premiers canaux AIRS remplacés par la moyenne des 10 canaux suivants (0 = aucun)
  modèles    : Ridge sur Rp² + astro (σ constant calculé sur l'entraînement)   |   Ridge Rp + astro (v0.1.0)

Deux évaluations, car 20 planètes de validation sont trop peu pour trancher :
  - VALIDATION FIXE (celle du projet) avec intervalle bootstrap ;
  - VALIDATION CROISÉE répétée (5 plis × 3) sur TOUTES les planètes disposant de courbes, mêmes plis pour toutes
    les configurations : on compare par DIFFÉRENCE appariée à la référence (extraction actuelle, bord 0) et on
    regarde la proportion de plis où la configuration gagne.
Les prédictions sont ramenées à ≥ 0 (la métrique officielle refuse les valeurs négatives).

    uv run python -m scripts.compare_extraction
    uv run python -m scripts.compare_extraction --edges 0 5 10 --repeats 1      # plus rapide
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from scripts.check_edge_channels import replace_edges
from scripts.compare_features import ridge_variant, with_sigma
from scripts.compare_models import bootstrap_scores, gll, load_spectra, to_arrays
from scripts.prototype_extraction import CURVES
from src.ml.features import compute_naive_stats
from src.models.hetero import Prediction, ridge_const
from src.signal_processing.transit_fit import extract_depths

N_WL = 283
EXTRACTIONS = [  # (nom, méthode, degré) ; méthode None = Rp du pipeline v0.1.0 (lu dans le npz)
    ("actuelle", None, 1),
    ("adaptée", "adaptee", 1),
    ("adaptée + droite", "adaptee_detrend", 1),
    ("gabarit + droite", "gabarit", 1),
    ("gabarit + parabole", "gabarit", 2),
]
REFERENCE = ("actuelle", 0)


# --------------------------------------------------------------------------- briques pures (testées)

def apply_edges(depth: np.ndarray, n_edge: int) -> np.ndarray:
    """depth : (283,) = [FGS1, AIRS...]. Remplace les n_edge premiers canaux AIRS par la moyenne des 10 suivants."""
    out = np.array(depth, dtype=float, copy=True)
    if n_edge > 0:
        out[1:] = replace_edges(out[1:], n_edge, slice(n_edge, n_edge + 10))
    return out


def depth_vector(curve: dict, method: str | None, deg: int) -> np.ndarray:
    """Profondeurs [FGS1, AIRS_1..282] d'une planète pour une méthode (None = pipeline v0.1.0)."""
    if method is None:
        return np.asarray(curve["rp"], dtype=float) ** 2
    return extract_depths(curve["airs"].astype(float), curve["fgs"].astype(float), method, deg)


def build_features(df: pd.DataFrame, depths: dict[str, np.ndarray]) -> pd.DataFrame:
    """Copie de df restreinte aux planètes de `depths`, avec Rp_i = sqrt(max(profondeur_i, 0))."""
    keep = df[df["planet_id"].astype(int).astype(str).isin(depths)].copy()
    cols = [f"Rp_{i}" for i in range(1, N_WL + 1)]
    mat = np.stack([np.sqrt(np.maximum(depths[str(int(p))], 0.0)) for p in keep["planet_id"]])
    keep[cols] = mat
    return keep


def paired_summary(diff: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Différences appariées (config − référence) sur les plis : moyenne, part de plis gagnants, IC 90 %."""
    diff = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
    return {"diff_moy": float(diff.mean()), "plis_gagnes": float((diff > 1e-6).mean()),
            "diff_ci05": float(np.percentile(boots, 5)), "diff_ci95": float(np.percentile(boots, 95))}


def fold_splits(n: int, n_splits: int, repeats: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Plis déterministes : identiques pour toutes les configurations (comparaison appariée)."""
    out = []
    for r in range(repeats):
        out += list(KFold(n_splits=min(n_splits, n), shuffle=True, random_state=seed + r).split(np.arange(n)))
    return out


def nonneg(fn):
    def f(Xtr, ytr, Xte):
        p = fn(Xtr, ytr, Xte)
        return Prediction(np.maximum(p.mu, 0.0), p.sigma)
    return f


def models() -> dict:
    mu_fn = ridge_variant(10.0, 2, True)
    return {"Rp² + astro, α=10, σ const.": nonneg(lambda a, b, c: with_sigma(mu_fn, a, b, c, "insample")),
            "Rp + astro, α=10 (modèle v0.1.0)": nonneg(ridge_const)}


# --------------------------------------------------------------------------- évaluation

def evaluate_config(X, y, ids, is_val, model, splits, boots) -> dict:
    tr, va = ~is_val, is_val
    nm, ns = compute_naive_stats(y[tr])
    pred = model(X[tr], y[tr], X[va])
    val = gll(y[va], pred, ids[va], nm, ns)
    val_boot = bootstrap_scores(y[va], pred, ids[va], nm, ns, boots)
    cv = []
    for a, b in splits:
        nm_f, ns_f = compute_naive_stats(y[a])
        p = model(X[a], y[a], X[b])
        cv.append(gll(y[b], p, ids[b], nm_f, ns_f))
    return {"val": val, "val_boot": val_boot, "cv": np.array(cv)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed/incremental")
    ap.add_argument("--curves", default=str(CURVES))
    ap.add_argument("--edges", type=int, nargs="*", default=[0, 5, 10, 15, 20])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/extraction")
    a = ap.parse_args()

    train_df, val_df = load_spectra(Path(a.root))
    all_df = pd.concat([val_df, train_df], ignore_index=True)
    val_ids = {str(int(p)) for p in val_df["planet_id"]}
    curves: dict[str, dict] = {}
    for p in all_df["planet_id"].astype(int).astype(str):
        f = Path(a.curves) / f"{p}.npz"
        if f.exists():
            z = np.load(f)
            curves[p] = {"airs": z["airs"], "fgs": z["fgs"], "rp": z["rp"]}
    n_val = len(val_ids & set(curves))
    print(f"{len(curves)} planètes avec courbes ({n_val} en validation fixe, {len(curves) - n_val} en entraînement)")
    if n_val < 5 or len(curves) - n_val < 20:
        raise SystemExit("Trop peu de planètes avec courbes (as-tu fini reprocess_curves ?).")

    # extraction une fois par méthode ; les variantes « bord » s'appliquent ensuite en mémoire
    base = {name: {p: depth_vector(c, m, d) for p, c in curves.items()} for name, m, d in EXTRACTIONS}
    configs, arrays = [], {}
    for name, _, _ in EXTRACTIONS:
        for n_edge in a.edges:
            depths = {p: apply_edges(v, n_edge) for p, v in base[name].items()}
            configs.append((name, n_edge))
            arrays[(name, n_edge)] = build_features(all_df, depths)
    # jeu commun de planètes (finies dans toutes les configurations), ordre commun
    common = None
    parts = {}
    for key, df in arrays.items():
        X, y, ids = to_arrays(df)
        parts[key] = (X, y, ids)
        s = set(map(str, ids.astype(int)))
        common = s if common is None else common & s
    order = [str(int(p)) for p in parts[configs[0]][2] if str(int(p)) in common]
    print(f"{len(order)} planètes communes à toutes les configurations")
    ids0 = np.array([int(p) for p in order])
    is_val = np.array([p in val_ids for p in order])
    splits = fold_splits(len(order), a.folds, a.repeats, a.seed)
    rng = np.random.default_rng(a.seed)
    boots = [rng.integers(0, int(is_val.sum()), int(is_val.sum())) for _ in range(a.n_boot)]

    rows, results = [], {}
    for key in configs:
        X, y, ids = parts[key]
        pos = {int(p): i for i, p in enumerate(ids)}
        idx = np.array([pos[int(p)] for p in ids0])
        for mname, model in models().items():
            results[(key, mname)] = evaluate_config(X[idx], y[idx], ids0, is_val, model, splits, boots)
        print(f"  {key[0]:20s} bord={key[1]:2d} évalué")
    for key in configs:
        for mname in models():
            r, ref = results[(key, mname)], results[(REFERENCE, mname)]
            s = paired_summary(r["cv"] - ref["cv"])
            rows.append({"extraction": key[0], "bord": key[1], "modèle": mname, "val": r["val"],
                         "val_ci05": float(np.percentile(r["val_boot"], 5)), "val_ci95": float(np.percentile(r["val_boot"], 95)),
                         "cv_moy": float(r["cv"].mean()), "cv_sd": float(r["cv"].std()), **s})
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 240, "display.max_colwidth", 40, "display.float_format", lambda v: f"{v:.4f}")
    for mname in models():
        t = table[table["modèle"] == mname].drop(columns="modèle").sort_values("cv_moy", ascending=False)
        print(f"\n=== {mname} ===  (référence : {REFERENCE[0]}, bord {REFERENCE[1]})")
        print(t.to_string(index=False))
    print("\ndiff_moy = score CV de la configuration − score CV de la référence (mêmes plis) ; plis_gagnes = part des plis "
          "où elle gagne ;\nIC 90 % par bootstrap sur les plis (plis non indépendants : optimiste). val = validation fixe du projet.")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "compare_extraction.csv", index=False)
    print(f"-> {out / 'compare_extraction.csv'}")


if __name__ == "__main__":
    main()
