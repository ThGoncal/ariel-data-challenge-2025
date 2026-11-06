"""
diagnose_outliers.py — Quelles planètes tirent le score vers le bas, et pourquoi ? (aucun téléchargement)
========================================================================================================
La métrique officielle est la MOYENNE (clippée une seule fois à la fin) du score de toutes les cellules
planète × canal. Une log-vraisemblance gaussienne est quadratique : une planète mal prédite avec un σ trop petit
contribue des scores très négatifs et peut, à elle seule, dominer la moyenne.

Ce script, sur data/processed/spectra_v2, calcule le score de CHAQUE planète hors échantillon (validation croisée
répétée, la planète n'est jamais dans l'entraînement de son propre modèle) puis répond à :

  1. À quel point quelques planètes dominent-elles ?  (score moyen si l'on retire les k pires)
  2. Pourquoi sont-elles mauvaises ?  extraction fausse (niveau mesuré ≠ vrai) ou modèle (niveau prédit ≠ vrai),
     σ trop petit (part de cellules à plus de 3 σ), profondeur très grande, astro hors des plages d'entraînement ?

    uv run python -m scripts.diagnose_outliers
    uv run python -m scripts.diagnose_outliers --top 20

Sorties : reports/diagnostic/outliers.csv (une ligne par planète) et un résumé imprimé.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.model_selection import KFold

from scripts.diagnose_errors import N_WL, cell_scores, channel_groups
from src.ml.features import build_feature_matrix, build_target_matrix
from src.models.champion import fit_champion, predict_champion
from src.pipelines.train_champion import load_layout

ROOT = Path("data/processed/spectra_v2")
OUT = Path("reports/diagnostic/outliers.csv")
ASTRO = ["Ms", "Mp", "P", "sma", "i", "L_stefan"]


def planet_table(all_df: pd.DataFrame, alpha: float = 10.0, power: int = 2, folds: int = 5, repeats: int = 3,
                 seed: int = 42, ramp: int = 10) -> pd.DataFrame:
    """Une ligne par planète : score hors échantillon (moyenne des répétitions) et indicateurs d'erreur."""
    X, cols = build_feature_matrix(all_df)
    y, _ = build_target_matrix(all_df)
    astro = X[:, N_WL:]
    depth_meas = X[:, :N_WL] ** 2                          # profondeur mesurée par l'extraction (Rp²)
    groups = channel_groups(ramp)
    az = np.abs((astro - astro.mean(axis=0)) / np.where(astro.std(axis=0) > 0, astro.std(axis=0), 1.0))

    acc: dict[int, list[dict]] = {i: [] for i in range(len(all_df))}
    for r in range(repeats):
        for a, b in KFold(n_splits=min(folds, len(all_df)), shuffle=True, random_state=seed + r).split(all_df):
            bundle = fit_champion(all_df.iloc[a], alpha=alpha, power=power)
            pred = predict_champion(bundle, all_df.iloc[b])
            cs = cell_scores(y[b], pred.mu, pred.sigma, bundle["naive_mean"], bundle["naive_sigma"])
            z = np.abs(y[b] - pred.mu) / pred.sigma
            for k, i in enumerate(b):
                row = {"score": cs[k].mean(), "frac_z3": float((z[k] > 3).mean()),
                       "niveau_modele": float(np.median(pred.mu[k] / y[i]))}
                row.update({f"score_{g}": float(cs[k][idx].mean()) for g, idx in groups.items()})
                acc[i].append(row)

    rows = []
    for i in range(len(all_df)):
        m = pd.DataFrame(acc[i]).mean()
        rows.append({"planet_id": int(all_df["planet_id"].iloc[i]), **m.to_dict(),
                     "profondeur_med": float(np.median(y[i])),
                     "niveau_extraction": float(np.median(depth_meas[i] / y[i])),
                     "astro_zmax": float(az[i].max()), "astro_zmax_var": ASTRO[int(az[i].argmax())]})
    return pd.DataFrame(rows).sort_values("score").reset_index(drop=True)


def without_worst(table: pd.DataFrame, ks=(0, 1, 2, 5, 10, 20)) -> pd.DataFrame:
    """Score moyen (clippé comme la métrique) si l'on retire les k pires planètes."""
    s = np.sort(table["score"].to_numpy())
    return pd.DataFrame({"k_pires_retirees": [k for k in ks if k < len(s)],
                         "score": [float(np.clip(s[k:].mean(), 0, 1)) for k in ks if k < len(s)]})


def correlations(table: pd.DataFrame) -> pd.DataFrame:
    """Corrélations de rang (Spearman) entre le score par planète et des indicateurs candidats."""
    cand = {"profondeur_med": table["profondeur_med"],
            "|log niveau_extraction|": np.abs(np.log(table["niveau_extraction"].clip(lower=1e-6))),
            "|log niveau_modele|": np.abs(np.log(table["niveau_modele"].clip(lower=1e-6))),
            "astro_zmax": table["astro_zmax"]}
    out = []
    for name, v in cand.items():
        rho, p = scipy.stats.spearmanr(v, table["score"])
        out.append({"indicateur": name, "rho_spearman": float(rho), "p": float(p)})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    train, val = load_layout(Path(a.root))
    all_df = pd.concat([val, train], ignore_index=True)
    t = planet_table(all_df, a.alpha, 2, 5, a.repeats)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(f"{len(t)} planètes, score hors échantillon (5 plis × {a.repeats}) : moyenne {t['score'].mean():.3f}, "
          f"médiane {t['score'].median():.3f}, min {t['score'].min():.2f}")
    print(f"planètes à score < 0 : {int((t['score'] < 0).sum())}   |   < 0,2 : {int((t['score'] < 0.2).sum())}")
    print("\n1. Le score moyen si l'on retire les k pires planètes :")
    print(without_worst(t).round(3).to_string(index=False))
    cols = ["planet_id", "score", "frac_z3", "profondeur_med", "niveau_extraction", "niveau_modele",
            *[c for c in t.columns if c.startswith("score_")], "astro_zmax", "astro_zmax_var"]
    print(f"\n2. Les {a.top} pires planètes (niveau = médiane des rapports ; 1 = parfait) :")
    print(t[cols].head(a.top).round(3).to_string(index=False))
    print("\n3. Corrélation de rang entre le score par planète et des indicateurs :")
    print(correlations(t).round(3).to_string(index=False))
    print(f"\nTableau complet : {a.out}")


if __name__ == "__main__":
    main()
