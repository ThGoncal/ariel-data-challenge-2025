"""
diagnose_errors.py — Où se concentre l'erreur du modèle Ridge ? (aucun téléchargement)
=======================================================================================
Deux questions, sur les spectres déjà traités (data/processed/incremental) :

  A. Qualité de l'EXTRACTION (avant tout modèle) : le spectre mesuré Rp_* ressemble-t-il à la
     cible wl_* ? Erreur systématique (rampe, décalage) contre aléatoire, comparée à la variation
     réelle du spectre d'une planète à l'autre.
  B. Erreur du MODÈLE (Ridge v0.1.0) sur la validation fixe : par groupe de canaux
     (FGS1 / rampe bleue AIRS / reste AIRS) et par planète (décalage de niveau ou erreur de forme),
     avec la contribution de chaque groupe au score GLL.

    uv run python -m scripts.diagnose_errors
    uv run python -m scripts.diagnose_errors --n-train 140 --ramp 10

Sorties : reports/diagnostic/{by_group,by_channel,by_planet,extraction}.csv + diagnostic.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats

from scripts.compare_models import load_spectra
from src.ml.features import build_feature_matrix, build_target_matrix, compute_naive_stats
from src.ml.metrics import evaluate_on_validation
from src.models.hetero import ridge_const

N_WL = 283
FGS_SIGMA, AIRS_SIGMA = 1e-6, 1e-5       # bruit instrumental de référence de la métrique officielle


def get_arrays(df: pd.DataFrame):
    """X, Rp (283 premières colonnes de X), y et ids, lignes finies uniquement."""
    X, _ = build_feature_matrix(df)
    y, _ = build_target_matrix(df)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y).all(axis=1)
    return X[ok], X[ok][:, :N_WL], y[ok], df["planet_id"].to_numpy()[ok]


def channel_groups(ramp: int = 10) -> dict[str, np.ndarray]:
    """FGS1 = canal 0 ; rampe bleue = les `ramp` premiers canaux AIRS ; le reste d'AIRS."""
    idx = np.arange(N_WL)
    return {"FGS1": idx[:1], f"AIRS bleu (1-{ramp})": idx[1:1 + ramp], "AIRS reste": idx[1 + ramp:]}


def cell_scores(y, mu, sigma, naive_mean, naive_sigma) -> np.ndarray:
    """Score GLL normalisé de CHAQUE cellule (planète × canal), formule de la métrique officielle."""
    sig_true = np.r_[FGS_SIGMA, np.full(N_WL - 1, AIRS_SIGMA)]
    sigma = np.clip(sigma, 1e-15, None)
    g_pred = scipy.stats.norm.logpdf(y, loc=mu, scale=sigma)
    g_true = scipy.stats.norm.logpdf(y, loc=y, scale=sig_true)
    g_mean = scipy.stats.norm.logpdf(y, loc=naive_mean, scale=naive_sigma)
    return (g_pred - g_mean) / (g_true - g_mean)


def pick_transform(Rp: np.ndarray, y: np.ndarray):
    """Rp est-il comparable à wl directement (Rp/Rs) ou après élévation au carré (profondeur) ?"""
    rel = lambda a: float(np.median(np.abs(a - y) / np.maximum(np.abs(y), 1e-12)))
    return ("Rp", Rp) if rel(Rp) <= rel(Rp ** 2) else ("Rp²", Rp ** 2)


def group_table(groups, E, y_tr, ytr_mean, y_va, S) -> pd.DataFrame:
    """Erreur du modèle par groupe de canaux : RMSE, compétence vs moyenne naïve, biais, contribution au score."""
    tot_sq, rows = float((E ** 2).sum()), []
    for name, idx in groups.items():
        e = E[:, idx]
        mse = float((e ** 2).mean())
        mse_naive = float(((y_va[:, idx] - ytr_mean[idx]) ** 2).mean())
        bias_sq = float((e.mean(axis=0) ** 2).mean())            # part systématique (même signe chez toutes les planètes)
        rows.append({"groupe": name, "n_canaux": len(idx), "rmse": mse ** 0.5,
                     "competence_vs_moyenne": 1 - mse / max(mse_naive, 1e-30),
                     "part_biais": bias_sq / max(mse, 1e-30),
                     "part_erreur_totale": float((e ** 2).sum()) / tot_sq,
                     "contribution_score": float(S[:, idx].sum()) / S.size})
    return pd.DataFrame(rows)


def planet_table(ids, E, S) -> pd.DataFrame:
    """Par planète : erreur de niveau (décalage commun à tous les canaux) contre erreur de forme."""
    level = E.mean(axis=1)
    shape = np.sqrt(((E - level[:, None]) ** 2).mean(axis=1))
    rmse = np.sqrt((E ** 2).mean(axis=1))
    out = pd.DataFrame({"planet_id": ids, "rmse": rmse, "niveau": level, "forme_rms": shape,
                        "part_niveau": level ** 2 / np.maximum(rmse ** 2, 1e-30), "score_moyen": S.mean(axis=1)})
    return out.sort_values("rmse", ascending=False).reset_index(drop=True)


def extraction_table(groups, Rp_t, y) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Erreur d'extraction D = Rp_t − wl sur toutes les planètes : systématique, aléatoire, vs variation vraie."""
    D = Rp_t - y
    bias_c, noise_c, true_c = D.mean(axis=0), D.std(axis=0), y.std(axis=0)
    per_channel = pd.DataFrame({"canal": np.arange(N_WL), "biais": bias_c, "bruit": noise_c, "variation_vraie": true_c})
    rows = []
    for name, idx in groups.items():
        rows.append({"groupe": name, "biais_rms": float(np.sqrt((bias_c[idx] ** 2).mean())),
                     "bruit_rms": float(np.sqrt((noise_c[idx] ** 2).mean())),
                     "variation_vraie_rms": float(np.sqrt((true_c[idx] ** 2).mean())),
                     "bruit_sur_variation": float(np.sqrt((noise_c[idx] ** 2).mean() / (true_c[idx] ** 2).mean()))})
    return pd.DataFrame(rows), per_channel


def plot(path, per_channel, E, y_va, planets, groups, unit) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    c = per_channel["canal"]
    ax[0, 0].plot(c, per_channel["variation_vraie"], label="variation vraie (écart-type de wl)")
    ax[0, 0].plot(c, per_channel["bruit"], label=f"bruit d'extraction (écart-type de {unit} − wl)")
    ax[0, 0].plot(c, np.sqrt((E ** 2).mean(axis=0)), label="erreur du Ridge (RMSE)")
    ax[0, 0].set_yscale("log"); ax[0, 0].set_xlabel("canal (0 = FGS1)")
    ax[0, 0].set_ylim(top=float(per_channel["variation_vraie"].max()) * 12)   # place pour la légende
    ax[0, 0].legend(fontsize=8, loc="upper center")
    ax[0, 0].set_title("A. Bruit contre signal, par canal")
    ax[0, 1].plot(c, per_channel["biais"], color="C3")
    ax[0, 1].axhline(0, color="k", lw=.5); ax[0, 1].set_xlabel("canal")
    ax[0, 1].set_title(f"A. Biais systématique de l'extraction : moyenne({unit} − wl)")
    p = planets.head(10)
    ax[1, 0].bar(p["planet_id"].astype(str), p["niveau"].abs(), label="|niveau|")
    ax[1, 0].bar(p["planet_id"].astype(str), p["forme_rms"], bottom=p["niveau"].abs(), label="forme (rms)", alpha=.7)
    ax[1, 0].tick_params(axis="x", rotation=60); ax[1, 0].legend(fontsize=8)
    ax[1, 0].set_title("B. 10 planètes les plus mal prédites (validation)")
    for name, idx in groups.items():
        ax[1, 1].plot(idx, np.sqrt((E[:, idx] ** 2).mean(axis=0)), ".", label=name, ms=4)
    ax[1, 1].set_yscale("log"); ax[1, 1].set_xlabel("canal"); ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_title("B. RMSE du Ridge par canal, par groupe")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed/incremental")
    ap.add_argument("--n-train", type=int, default=None, help="n premières planètes (défaut : toutes)")
    ap.add_argument("--ramp", type=int, default=10, help="nombre de canaux AIRS de la rampe bleue")
    ap.add_argument("--out", default="reports/diagnostic")
    a = ap.parse_args()

    train_df, val_df = load_spectra(Path(a.root))
    if a.n_train:
        train_df = train_df.iloc[:a.n_train]
    Xtr, Rp_tr, ytr, _ = get_arrays(train_df)
    Xva, Rp_va, yva, ids_va = get_arrays(val_df)
    groups = channel_groups(a.ramp)
    nm, ns = compute_naive_stats(ytr)

    pred = ridge_const(Xtr, ytr, Xva)
    E = yva - pred.mu
    S = cell_scores(yva, pred.mu, pred.sigma, nm, ns)
    official = float(evaluate_on_validation(yva, pred.mu, pred.sigma, ids_va, nm, ns))
    mine = float(np.clip(S.mean(), 0, 1))
    assert abs(official - mine) < 1e-6, f"incohérence métrique : {official} contre {mine}"

    by_group = group_table(groups, E, ytr, ytr.mean(axis=0), yva, S)
    by_planet = planet_table(ids_va, E, S)
    unit, Rp_t = pick_transform(np.vstack([Rp_tr, Rp_va]), np.vstack([ytr, yva]))
    extraction, per_channel = extraction_table(groups, Rp_t, np.vstack([ytr, yva]))

    pd.set_option("display.width", 200, "display.float_format", lambda v: f"{v:.4g}")
    print(f"\nRidge sur {len(Xtr)} planètes, validation {len(Xva)} planètes — score GLL {official:.4f}")
    print(f"\nA. EXTRACTION ({unit} comparé à wl, {len(Xtr) + len(Xva)} planètes)\n{extraction.to_string(index=False)}")
    print(f"\nB. MODÈLE par groupe de canaux\n{by_group.to_string(index=False)}")
    print(f"\nB. MODÈLE par planète (10 pires)\n{by_planet.head(10).to_string(index=False)}")
    part10 = (by_planet['rmse'].head(3) ** 2).sum() / (by_planet['rmse'] ** 2).sum()
    print(f"\nLes 3 pires planètes portent {part10:.0%} de l'erreur quadratique de la validation.")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    by_group.to_csv(out / "by_group.csv", index=False)
    by_planet.to_csv(out / "by_planet.csv", index=False)
    extraction.to_csv(out / "extraction.csv", index=False)
    per_channel.to_csv(out / "by_channel.csv", index=False)
    plot(out / "diagnostic.png", per_channel, E, yva, by_planet, groups, unit)
    print(f"-> {out}/")


if __name__ == "__main__":
    main()
