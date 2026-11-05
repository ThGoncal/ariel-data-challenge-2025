"""
learning_curve.py — Le score progresse-t-il quand on ajoute des planètes d'entraînement ?
=========================================================================================
Sur les planètes de data/processed/spectra_v2 (validation + entraînement), validation croisée répétée : pour chaque
pli, le modèle est entraîné sur les n PREMIÈRES planètes d'un ordre aléatoire du reste (sous-ensembles imbriqués :
la courbe ne peut pas varier par hasard de tirage entre deux n) puis testé sur le pli. Les plis de test sont les mêmes
pour tous les n : on lit donc la progression par DIFFÉRENCES appariées entre tailles successives.

Lecture : si le gain entre les deux dernières tailles reste nettement positif (bien plus grand que son erreur
standard), ajouter des planètes a des chances de servir ; s'il est proche de zéro, le facteur limitant est ailleurs
(extraction, modèle, bruit).

    uv run python -m scripts.learning_curve
    uv run python -m scripts.learning_curve --sizes 20 40 80 128 --repeats 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.models.champion import fit_champion, score_champion
from src.pipelines.train_champion import load_layout

ROOT = Path("data/processed/spectra_v2")
OUT = Path("reports/extraction/learning_curve.csv")


def learning_curve(all_df: pd.DataFrame, sizes: list[int], alpha: float = 10.0, power: int = 2,
                   folds: int = 5, repeats: int = 3, seed: int = 42) -> pd.DataFrame:
    """Une ligne par (répétition, pli, taille) : score GLL normalisé sur le pli de test."""
    rows = []
    for r in range(repeats):
        for k, (a, b) in enumerate(KFold(n_splits=min(folds, len(all_df)), shuffle=True, random_state=seed + r).split(all_df)):
            order = np.random.default_rng(seed * 1000 + r * 100 + k).permutation(a)       # ordre imbriqué
            test = all_df.iloc[b]
            for n in sorted({min(int(s), len(order)) for s in sizes}):
                bundle = fit_champion(all_df.iloc[order[:n]], alpha=alpha, power=power)
                rows.append({"repeat": r, "fold": k, "n": n, "score": score_champion(bundle, test)})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Score moyen par taille, et gain apparié par rapport à la taille précédente (moyenne et erreur standard)."""
    wide = df.pivot_table(index=["repeat", "fold"], columns="n", values="score")
    ns = list(wide.columns)
    out = []
    for i, n in enumerate(ns):
        row = {"n": n, "score_moyen": float(wide[n].mean()), "ecart_type_plis": float(wide[n].std())}
        if i:
            d = wide[n] - wide[ns[i - 1]]
            row.update({"gain_vs_precedent": float(d.mean()), "erreur_std_gain": float(d.std() / np.sqrt(len(d))),
                        "plis_gagnes": float((d > 1e-6).mean())})
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--sizes", type=int, nargs="+", default=[20, 40, 60, 80, 100, 128])
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    train, val = load_layout(Path(a.root))
    all_df = pd.concat([val, train], ignore_index=True)
    res = learning_curve(all_df, a.sizes, a.alpha, 2, a.folds, a.repeats)
    s = summarize(res)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(a.out, index=False)
    pd.set_option("display.width", 140)
    print(f"{len(all_df)} planètes, {a.folds} plis × {a.repeats} répétitions, alpha={a.alpha} (même alpha pour tous les n)")
    print(s.round(4).to_string(index=False))
    last = s.iloc[-1]
    if "gain_vs_precedent" in last and not np.isnan(last["gain_vs_precedent"]):
        g, e = last["gain_vs_precedent"], last["erreur_std_gain"]
        print(f"\nDernier palier : {g:+.4f} ± {e:.4f} ({g / e:+.1f} erreurs standard)" if e > 0 else f"\nDernier palier : {g:+.4f}")


if __name__ == "__main__":
    main()
