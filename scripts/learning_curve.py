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
    uv run python -m scripts.learning_curve --plot docs/learning_curve.png          # calcule, écrit le CSV et trace
    uv run python -m scripts.learning_curve --plot-from reports/extraction/learning_curve.csv --plot docs/learning_curve.png
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
    splits = [list(KFold(n_splits=min(folds, len(all_df)), shuffle=True, random_state=seed + r).split(all_df))
              for r in range(repeats)]
    pool = min(len(a) for sp in splits for a, _ in sp)       # plus petit jeu d'entraînement : tailles communes à tous les plis
    for r in range(repeats):
        for k, (a, b) in enumerate(splits[r]):
            order = np.random.default_rng(seed * 1000 + r * 100 + k).permutation(a)       # ordre imbriqué
            test = all_df.iloc[b]
            for n in sorted({min(int(s), pool) for s in sizes}):
                bundle = fit_champion(all_df.iloc[order[:n]], alpha=alpha, power=power)
                rows.append({"repeat": r, "fold": k, "n": n, "score": score_champion(bundle, test)})
    return pd.DataFrame(rows)


def default_sizes(n_total: int, folds: int = 5, start: int = 20) -> list[int]:
    """Tailles doublées (20, 40, 80, ...) jusqu'au plus petit jeu d'entraînement d'un pli."""
    pool = n_total - int(np.ceil(n_total / folds))
    sizes = [start]
    while sizes[-1] * 2 < pool:
        sizes.append(sizes[-1] * 2)
    return sizes + [pool]


def plot_curve(summary: pd.DataFrame, path: Path, title: str | None = None, n_total: int | None = None) -> Path:
    """Score moyen en fonction du nombre de planètes d'entraînement ; bande = ± 1 écart-type entre plis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = summary["n"].to_numpy(float)
    m = summary["score_moyen"].to_numpy(float)
    sd = summary["ecart_type_plis"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=150)
    ax.fill_between(x, m - sd, m + sd, color="#3b6ea5", alpha=0.15, linewidth=0, label="± 1 écart-type entre plis")
    ax.plot(x, m, "o-", color="#3b6ea5", linewidth=2, markersize=6, label="score moyen (validation croisée)")
    for xi, mi in zip(x, m):
        ax.annotate(f"{mi:.3f}", (xi, mi), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x])
    ax.minorticks_off()
    ax.set_xlabel("Planètes d'entraînement (échelle logarithmique)")
    ax.set_ylabel("Score GLL normalisé")
    ax.set_title(title or "Score en fonction du nombre de planètes — Ridge, extraction adaptative", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    note = "mêmes plis de test pour toutes les tailles" + (f" · {n_total} planètes au total" if n_total else "")
    fig.text(0.01, 0.01, note, fontsize=7, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


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
    ap.add_argument("--sizes", type=int, nargs="+", default=None,
                    help="tailles d'entraînement (défaut : 20, 40, 80, ... jusqu'au maximum disponible par pli)")
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--plot", metavar="PNG", default=None, help="trace aussi le graphique dans ce fichier")
    ap.add_argument("--plot-from", metavar="CSV", default=None,
                    help="trace --plot depuis un CSV déjà calculé (aucun recalcul)")
    a = ap.parse_args()
    if a.plot_from:
        if not a.plot:
            raise SystemExit("--plot-from nécessite --plot")
        print("graphique :", plot_curve(pd.read_csv(a.plot_from), Path(a.plot)))
        return
    train, val = load_layout(Path(a.root))
    all_df = pd.concat([val, train], ignore_index=True)
    sizes = a.sizes or default_sizes(len(all_df), a.folds)
    res = learning_curve(all_df, sizes, a.alpha, 2, a.folds, a.repeats)
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
    if a.plot:
        print("graphique :", plot_curve(s, Path(a.plot), n_total=len(all_df)))


if __name__ == "__main__":
    main()
