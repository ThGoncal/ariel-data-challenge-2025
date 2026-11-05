"""
prototype_extraction.py — L'extraction corrigée réduit-elle la dispersion du niveau d'une planète à l'autre ?
============================================================================================================
Travaille sur les courbes de lumière déjà sauvegardées par check_transit_window.py
(data/processed/transit_curves/*.npz : flux sommé sur l'espace, calibré et binné) : AUCUN téléchargement.

Méthodes comparées (profondeur par canal, voir src/signal_processing/transit_fit.py) :
  actuelle (pipeline)   Rp² du pipeline v0.1.0 (lu dans le npz)
  fixe reproduite       fenêtre ±10 % recalculée depuis les courbes : doit redonner « actuelle »
  adaptee               fenêtre détectée sur la courbe blanche
  adaptee + droite/parabole   idem, ligne de base polynomiale retirée avant les médianes
  gabarit + droite/parabole   ajustement du gabarit de transit de la courbe blanche

Chaque méthode est comparée aux vrais labels wl (train.csv) sur les 282 canaux AIRS. La profondeur du fond du
transit dépasse la profondeur géométrique (assombrissement du bord) d'un FACTEUR k à peu près constant : on le
retire par un k unique, estimé sur les AUTRES planètes (leave-one-out). Ce qui compte est donc :
  disp_niveau   écart-type de ln(niveau estimé / niveau vrai) entre planètes (0 = niveau toujours juste à k près)
  rmse_k_loo    erreur par cellule après rescalage k (leave-one-out) : directement liée au score GLL
  rmse_forme    erreur de forme spectrale (niveau de chaque planète retiré) : ne doit pas se dégrader

    uv run python -m scripts.prototype_extraction
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.signal_processing.transit_fit import extract_depths

RAW = Path("data/raw")
CURVES = Path("data/processed/transit_curves")

METHODS = [  # (nom affiché, méthode, degré)
    ("fixe reproduite", "fixe", 1),
    ("adaptée", "adaptee", 1),
    ("adaptée + droite", "adaptee_detrend", 1),
    ("adaptée + parabole", "adaptee_detrend", 2),
    ("gabarit + droite", "gabarit", 1),
    ("gabarit + parabole", "gabarit", 2),
]


def evaluate(est: np.ndarray, truth: np.ndarray) -> dict:
    """est, truth : (n_planètes, n_canaux AIRS). Niveau = moyenne sur les canaux."""
    level = np.maximum(est.mean(axis=1), 1e-12) / truth.mean(axis=1)
    ln = np.log(level)
    p = len(ln)
    k_all = float(np.exp(-ln.mean()))
    err_loo = []
    for i in range(p):
        k_i = float(np.exp(-np.delete(ln, i).mean())) if p > 1 else 1.0
        err_loo.append(k_i * est[i] - truth[i])
    # k PAR CANAL (leave-one-out, médiane des rapports des autres planètes) : absorbe l'assombrissement du bord,
    # plus fort dans le bleu, et c'est ce que le Ridge apprend ensuite ; robuste aux canaux aberrants
    ratio = est / truth
    err_kj = []
    for i in range(p):
        kj = 1.0 / np.median(np.delete(ratio, i, axis=0), axis=0) if p > 1 else np.ones(est.shape[1])
        err_kj.append(kj * est[i] - truth[i])
    err_kj = np.asarray(err_kj)
    level_med = np.median(ratio, axis=1)
    # forme : on ramène chaque planète à son propre niveau vrai (niveau estimé divisé par son rapport)
    est_c = (est - est.mean(axis=1, keepdims=True)) / level[:, None]
    tru_c = truth - truth.mean(axis=1, keepdims=True)
    return {"biais_k": k_all, "disp_niveau": float(ln.std()),
            "rmse_k_loo": float(np.sqrt(np.mean(np.square(err_loo)))),
            "rmse_forme": float(np.sqrt(np.mean((est_c - tru_c) ** 2))),
            "rmse_kj_loo": float(np.sqrt(np.mean(err_kj ** 2))),
            "rmse_kj_loo_sans5": float(np.sqrt(np.mean(err_kj[:, 5:] ** 2))),
            "disp_niveau_med": float(np.log(level_med).std()), "level": level, "level_med": level_med}


def load_all(curves_dir: Path, labels_csv: Path, ids: list[str] | None):
    labels = pd.read_csv(labels_csv).set_index("planet_id")
    files = sorted(curves_dir.glob("*.npz")) if not ids else [curves_dir / f"{i}.npz" for i in ids]
    data = []
    for f in files:
        if not f.exists():
            print(f"[{f.stem}] pas de courbes, ignorée")
            continue
        z = np.load(f)
        pid = int(f.stem)
        truth = labels.loc[pid, [f"wl_{i}" for i in range(1, 284)]].to_numpy(dtype=float)
        data.append({"planet_id": pid, "airs": z["airs"].astype(float), "fgs": z["fgs"].astype(float),
                     "rp": z["rp"].astype(float), "truth": truth})
    return data


def plot(path: Path, data, estimates: dict, table: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(estimates)
    cols = plt.get_cmap("tab10")
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    ids = [str(d["planet_id"]) for d in data]
    for m, name in enumerate(names):
        lv = table.loc[name, "level"]
        ax[0, 0].plot(np.arange(len(ids)) + (m - len(names) / 2) * 0.09, lv, "o", ms=4, color=cols(m), label=name)
    ax[0, 0].axhline(1, color="k", lw=.5)
    ax[0, 0].set_xticks(range(len(ids)), ids, rotation=60, fontsize=7)
    ax[0, 0].set_ylabel("niveau estimé / niveau vrai"); ax[0, 0].legend(fontsize=6, ncol=2)
    ax[0, 0].set_title("Niveau par planète (1 = juste ; le facteur k constant est retiré plus loin)")
    ax[0, 1].bar(range(len(names)), [table.loc[n, "disp_niveau"] for n in names], color=[cols(i) for i in range(len(names))])
    ax[0, 1].set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=7)
    ax[0, 1].set_title("Dispersion du niveau entre planètes (plus bas = mieux)")
    ax[1, 0].bar(range(len(names)), [table.loc[n, "rmse_k_loo"] for n in names], color=[cols(i) for i in range(len(names))])
    ax[1, 0].set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=7)
    ax[1, 0].set_yscale("log"); ax[1, 0].set_title("Erreur par cellule après rescalage k (leave-one-out)")
    # spectres de la planète au niveau le plus mal estimé par « actuelle »
    ref = names[0]
    w = int(np.argmax(np.abs(np.log(table.loc[ref, "level"]))))
    best = table.drop(index=ref)["rmse_k_loo"].idxmin()
    truth = data[w]["truth"][1:]
    for name, color in ((ref, "C3"), (best, "C2")):
        est = estimates[name][w]
        ax[1, 1].plot(est * table.loc[name, "biais_k"], color=color, lw=.8, label=f"{name} × k")
    ax[1, 1].plot(truth, "k", lw=1.2, label="vrai")
    ax[1, 1].set_title(f"Spectre de la planète {ids[w]} (niveau le plus faux avec « {ref} »)")
    ax[1, 1].set_xlabel("canal AIRS"); ax[1, 1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--curves", default=str(CURVES))
    ap.add_argument("--labels", default=str(RAW / "train.csv"))
    ap.add_argument("--out", default="reports/extraction")
    a = ap.parse_args()

    data = load_all(Path(a.curves), Path(a.labels), a.ids)
    if len(data) < 3:
        raise SystemExit("Il faut au moins 3 planètes avec des courbes (lance d'abord check_transit_window).")
    truth = np.stack([d["truth"] for d in data])[:, 1:]            # canaux AIRS (le FGS1 est exclu)
    estimates = {"actuelle (pipeline)": np.stack([d["rp"] for d in data])[:, 1:] ** 2}
    for name, method, deg in METHODS:
        estimates[name] = np.stack([extract_depths(d["airs"], d["fgs"], method, deg)[1:] for d in data])

    repro = np.abs(estimates["fixe reproduite"] / estimates["actuelle (pipeline)"] - 1)
    print(f"{len(data)} planètes. Contrôle : « fixe reproduite » vs pipeline, écart relatif médian "
          f"{np.median(repro):.1e}, max {np.nanmax(repro):.1e}")

    rows = {n: evaluate(e, truth) for n, e in estimates.items()}
    table = pd.DataFrame(rows).T
    table["level"] = [rows[n]["level"] for n in table.index]
    view = table.drop(columns=["level", "level_med"]).astype(float)
    view["gain_k"] = view.loc["actuelle (pipeline)", "rmse_k_loo"] / view["rmse_k_loo"]
    view["gain_kj"] = view.loc["actuelle (pipeline)", "rmse_kj_loo"] / view["rmse_kj_loo"]
    pd.set_option("display.width", 200, "display.float_format", lambda v: f"{v:.4g}")
    print("\n" + view.to_string())
    lv = pd.DataFrame({n: rows[n]["level"] for n in rows}, index=[d["planet_id"] for d in data])
    print("\nNiveau estimé / niveau vrai, par planète, MOYENNE sur les canaux (sensible aux canaux de bord) :")
    print(lv.to_string())
    lvm = pd.DataFrame({n: rows[n]["level_med"] for n in rows}, index=[d["planet_id"] for d in data])
    print("\nIdem, MÉDIANE sur les canaux (robuste ; un facteur ~1,1 est normal) :")
    print(lvm.to_string())
    print("\nOrdre de grandeur du score : environ +0,13 chaque fois que l'erreur par cellule est divisée par 2,7 "
          "(calcul approché de la métrique) ; à confirmer sur la validation complète.")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    view.to_csv(out / "extraction_methods.csv")
    lv.to_csv(out / "extraction_levels.csv")
    lvm.to_csv(out / "extraction_levels_median.csv")
    plot(out / "extraction_methods.png", data, estimates, table)
    print(f"-> {out}/")


if __name__ == "__main__":
    main()
