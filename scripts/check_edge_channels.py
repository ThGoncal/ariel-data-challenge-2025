"""
check_edge_channels.py — Pourquoi les 5 premiers canaux AIRS sont-ils si mal estimés ?
======================================================================================
Constat (prototype_extraction) : ces 1,8 % du spectre portent presque toute l'erreur restante (RMSE ~0,045
contre ~0,001 ailleurs). Ce script, sur les courbes déjà sauvegardées (aucun téléchargement), teste :

  1. le niveau de flux et le bruit relatif des premiers canaux (flux très faible ?) ;
  2. l'hypothèse « décalage additif » : un flux mesuré F + c (fond / courant d'obscurité mal soustrait) donne
     une profondeur estimée d·F/(F+c) ; donc 1/ratio − 1 = c/F. On ajuste c sur les canaux 0..N, avec un terme
     linéaire en numéro de canal qui représente l'assombrissement du bord (physique), et on regarde si le terme
     c/F améliore vraiment l'ajustement (R²) et si c a le même signe et le même ordre de grandeur d'une planète
     à l'autre ;
  3. trois correctifs candidats sur l'erreur de ces canaux (k par canal leave-one-out, comme à l'entraînement) :
       brut | plafond (2× la médiane des canaux 5-14) | remplacement par la moyenne des canaux 5-14.

    uv run python -m scripts.check_edge_channels
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.prototype_extraction import CURVES, RAW, load_all
from src.signal_processing.transit_fit import extract_depths, oot_mask, transit_window

N_EDGE = 5            # canaux « de bord » étudiés
REF = slice(5, 15)    # canaux voisins de référence
N_FIT = 40            # canaux utilisés pour l'ajustement du décalage additif


# --------------------------------------------------------------------------- briques pures (testées)

def offset_fit(ratio: np.ndarray, flux: np.ndarray, n: int = N_FIT) -> dict:
    """1/ratio − 1 = c/F + a + b·j sur les n premiers canaux. flux : normalisé par la médiane des canaux (≈1).
    Retourne c (en unités du flux médian), R² sans et avec le terme c/F."""
    r, f = np.asarray(ratio[:n], float), np.asarray(flux[:n], float)
    ok = (r > 0) & (f > 0)
    if ok.sum() < 8:
        return {"c": np.nan, "r2_sans": np.nan, "r2_avec": np.nan}
    j = np.arange(len(r))[ok].astype(float)
    y = 1.0 / r[ok] - 1.0
    base = np.c_[np.ones_like(j), j]
    full = np.c_[1.0 / f[ok], base]
    sst = float(((y - y.mean()) ** 2).sum()) or 1e-30

    def fit(X):
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return coef, 1.0 - float(((y - X @ coef) ** 2).sum()) / sst

    _, r2_sans = fit(base)
    coef, r2_avec = fit(full)
    return {"c": float(coef[0]), "r2_sans": r2_sans, "r2_avec": r2_avec}


def cap_edges(est: np.ndarray, n_edge: int = N_EDGE, ref: slice = REF, factor: float = 2.0) -> np.ndarray:
    out = np.array(est, dtype=float, copy=True)
    out[:n_edge] = np.minimum(out[:n_edge], factor * float(np.median(out[ref])))
    return out


def replace_edges(est: np.ndarray, n_edge: int = N_EDGE, ref: slice = REF) -> np.ndarray:
    out = np.array(est, dtype=float, copy=True)
    out[:n_edge] = float(np.mean(out[ref]))
    return out


def kj_errors(est: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Erreur par cellule après un k par canal (médiane des rapports des AUTRES planètes). est, truth : (p, c)."""
    ratio = est / truth
    out = []
    for i in range(len(est)):
        kj = 1.0 / np.median(np.delete(ratio, i, axis=0), axis=0)
        out.append(kj * est[i] - truth[i])
    return np.asarray(out)


def rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


# --------------------------------------------------------------------------- analyse

def channel_flux(airs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flux médian et bruit relatif hors transit, par canal."""
    white = airs.sum(axis=1)
    mask = oot_mask(len(white), transit_window(white))
    seg = airs[mask]
    med = np.median(seg, axis=0)
    first, second = np.array_split(seg, 2)
    noise = np.std(seg - np.median(seg, axis=0), axis=0) / np.maximum(np.abs(med), 1e-12)
    return med, noise


def plot(path: Path, ids, flux, ratio, offset_pts) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for pid, f, r in zip(ids, flux, ratio):
        ax[0].plot(f[:40], lw=.8, label=str(pid))
        ax[1].plot(r[:40], lw=.8)
    ax[0].set_yscale("log"); ax[0].set_title("flux médian / flux médian des canaux (40 premiers canaux)")
    ax[0].legend(fontsize=6); ax[0].set_xlabel("canal AIRS")
    ax[1].set_ylim(0.8, 6); ax[1].axhline(1, color="k", lw=.5)
    ax[1].set_title("profondeur estimée / vraie (gabarit + droite)"); ax[1].set_xlabel("canal AIRS")
    for pid, (x, y) in zip(ids, offset_pts):
        ax[2].plot(x, y, ".", ms=4, label=str(pid))
    ax[2].set_xlabel("1 / flux relatif"); ax[2].set_ylabel("1/ratio − 1")
    ax[2].set_title("hypothèse décalage additif : droite de pente c ?")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", default=str(CURVES))
    ap.add_argument("--labels", default=str(RAW / "train.csv"))
    ap.add_argument("--out", default="reports/extraction")
    a = ap.parse_args()
    data = load_all(Path(a.curves), Path(a.labels), None)
    if len(data) < 3:
        raise SystemExit("Il faut au moins 3 planètes avec des courbes.")
    truth = np.stack([d["truth"] for d in data])[:, 1:]
    est = np.stack([extract_depths(d["airs"], d["fgs"], "gabarit", 1)[1:] for d in data])
    ids = [d["planet_id"] for d in data]

    rows, flux_all, offset_pts = [], [], []
    for d, e, t in zip(data, est, truth):
        med, noise = channel_flux(d["airs"])
        rel = med / np.median(med)
        flux_all.append(rel)
        ratio = e / t
        fit = offset_fit(ratio, rel)
        ok = (ratio[:N_FIT] > 0) & (rel[:N_FIT] > 0)
        offset_pts.append((1.0 / rel[:N_FIT][ok], 1.0 / ratio[:N_FIT][ok] - 1.0))
        rows.append({"planet_id": d["planet_id"], "flux_c0": rel[0], "flux_c1": rel[1], "flux_c4": rel[4],
                     "flux_c10": rel[10], "flux_min_temps_c0": float(d["airs"][:, 0].min() / np.median(d["airs"][:, 0])),
                     "bruit_rel_c0": float(noise[0]), "bruit_rel_c10": float(noise[10]), "bruit_rel_median": float(np.median(noise)),
                     "ratio_c0": ratio[0], "ratio_c1": ratio[1], "ratio_c4": ratio[4], "ratio_c10": ratio[10],
                     "decalage_c": fit["c"], "r2_sans_terme_c": fit["r2_sans"], "r2_avec_terme_c": fit["r2_avec"]})
    table = pd.DataFrame(rows).set_index("planet_id")
    pd.set_option("display.width", 220, "display.float_format", lambda v: f"{v:.4g}")
    print("Flux, bruit et hypothèse « décalage additif » (c en unités du flux médian des canaux ; c<0 = sur-soustraction)\n")
    print(table.T.to_string())
    c = table["decalage_c"]
    print(f"\ndécalage c : médiane {c.median():.3g}, signes identiques sur {int((np.sign(c) == np.sign(c.median())).sum())}/{len(c)} planètes ; "
          f"R² médian sans / avec terme c : {table['r2_sans_terme_c'].median():.2f} / {table['r2_avec_terme_c'].median():.2f}")

    variants = {"brut": est,
                "plafond 2×": np.stack([cap_edges(e) for e in est]),
                "remplacé par moy. canaux 5-14": np.stack([replace_edges(e) for e in est])}
    res = {}
    for name, v in variants.items():
        err = kj_errors(v, truth)
        res[name] = {"rmse_canaux_0-4": rmse(err[:, :N_EDGE]), "rmse_canaux_5+": rmse(err[:, N_EDGE:]), "rmse_total": rmse(err)}
    cmp = pd.DataFrame(res).T
    cmp["gain_total_vs_brut"] = cmp.loc["brut", "rmse_total"] / cmp["rmse_total"]
    print("\nCorrectifs candidats (k par canal leave-one-out) :\n")
    print(cmp.to_string())

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "edge_channels.csv")
    cmp.to_csv(out / "edge_channels_fixes.csv")
    plot(out / "edge_channels.png", ids, flux_all, [e / t for e, t in zip(est, truth)], offset_pts)
    print(f"-> {out}/")


if __name__ == "__main__":
    main()
