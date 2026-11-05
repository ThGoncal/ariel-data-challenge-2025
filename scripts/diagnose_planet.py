"""
diagnose_planet.py — Pourquoi une planète est-elle sur-estimée par TOUTES les méthodes ?
=======================================================================================
Pour chaque planète (par défaut 2033197078 et 340799832), à partir des courbes déjà sauvegardées :
  - forme du fond du transit (courbure, asymétrie) sur la courbe blanche AIRS ;
  - profondeur mesurée sur des sous-fenêtres du fond (centre 30 %, centre 60 %, tout le fond) / profondeur vraie ;
  - bruit et pente hors transit ; rapport estimé/vrai pour FGS1 et pour AIRS ;
  - figure : courbe blanche + fenêtres, zoom sur le fond, rapport estimé/vrai par canal.

    uv run python -m scripts.diagnose_planet
    uv run python -m scripts.diagnose_planet --ids 2033197078
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.signal_processing.transit_fit import detrended, extract_depths, oot_mask, transit_window

RAW = Path("data/raw")
CURVES = Path("data/processed/transit_curves")


def bottom_stats(f: np.ndarray, st: dict) -> dict:
    """f : courbe blanche normalisée par la ligne de base (≈1 hors transit). Retourne des mesures de forme du fond."""
    seg = f[st["a_in"]: st["b_in"] + 1]
    n = len(seg)
    depth = 1.0 - float(np.median(seg))
    if n < 8:
        return {"n_fond": n, "courbure": np.nan, "asymetrie": np.nan, "pente_fond": np.nan}
    x = np.linspace(-1, 1, n)
    c2, c1, c0 = np.polyfit(x, 1.0 - seg, 2)
    half = n // 2
    return {"n_fond": n,
            # variation de profondeur entre le centre et les bords du fond, relative à la profondeur
            "courbure": float(-c2 / max(depth, 1e-12)),
            "asymetrie": float((np.median(seg[:half]) - np.median(seg[-half:])) / max(depth, 1e-12)),
            "pente_fond": float(c1 / max(depth, 1e-12))}


def sub_depths(f: np.ndarray, st: dict) -> dict:
    a, b = st["a_in"], st["b_in"]
    mid, half = (a + b) // 2, (b - a) // 2
    out = {}
    for name, frac in (("centre30", 0.3), ("centre60", 0.6), ("fond", 1.0)):
        h = max(int(half * frac), 2)
        out[name] = 1.0 - float(np.median(f[max(mid - h, 0): mid + h + 1]))
    return out


def analyse(pid: str, labels: pd.DataFrame) -> tuple[dict, dict]:
    z = np.load(CURVES / f"{pid}.npz")
    airs, fgs = z["airs"].astype(float), z["fgs"].astype(float)
    white = airs.sum(axis=1)
    st = transit_window(white)
    row = {"planet_id": pid, "detecte": st["detected"]}
    if not st["detected"]:
        return row, {}
    mask = oot_mask(len(white), st)
    f = detrended(white[:, None], mask, 1)[:, 0]
    true = labels.loc[int(pid)]
    true_air = float(true[[f"wl_{i}" for i in range(2, 284)]].astype(float).mean())
    true_fgs = float(true["wl_1"])
    row.update({k: st[k] for k in ("dur50", "dur10", "center_offset", "depth_bottom")})
    row.update(bottom_stats(f, st))
    row.update({f"vrai_vs_{k}": v / true_air for k, v in sub_depths(f, st).items()})
    row["bruit_hors_transit"] = float(1.4826 * np.median(np.abs(f[mask] - np.median(f[mask]))))
    row["vraie_profondeur"] = true_air
    for m, deg in (("fixe", 1), ("adaptee", 1), ("gabarit", 1)):
        d = extract_depths(airs, fgs, m, deg)
        row[f"airs_{m}/vrai"] = float(d[1:].mean() / true_air)
        row[f"fgs_{m}/vrai"] = float(d[0] / true_fgs)
    truth = true[[f"wl_{i}" for i in range(2, 284)]].astype(float).to_numpy()
    est = {m: extract_depths(airs, fgs, m, 1)[1:] for m in ("fixe", "gabarit")}
    for m, e in est.items():
        r = e / truth
        for a, b in ((0, 5), (5, 20), (20, 100), (100, 282)):
            row[f"{m}_med_canaux_{a}-{b}"] = float(np.median(r[a:b]))
        row[f"{m}_niveau_sans_5_premiers"] = float(e[5:].mean() / truth[5:].mean())
        row[f"{m}_poids_5_premiers_vrai"] = float(truth[:5].sum() / truth.sum())
        row[f"{m}_apport_5_premiers_est"] = float(e[:5].sum() / truth.sum())
    curves = {"f": f, "st": st, "ratio": est["gabarit"] / truth, "ratio_fixe": est["fixe"] / truth}
    return row, curves


def plot(path: Path, items: list[tuple[str, dict]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(len(items), 3, figsize=(15, 3.6 * len(items)), squeeze=False)
    for r, (pid, c) in enumerate(items):
        f, st = c["f"], c["st"]
        ax[r, 0].plot(f, lw=.5, color="k")
        ax[r, 0].axvspan(len(f) // 2 - len(f) // 10, len(f) // 2 + len(f) // 10, color="C1", alpha=.25, label="fenêtre fixe")
        ax[r, 0].axvspan(st["a_in"], st["b_in"], color="C2", alpha=.25, label="fond détecté")
        ax[r, 0].set_title(f"{pid} — courbe blanche (ligne de base retirée)", fontsize=9); ax[r, 0].legend(fontsize=7)
        a, b = st["a_in"], st["b_in"]
        pad = max((b - a) // 2, 10)
        xs = np.arange(max(a - pad, 0), min(b + pad, len(f)))
        ax[r, 1].plot(xs, f[xs], lw=.6, color="k")
        ax[r, 1].axvspan(a, b, color="C2", alpha=.2)
        ax[r, 1].set_title("zoom sur le fond du transit", fontsize=9)
        ax[r, 2].plot(c["ratio_fixe"], lw=.7, color="C1", label="fixe")
        ax[r, 2].plot(c["ratio"], lw=.7, color="C0", label="gabarit + droite")
        ax[r, 2].axhline(1, color="k", lw=.5); ax[r, 2].legend(fontsize=7)
        ax[r, 2].set_ylim(0.8, 2.0)   # le pic du premier canal dépasse souvent l'échelle : voir le CSV
        ax[r, 2].set_title("profondeur estimée / vraie, par canal AIRS", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=["2033197078", "340799832"])
    ap.add_argument("--out", default="reports/extraction")
    a = ap.parse_args()
    labels = pd.read_csv(RAW / "train.csv").set_index("planet_id")
    rows, items = [], []
    for pid in a.ids:
        if not (CURVES / f"{pid}.npz").exists():
            print(f"[{pid}] pas de courbes, ignorée")
            continue
        row, c = analyse(pid, labels)
        rows.append(row)
        if c:
            items.append((pid, c))
    table = pd.DataFrame(rows).set_index("planet_id").T
    pd.set_option("display.width", 200, "display.float_format", lambda v: f"{v:.4g}")
    print(table.to_string())
    print("\nLecture : courbure > 0 = fond arrondi (plus profond au centre) ; asymétrie ≠ 0 = fond incliné ;"
          "\n  vrai_vs_centre30 ~ vrai_vs_fond => profondeur uniforme sur le fond ; airs_*/vrai vs fgs_*/vrai => canal en cause.")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "diagnose_planet.csv")
    if items:
        plot(out / "diagnose_planet.png", items)
    print(f"-> {out}/")


if __name__ == "__main__":
    main()
