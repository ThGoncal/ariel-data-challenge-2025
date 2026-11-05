"""
check_transit_window.py — La fenêtre de transit FIXE explique-t-elle l'erreur de niveau ?
==========================================================================================
Hypothèse (voir diagnose_errors.py) : calcul_Rp_median prend « dans le transit » = ±10 % autour
du milieu de la série, pour toutes les planètes. Si la durée réelle du transit diffère de cette
fenêtre, la profondeur est faussée d'un montant qui varie d'une planète à l'autre.

Pour N planètes (par défaut : les 3 pires de la validation + 5 planètes typiques, lues dans
reports/diagnostic/by_planet.csv), le script :
  1. télécharge le brut, traite le signal, sauvegarde les COURBES de lumière (npz, réutilisables)
     puis supprime le brut ;
  2. mesure sur la courbe de lumière blanche la durée et la position réelles du transit ;
  3. compare la profondeur « fenêtre fixe » (méthode actuelle) à une profondeur « fenêtre adaptée »,
     par rapport à la vraie profondeur moyenne (labels train.csv) ;
  4. trace les courbes avec la fenêtre fixe et la fenêtre détectée.

    uv run python -m scripts.check_transit_window                 # 3 pires + 5 typiques (~35 min)
    uv run python -m scripts.check_transit_window --ids 340799832 2199983617
    uv run python -m scripts.check_transit_window --reuse         # refait l'analyse sans retélécharger

Attention : la médiane est robuste. La fenêtre fixe ne fausse la profondeur que pour un transit très
court (< ~12 % de la série) ou très long (> ~50 %) ; entre les deux, l'erreur vient d'ailleurs.

Lecture : le rapport estimé/vrai n'est pas égal à 1 même avec une fenêtre parfaite (assombrissement
du bord de l'étoile : le fond du transit est plus profond que la profondeur géométrique). Ce qui
compte est la DISPERSION de ce rapport d'une planète à l'autre : un rapport presque constant se
corrige par un facteur, un rapport qui varie d'une planète à l'autre donne une erreur de niveau.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
CURVES = Path("data/processed/transit_curves")


# --------------------------------------------------------------------------- analyse (pure, testée)
# La détection de fenêtre vit dans src/ : elle sert aussi au prototype d'extraction et, plus tard, au pipeline.
from src.signal_processing.transit_fit import transit_window  # noqa: E402,F401


def dispersion(ratios) -> float:
    """Écart-type du log(rapport estimé/vrai) : 0 = rapport identique pour toutes les planètes."""
    r = np.asarray([x for x in ratios if x is not None and np.isfinite(x) and x > 0])
    return float(np.std(np.log(r))) if len(r) > 1 else float("nan")


# --------------------------------------------------------------------------- acquisition (réseau)

def choose_ids(by_planet_csv: Path, n_worst: int, n_typical: int) -> list[tuple[str, str]]:
    d = pd.read_csv(by_planet_csv).sort_values("rmse", ascending=False).reset_index(drop=True)
    mid = len(d) // 2
    typ = d.iloc[mid - n_typical // 2: mid - n_typical // 2 + n_typical]
    return [(str(int(p)), "pire") for p in d.head(n_worst)["planet_id"]] + \
           [(str(int(p)), "typique") for p in typ["planet_id"]]


def acquire(ids: list[str], binning_airs: int, binning_fgs1: int, reuse: bool) -> None:
    """Télécharge, traite et sauvegarde les courbes des planètes absentes de CURVES, puis supprime le brut."""
    CURVES.mkdir(parents=True, exist_ok=True)
    todo = [p for p in ids if not (CURVES / f"{p}.npz").exists()]
    if not todo or reuse:
        return
    import shutil

    from src.data import collect
    from src.signal_processing.io import wavelength_index
    from src.signal_processing.pipeline import process_one_planet

    api = collect.get_api()
    collect.download_meta(api)
    wl = wavelength_index(RAW / "wavelengths.csv", RAW / "axis_info.parquet")
    cut_inf, cut_sup = int(wl[0]), int(wl[-1]) + 1
    adc = pd.read_csv(RAW / "adc_info.csv")
    gain, offset = float(adc["AIRS-CH0_adc_gain"].iloc[0]), float(adc["AIRS-CH0_adc_offset"].iloc[0])
    axis = pd.read_parquet(RAW / "axis_info.parquet")
    for pid in todo:
        try:
            if not collect.download_planet(api, pid, "train"):
                print(f"[{pid}] téléchargement incomplet, ignorée")
                continue
            cubes: dict = {}
            row = process_one_planet(pid, RAW, "train", axis, gain, offset, cut_inf, cut_sup,
                                     binning_airs=binning_airs, binning_fgs1=binning_fgs1, _cubes_out=cubes)
            if row is None:
                print(f"[{pid}] échec du traitement, ignorée")
                continue
            np.savez_compressed(CURVES / f"{pid}.npz",
                                airs=cubes["airs"].sum(axis=1).astype(np.float32),     # (n_time, n_wl)
                                fgs=cubes["fgs1"].sum(axis=(1, 2)).astype(np.float32),  # (n_time,)
                                rp=np.array([row[f"Rp_{i}"] for i in range(1, 284)]))
            print(f"[{pid}] courbes sauvegardées")
        finally:
            shutil.rmtree(RAW / "train" / pid, ignore_errors=True)


def plot(path: Path, entries: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 4
    rows = int(np.ceil(len(entries) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), squeeze=False)
    for ax, e in zip(axs.ravel(), entries):
        f, st = e["curve"], e["stats"]
        n = len(f)
        ax.plot(f / np.median(np.r_[f[: n // 10], f[-n // 10:]]), lw=.6, color="k")
        mid, win = n // 2, max(n // 10, 1)
        ax.axvspan(mid - win, mid + win, color="C1", alpha=.25, label="fenêtre fixe (±10 %)")
        if st["detected"]:
            ax.axvline(st["s50"], color="C2"); ax.axvline(st["e50"], color="C2", label="transit détecté (mi-profondeur)")
        ax.set_title(f"{e['planet_id']} ({e['role']})\n{st.get('verdict_fixed', 'transit non détecté')}", fontsize=8)
        ax.tick_params(labelsize=7)
    axs.ravel()[0].legend(fontsize=6, loc="lower left")
    for ax in axs.ravel()[len(entries):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--worst", type=int, default=3)
    ap.add_argument("--typical", type=int, default=5)
    ap.add_argument("--by-planet", default="reports/diagnostic/by_planet.csv")
    ap.add_argument("--reuse", action="store_true", help="n'utilise que les courbes déjà sauvegardées")
    ap.add_argument("--binning-airs", type=int, default=5)
    ap.add_argument("--binning-fgs1", type=int, default=20)
    ap.add_argument("--out", default="reports/transit_window")
    a = ap.parse_args()

    chosen = [(str(i), "choisie") for i in a.ids] if a.ids else choose_ids(Path(a.by_planet), a.worst, a.typical)
    print("planètes :", ", ".join(f"{p} ({r})" for p, r in chosen))
    acquire([p for p, _ in chosen], a.binning_airs, a.binning_fgs1, a.reuse)

    labels = pd.read_csv(RAW / "train.csv").set_index("planet_id")
    rows, entries = [], []
    for pid, role in chosen:
        fp = CURVES / f"{pid}.npz"
        if not fp.exists():
            print(f"[{pid}] pas de courbes, ignorée")
            continue
        z = np.load(fp)
        st = transit_window(z["airs"].astype(float).sum(axis=1))
        true = float(labels.loc[int(pid), [f"wl_{i}" for i in range(2, 284)]].astype(float).mean())
        r = lambda v: float(v) / true if v is not None and np.isfinite(v) else np.nan
        rows.append({"planet_id": pid, "role": role, "duree_10pct": st.get("dur10"), "duree_50pct": st.get("dur50"),
                     "decalage_centre": st.get("center_offset"), "fenetre_fixe": st["window_frac"],
                     "verdict": st.get("verdict_fixed", "non détecté"), "vraie_profondeur": true,
                     "pipeline/vrai": r(np.mean(z["rp"][1:] ** 2)), "fixe_blanche/vrai": r(st["depth_fixed"]),
                     "adaptee/vrai": r(st.get("depth_adapt", np.nan))})
        entries.append({"planet_id": pid, "role": role, "curve": z["airs"].astype(float).sum(axis=1), "stats": st})

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 220, "display.float_format", lambda v: f"{v:.4g}")
    print("\n" + table.to_string(index=False))
    d = {c: dispersion(table[c]) for c in ("pipeline/vrai", "fixe_blanche/vrai", "adaptee/vrai")}
    print("\nDispersion de log(estimé/vrai) entre planètes (plus petit = meilleur) :")
    for c, v in d.items():
        print(f"  {c:20s} {v:.3f}")
    if np.isfinite(d["adaptee/vrai"]) and np.isfinite(d["fixe_blanche/vrai"]):
        verdict = "SOUTENUE" if d["adaptee/vrai"] < 0.5 * d["fixe_blanche/vrai"] else "NON démontrée"
        print(f"Hypothèse « la fenêtre fixe cause l'erreur de niveau » : {verdict} (critère : dispersion divisée par 2).")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "window_check.csv", index=False)
    plot(out / "window_check.png", entries)
    print(f"-> {out}/")


if __name__ == "__main__":
    main()
