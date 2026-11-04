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

def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    pad = np.pad(x, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(pad, np.ones(w) / w, mode="valid")


def _run(s: np.ndarray, level: float, i0: int) -> tuple[int, int]:
    """Plus grand intervalle contigu autour de i0 où s < level."""
    a = b = i0
    while a > 0 and s[a - 1] < level:
        a -= 1
    while b < len(s) - 1 and s[b + 1] < level:
        b += 1
    return a, b


def transit_window(flux: np.ndarray, edge: float = 0.10) -> dict:
    """Mesure le transit sur une courbe de lumière blanche et compare fenêtre fixe / fenêtre adaptée.

    flux : (n_time,) flux total. Les `edge` (10 %) premières et dernières frames servent de référence
    hors transit.
    """
    flux = np.asarray(flux, dtype=float)
    n = len(flux)
    k = max(int(edge * n), 5)
    base = float(np.median(np.r_[flux[:k], flux[-k:]]))
    f = flux / base
    w = max(5, n // 100)
    s = _smooth(f, w)
    noise = 1.4826 * float(np.median(np.abs(np.r_[f[:k], f[-k:]] - 1.0)))
    c = max(int(0.01 * n), 2)

    # --- méthode actuelle (calcul_Rp_median) appliquée à la courbe blanche
    mid, win = n // 2, max(n // 10, 1)
    depth_fixed = 1.0 - np.median(f[mid - win: mid + win]) / np.median(np.r_[f[: mid - win], f[mid + win:]])

    i0 = int(np.argmin(s))
    lo, hi = max(i0 - c, 0), min(i0 + c + 1, n)
    depth_bottom = 1.0 - float(np.median(s[lo:hi]))
    out = {"n": n, "noise": noise, "depth_bottom": depth_bottom, "depth_fixed": float(depth_fixed),
           "window_frac": 2 * win / n, "detected": False}
    if depth_bottom < 5 * noise / np.sqrt(w) + 1e-12:
        return out                                            # pas de transit détectable

    s50, e50 = _run(s, 1.0 - 0.5 * depth_bottom, i0)
    has10 = 0.1 * depth_bottom > 3 * noise / np.sqrt(w)
    s10, e10 = _run(s, 1.0 - 0.1 * depth_bottom, i0) if has10 else (s50, e50)
    dur50, dur10 = (e50 - s50 + 1) / n, (e10 - s10 + 1) / n

    # fenêtre adaptée : fond du transit (sans les bords) contre le hors-transit élargi d'une marge
    m_in = max(int(0.15 * (e50 - s50)), 1)
    m_out = max(int(0.02 * n), 2) + (0 if has10 else int(0.5 * (e50 - s50)))
    a_in, b_in = min(s50 + m_in, e50), max(e50 - m_in, s50)
    outside = np.r_[f[: max(s10 - m_out, 0)], f[min(e10 + m_out + 1, n):]]
    depth_adapt = 1.0 - np.median(f[a_in: b_in + 1]) / np.median(outside) if len(outside) else np.nan

    # la médiane résiste tant que moins de ~40 % des frames « hors transit » sont en transit (F_out)
    # et que plus de ~60 % des frames de la fenêtre sont en transit (F_in)
    wf = out["window_frac"]
    contamination_out = max(dur10 - wf, 0.0) / (1.0 - wf)
    verdict = "F_out contaminé : transit trop long" if contamination_out > 0.4 else \
              "F_in dilué : transit trop court" if dur50 < 0.6 * wf else "fenêtre acceptable"
    out.update(detected=True, dur50=dur50, dur10=dur10, center_offset=((s50 + e50) / 2 - n / 2) / n,
               s50=s50, e50=e50, s10=s10, e10=e10, depth_adapt=float(depth_adapt), verdict_fixed=verdict)
    return out


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
