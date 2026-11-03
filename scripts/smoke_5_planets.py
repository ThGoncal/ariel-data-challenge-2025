"""
smoke_5_planets.py — valide la chaîne sur quelques vraies planètes.

    python scripts/smoke_5_planets.py --n 5 [--keep]

1. télécharge les 7 fichiers racine si absents (data/raw/)
2. télécharge n planètes (train) via l'API Kaggle
3. traite chaque planète (process_one_planet)
4. compare (Rp/Rs)^2 aux labels de train.csv : corrélation, ratio médian
5. supprime les fichiers bruts (sauf --keep)
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import collect                                    # noqa: E402
from src.signal_processing.io import wavelength_index           # noqa: E402
from src.signal_processing.pipeline import process_one_planet   # noqa: E402

RAW = Path("data/raw")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()

    api = collect.get_api()
    collect.download_meta(api)

    ids = collect.get_planet_ids("train")[: a.n]
    wl_idx = wavelength_index(RAW / "wavelengths.csv", RAW / "axis_info.parquet")
    cut_inf, cut_sup = int(wl_idx[0]), int(wl_idx[-1]) + 1
    assert cut_sup - cut_inf == len(wl_idx), "indices spectraux non contigus"
    print(f"coupe spectrale AIRS: [{cut_inf}:{cut_sup}] -> {len(wl_idx)} longueurs d'onde")

    adc = pd.read_csv(RAW / "adc_info.csv")
    print(adc.head())
    gain, offset = float(adc["AIRS-CH0_adc_gain"].iloc[0]), float(adc["AIRS-CH0_adc_offset"].iloc[0])
    axis = pd.read_parquet(RAW / "axis_info.parquet")
    train = pd.read_csv(RAW / "train.csv").set_index("planet_id")
    wl_cols = [c for c in train.columns if c.startswith("wl_")]

    rows = []
    for pid in ids:
        if not collect.download_planet(api, pid, "train"):
            print(f"[{pid}] téléchargement incomplet"); continue
        row = process_one_planet(pid, RAW, "train", axis, gain, offset, cut_inf, cut_sup)
        if row is None:
            print(f"[{pid}] échec du traitement"); continue
        rp = np.array([row[f"Rp_{i}"] for i in range(1, 284)])
        depth = rp ** 2
        y = train.loc[int(pid), wl_cols].to_numpy(dtype=float)
        m = np.isfinite(depth) & np.isfinite(y)
        corr = np.corrcoef(depth[m], y[m])[0, 1]
        # erreur relative du niveau moyen, et RMSE vs prédicteur "constante = moyenne vraie"
        lvl = depth[m].mean() / y[m].mean()
        rmse = np.sqrt(np.mean((depth[m] - y[m]) ** 2))
        rmse_const = np.sqrt(np.mean((y[m].mean() - y[m]) ** 2))
        print(f"[{pid}] n={m.sum()} corr={corr:.3f} ratio médian={np.median(depth[m] / y[m]):.3f} "
              f"niveau moyen pred/vrai={lvl:.3f} rmse={rmse:.2e} rmse(constante vraie)={rmse_const:.2e}")
        rows.append(row)
        if not a.keep:
            shutil.rmtree(RAW / "train" / pid, ignore_errors=True)
    if rows:   # on garde les spectres calculés pour ne pas retélécharger
        out = Path("data/processed"); out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out / "smoke_spectra.csv", index=False)
        print(f"spectres sauvegardés dans {out / 'smoke_spectra.csv'}")


if __name__ == "__main__":
    main()
