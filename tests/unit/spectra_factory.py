"""Données synthétiques pour les tests d'intégration v0.2.0 : courbes npz, labels, astro, découpage."""
import numpy as np
import pandas as pd

N_WL_AIRS = 282


def synth_curve(seed, n=600, depth=0.01, dur=0.35, center=0.5, drift=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / n
    shape = np.clip((dur / 2 - np.abs(t - center)) / 0.04, 0, 1)
    d_j = depth * (1 + 0.05 * np.sin(np.linspace(0, 3, N_WL_AIRS)))
    base = 1 + drift * (t - 0.5)
    F = 1e6 * base[:, None] * (1 - d_j[None, :] * shape[:, None]) * (1 + 3e-5 * rng.normal(size=(n, N_WL_AIRS)))
    fgs = 1e6 * base * (1 - depth * shape) * (1 + 3e-5 * rng.normal(size=n))
    return F, fgs, np.r_[depth, d_j]


def make_world(tmp_path, n_planets=30, n_val=8, seed=0, skip_curves=()):
    """Écrit tmp/raw/{train.csv,train_star_info.csv}, tmp/curves/*.npz ; retourne (split, curves_dir, raw_dir)."""
    rng = np.random.default_rng(seed)
    raw, curves = tmp_path / "raw", tmp_path / "curves"
    raw.mkdir(); curves.mkdir()
    ids = [1000 + i for i in range(n_planets)]
    labels, stars = [], []
    for k, pid in enumerate(ids):
        depth = rng.uniform(0.004, 0.02)
        F, fgs, d = synth_curve(seed * 1000 + k, depth=depth, dur=rng.uniform(0.25, 0.45),
                                center=rng.uniform(0.4, 0.6), drift=rng.uniform(0, 0.006))
        if pid not in skip_curves:
            np.savez(curves / f"{pid}.npz", airs=F.astype(np.float32), fgs=fgs.astype(np.float32), rp=np.sqrt(d))
        labels.append({"planet_id": pid, **{f"wl_{j + 1}": float(d[j]) / 1.05 for j in range(283)}})
        stars.append({"planet_id": pid, "Rs": 1.0, "Ts": 5700 + 100 * rng.normal(), "Ms": 1 + 0.1 * rng.normal(),
                      "Mp": rng.uniform(0.5, 2), "P": rng.uniform(1, 5), "sma": rng.uniform(0.02, 0.06),
                      "i": 89 + 0.3 * rng.normal()})
    pd.DataFrame(labels).to_csv(raw / "train.csv", index=False)
    pd.DataFrame(stars).to_csv(raw / "train_star_info.csv", index=False)
    split = {"val": ids[:n_val], "train": ids[n_val:], "batch_size": 10}
    return split, curves, raw
