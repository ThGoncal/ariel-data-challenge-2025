"""Test bout-en-bout de process_one_planet sur parquets synthétiques (structure réelle)."""
import numpy as np
import pandas as pd
import pytest

from src.signal_processing.pipeline import process_one_planet

N_T_AIRS, N_T_FGS = 400, 800
N_AIRS_PIX, N_FGS_PIX = 32 * 356, 32 * 32


def _write_calib(d, n_pix, rng):
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(np.zeros((1, n_pix)).reshape(32, -1)).to_parquet(d / "dark.parquet")
    pd.DataFrame(np.ones((1, n_pix)).reshape(32, -1)).to_parquet(d / "flat.parquet")
    pd.DataFrame(np.zeros((1, n_pix), dtype=bool).reshape(32, -1)).to_parquet(d / "dead.parquet")
    pd.DataFrame(np.zeros((1, n_pix)).reshape(32, -1)).to_parquet(d / "read.parquet")
    lc = np.zeros((6, n_pix)); lc[1] = 1.0   # polynôme identité (ordre croissant)
    pd.DataFrame(lc.reshape(6 * 32, -1)).to_parquet(d / "linear_corr.parquet")


def _signal(n_t, n_pix, rng, depth=0.01):
    base = 1000.0 + rng.normal(0, 0.5, size=(n_t, n_pix))
    t = np.arange(n_t)
    in_transit = (t > 0.4 * n_t) & (t < 0.6 * n_t)
    base[in_transit] *= (1 - depth)
    return base


@pytest.mark.slow
def test_process_one_planet_synthetique(tmp_path):
    rng = np.random.default_rng(0)
    pdir = tmp_path / "train" / "1"
    pdir.mkdir(parents=True)
    pd.DataFrame(_signal(N_T_AIRS, N_AIRS_PIX, rng)).astype("float64").to_parquet(pdir / "AIRS-CH0_signal_0.parquet")
    pd.DataFrame(_signal(N_T_FGS, N_FGS_PIX, rng)).astype("float64").to_parquet(pdir / "FGS1_signal_0.parquet")
    _write_calib(pdir / "AIRS-CH0_calibration_0", N_AIRS_PIX, rng)
    _write_calib(pdir / "FGS1_calibration_0", N_FGS_PIX, rng)
    axis = pd.DataFrame({"AIRS-CH0-integration_time": np.full(N_T_AIRS, 0.1)})

    row = process_one_planet(1, tmp_path, "train", axis, gain=1.0, offset=0.0,
                             cut_inf=39, cut_sup=321, binning_airs=5, binning_fgs1=10)
    assert row is not None, "process_one_planet a renvoyé None (voir logs)"
    rp_cols = [k for k in row if k.startswith("Rp_")]
    assert len(rp_cols) == 283
    assert set(rp_cols) == {f"Rp_{i}" for i in range(1, 284)}
