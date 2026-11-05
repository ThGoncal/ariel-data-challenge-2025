"""Tests des mesures de forme du fond de transit."""
import numpy as np

from scripts.diagnose_planet import bottom_stats, sub_depths
from src.signal_processing.transit_fit import transit_window


def curve(bottom, n=1000, dur=0.4, ingress=0.05, depth=0.01):
    t = np.arange(n) / n
    shape = np.clip((dur / 2 - np.abs(t - 0.5)) / ingress, 0, 1)
    x = np.clip((t - 0.5) / (dur / 2), -1, 1)
    return 1 - depth * shape * bottom(x)


def test_fond_plat():
    f = curve(lambda x: 1 + 0 * x)
    s = bottom_stats(f, transit_window(f))
    assert abs(s["courbure"]) < 0.05 and abs(s["asymetrie"]) < 0.05


def test_fond_arrondi_courbure_positive():
    f = curve(lambda x: 1 - 0.1 * x ** 2)
    st = transit_window(f)
    assert bottom_stats(f, st)["courbure"] > 0.03
    d = sub_depths(f, st)
    assert d["centre30"] > d["fond"] * 0.99


def test_fond_incline_asymetrie_non_nulle():
    f = curve(lambda x: 1 + 0.1 * x)
    assert abs(bottom_stats(f, transit_window(f))["asymetrie"]) > 0.03
