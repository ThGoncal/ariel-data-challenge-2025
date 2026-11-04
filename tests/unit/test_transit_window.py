"""Tests de l'analyse de fenêtre de transit sur des courbes de lumière synthétiques (trapèzes)."""
import numpy as np

from scripts.check_transit_window import dispersion, transit_window


def trapezoid(n=1100, depth=0.01, dur=0.30, ingress=0.03, center=0.5, noise=2e-5, seed=0):
    """Flux = 1 − depth·trapèze ; dur et ingress sont des fractions de la série."""
    t = np.arange(n) / n
    half = dur / 2
    shape = np.clip((half - np.abs(t - center)) / ingress, 0, 1)
    return 1.0 - depth * shape + np.random.default_rng(seed).normal(scale=noise, size=n)


def test_mesure_duree_et_centre():
    st = transit_window(trapezoid(dur=0.30, ingress=0.03, center=0.55))
    assert st["detected"]
    assert abs(st["dur50"] - 0.27) < 0.02                 # largeur à mi-profondeur = base − ingress
    assert abs(st["center_offset"] - 0.05) < 0.02


def test_transit_tres_long_fausse_la_fenetre_fixe():
    # 75 % de la série : plus de 40 % des frames « hors transit » sont en transit, la médiane bascule
    st = transit_window(trapezoid(dur=0.75, ingress=0.02, depth=0.01))
    assert abs(st["depth_adapt"] - 0.01) < 3e-4
    assert st["depth_fixed"] < 0.006                        # fortement sous-estimée
    assert st["verdict_fixed"] == "F_out contaminé : transit trop long"


def test_la_mediane_resiste_pour_un_transit_moyen_ou_long():
    # 50 % de la série : la fenêtre fixe reste correcte (la médiane est robuste)
    st = transit_window(trapezoid(dur=0.50, ingress=0.03, depth=0.01))
    assert abs(st["depth_fixed"] - 0.01) < 4e-4
    assert st["verdict_fixed"] == "fenêtre acceptable"


def test_fenetre_adaptee_retrouve_la_profondeur_quand_le_transit_est_court():
    st = transit_window(trapezoid(dur=0.08, ingress=0.01, depth=0.01))
    assert abs(st["depth_adapt"] - 0.01) < 5e-4
    assert st["depth_fixed"] < 0.0095                       # du hors-transit dans F_in
    assert st["verdict_fixed"] == "F_in dilué : transit trop court"


def test_fenetre_fixe_correcte_quand_la_duree_correspond():
    st = transit_window(trapezoid(dur=0.20, ingress=0.02, depth=0.01))
    assert abs(st["depth_fixed"] - 0.01) < 3e-4 and abs(st["depth_adapt"] - 0.01) < 3e-4
    assert st["verdict_fixed"] == "fenêtre acceptable"


def test_pas_de_transit_detecte():
    st = transit_window(1.0 + np.random.default_rng(1).normal(scale=1e-4, size=1000))
    assert st["detected"] is False


def test_dispersion():
    assert dispersion([1.2, 1.2, 1.2]) < 1e-12
    assert dispersion([0.5, 1.0, 2.0]) > 0.5
