"""Tests des briques du diagnostic : la formule locale du score doit égaler la métrique officielle."""
import numpy as np

from scripts.diagnose_errors import (N_WL, cell_scores, channel_groups, extraction_table, group_table,
                                     pick_transform, planet_table)
from src.ml.features import compute_naive_stats
from src.ml.metrics import evaluate_on_validation


def synth(n=12, seed=0):
    rng = np.random.default_rng(seed)
    y = 0.05 + 0.005 * rng.normal(size=(n, N_WL))
    mu = np.abs(y + 2e-3 * rng.normal(size=(n, N_WL)))     # la métrique refuse les valeurs négatives
    sigma = np.full((n, N_WL), 2e-3)
    return y, mu, sigma


def test_groupes_partitionnent_les_283_canaux():
    g = channel_groups(10)
    idx = np.concatenate(list(g.values()))
    assert sorted(idx) == list(range(N_WL))


def test_score_cellule_egale_metrique_officielle():
    y, mu, sigma = synth()
    nm, ns = compute_naive_stats(y)
    S = cell_scores(y, mu, sigma, nm, ns)
    official = evaluate_on_validation(y, mu, sigma, np.arange(len(y)), nm, ns)
    assert abs(float(np.clip(S.mean(), 0, 1)) - official) < 1e-9


def test_decomposition_niveau_forme():
    y, mu, sigma = synth()
    E = y - mu
    E[0] += 0.01                                   # planète 0 : décalage de niveau pur
    t = planet_table(np.arange(len(y)), E, np.zeros_like(E))
    assert t.iloc[0]["planet_id"] == 0 and t.iloc[0]["part_niveau"] > 0.8


def test_parts_du_tableau_par_groupe_somment_a_un():
    y, mu, sigma = synth()
    nm, ns = compute_naive_stats(y)
    S = cell_scores(y, mu, sigma, nm, ns)
    g = group_table(channel_groups(10), y - mu, y, y.mean(axis=0), y, S)
    assert abs(g["part_erreur_totale"].sum() - 1) < 1e-9


def test_extraction_detecte_une_rampe_et_l_unite():
    y, _, _ = synth(40)
    ramp = np.zeros(N_WL); ramp[1:11] = 0.01        # biais constant sur les 10 premiers canaux AIRS
    Rp = y + ramp                                    # même unité que wl
    unit, Rp_t = pick_transform(Rp, y)
    assert unit == "Rp"
    unit2, _ = pick_transform(np.sqrt(y), y)         # Rp = racine de la profondeur
    assert unit2 == "Rp²"
    table, per_channel = extraction_table(channel_groups(10), Rp_t, y)
    assert table.set_index("groupe").loc["AIRS bleu (1-10)", "biais_rms"] > 0.009
    assert table.set_index("groupe").loc["AIRS reste", "biais_rms"] < 1e-9
