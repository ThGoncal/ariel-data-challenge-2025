"""
spectra.py — Spectre de profondeur d'une planète : UNE seule définition, pour l'entraînement ET l'inférence
=========================================================================================================
L'entraînement lit des courbes sauvegardées (data/processed/transit_curves/*.npz) ; l'inférence sur le jeu de
test calcule les cubes à la volée (process_one_planet(..., _cubes_out=...)). Les deux passent par
`extract_spectrum` : pas de décalage entre ce qui est appris et ce qui est servi.

Paramètres (params.yaml, section `extraction`) :
    method         fixe | adaptee | adaptee_detrend | gabarit   (voir transit_fit.py)
    deg            degré de la ligne de base (méthodes *_detrend et gabarit)
    edge_channels  nombre de premiers canaux AIRS remplacés par la moyenne des `n_ref` canaux suivants
                   (leur flux est très faible : la profondeur y est dominée par le bruit)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.signal_processing.transit_fit import METHODS, extract_depths

N_WL = 283
RP_COLS = [f"Rp_{i}" for i in range(1, N_WL + 1)]


@dataclass(frozen=True)
class ExtractionParams:
    method: str = "adaptee"
    deg: int = 1
    edge_channels: int = 10

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"extraction.method inconnue : {self.method!r} (attendu : {METHODS})")
        if self.deg < 0 or self.edge_channels < 0:
            raise ValueError("extraction.deg et extraction.edge_channels doivent être ≥ 0")

    @classmethod
    def from_params(cls, params: dict | None) -> "ExtractionParams":
        section = (params or {}).get("extraction", {}) or {}
        return cls(**{k: section[k] for k in ("method", "deg", "edge_channels") if k in section})

    def as_dict(self) -> dict:
        return {"method": self.method, "deg": self.deg, "edge_channels": self.edge_channels}


def replace_edge_channels(airs_depth: np.ndarray, n_edge: int, n_ref: int = 10) -> np.ndarray:
    """Remplace les n_edge premiers canaux par la moyenne des n_ref canaux suivants (copie)."""
    out = np.array(airs_depth, dtype=float, copy=True)
    if n_edge <= 0:
        return out
    if n_edge + n_ref > len(out):
        raise ValueError(f"trop de canaux de bord : {n_edge} + {n_ref} > {len(out)}")
    out[:n_edge] = float(np.mean(out[n_edge: n_edge + n_ref]))
    return out


def extract_spectrum(airs: np.ndarray, fgs: np.ndarray, p: ExtractionParams | None = None) -> np.ndarray:
    """Profondeurs [FGS1, AIRS_1..282] (283,). airs : (n_time, 282), fgs : (n_time,) — flux sommés sur l'espace."""
    p = p or ExtractionParams()
    d = extract_depths(np.asarray(airs, dtype=float), np.asarray(fgs, dtype=float), p.method, p.deg)
    d[1:] = replace_edge_channels(d[1:], p.edge_channels)
    return d


def depth_to_rp(depth: np.ndarray) -> np.ndarray:
    """Rp = sqrt(max(profondeur, 0)) : convention des colonnes Rp_* des spectres."""
    return np.sqrt(np.maximum(np.asarray(depth, dtype=float), 0.0))


def curves_from_cubes(cubes: dict) -> tuple[np.ndarray, np.ndarray]:
    """Cubes de process_one_planet(_cubes_out=...) -> courbes sommées sur l'espace (même format que les npz)."""
    return cubes["airs"].sum(axis=1), cubes["fgs1"].sum(axis=(1, 2))


def spectrum_from_cubes(cubes: dict, p: ExtractionParams | None = None) -> np.ndarray:
    airs, fgs = curves_from_cubes(cubes)
    return extract_spectrum(airs, fgs, p)
