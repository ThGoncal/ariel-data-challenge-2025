"""
src/signal_processing
=====================
Package de traitement du signal pour le challenge Ariel Data Challenge 2025.

Architecture des modules
------------------------
io.py           : Chargement des données brutes (parquet) et des calibrations
calibration.py  : Corrections instrumentales (ADC, dark, flat, hot pixels, non-linéarité)
transit.py      : Extraction du spectre de transit (Rp/Rs)
pipeline.py     : Orchestration planète par planète + parallélisation
visualization.py: Affichage des spectres (Plotly)

Usage rapide
------------
    from src.signal_processing.pipeline import build_spectra_dataframe
    from src.signal_processing.io import wavelength_index
"""

from src.signal_processing.pipeline import build_spectra_dataframe, process_one_planet
from src.signal_processing.io import wavelength_index, load_signal, load_calib

__all__ = [
    "build_spectra_dataframe",
    "process_one_planet",
    "wavelength_index",
    "load_signal",
    "load_calib",
]
