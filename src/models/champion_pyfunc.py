"""
champion_pyfunc.py — Enveloppe MLflow du modèle champion (importée seulement à l'enregistrement/chargement)
=========================================================================================================
Entrée  : DataFrame avec les colonnes de features du bundle (Rp_1..283 = sqrt(profondeur), Ms, Mp, P, sma, i, Rs, Ts)
Sortie  : DataFrame au format de soumission Kaggle (planet_id, wl_1..wl_283, wl_1_std..wl_283_std)
"""
from __future__ import annotations

import joblib
import mlflow.pyfunc
import numpy as np

from src.ml.metrics import build_submission_dataframe
from src.models.champion import predict_champion


class ChampionPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.bundle = joblib.load(context.artifacts["bundle"])

    def predict(self, context, model_input, params=None):
        pred = predict_champion(self.bundle, model_input)
        ids = model_input["planet_id"].to_numpy() if "planet_id" in model_input else np.arange(len(model_input))
        return build_submission_dataframe(ids, pred.mu, pred.sigma)
