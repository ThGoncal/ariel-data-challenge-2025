"""
src/models/train.py — Entraînement unifié — Ariel Data Challenge 2025
======================================================================
Remplace le stub du template par l'entraînement complet.
Supporte deux modes :
    - "baseline" : MultiTaskLassoCV + bootstrap (rapide, CPU)
    - "dl"       : MLP hétéroscédastique PyTorch (plus précis, GPU optionnel)

Interface compatible avec le template :
    train_model(X_train, y_train, model, params, experiment_name, run_name)

Interface Ariel étendue :
    train_ariel(df_train, df_test, params, mode, run_name)
"""

import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


# =============================================================================
# INTERFACE TEMPLATE (conservée)
# =============================================================================

def train_model(
    X_train,
    y_train,
    model,
    params: dict[str, Any],
    experiment_name: str = "ariel-2025",
    run_name: str | None = None,
) -> str:
    """
    Entraîne un modèle sklearn et loggue dans MLflow.
    Interface générique du template — conservée et étendue.

    Returns
    -------
    str : run_id MLflow
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"MLflow run démarré : {run.info.run_id}")

        mlflow.log_params(params)
        model.set_params(**{k: v for k, v in params.items() if hasattr(model, k)})
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")

        logger.info(f"Modèle loggué | run_id : {run.info.run_id}")
        return run.info.run_id


# =============================================================================
# BOOTSTRAP (partagé baseline et pipeline incrémental)
# =============================================================================

def train_bootstrap(
    X_train_scaled: np.ndarray,
    y_train_scaled: np.ndarray,
    X_val_scaled:   np.ndarray,
    n_bootstrap:    int,
    alphas_grid:    np.ndarray,
    cv:             int,
    max_iter:       int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Entraîne N modèles MultiTaskLassoCV sur des rééchantillonnages bootstrap.

    Chaque modèle est entraîné sur un échantillon avec remise du train set.
    La variabilité des prédictions entre les N modèles estime l'incertitude σ.

    Returns
    -------
    preds_mean_scaled : (n_val, 283) — moyenne des prédictions
    preds_std_scaled  : (n_val, 283) — écart-type des prédictions (= σ estimé)
    """
    from sklearn.linear_model import MultiTaskLassoCV
    from tqdm import tqdm

    all_preds = []

    for i in tqdm(range(n_bootstrap), desc="Bootstrap"):
        X_boot, y_boot = resample(X_train_scaled, y_train_scaled, random_state=i)

        model = MultiTaskLassoCV(
            alphas=alphas_grid, cv=cv, max_iter=max_iter,
            random_state=i, n_jobs=-1,
        )
        model.fit(X_boot, y_boot)
        all_preds.append(model.predict(X_val_scaled))

    preds_array = np.stack(all_preds, axis=0)          # (n_bootstrap, n_val, 283)
    return preds_array.mean(axis=0), preds_array.std(axis=0)


# =============================================================================
# ENTRAÎNEMENT BASELINE COMPLET
# =============================================================================

def train_ariel_baseline(
    df_train:        pd.DataFrame,
    df_test:         pd.DataFrame,
    params:          dict,
    run_name:        str | None = None,
    n_planets:       int        = None,
) -> tuple[float, str]:
    """
    Entraîne le modèle baseline MultiTaskLassoCV + bootstrap sur les données Ariel.

    Parameters
    ----------
    df_train   : DataFrame train_with_spectra.csv (données accumulées)
    df_test    : DataFrame test_with_spectra.csv
    params     : dictionnaire params.yaml complet
    run_name   : nom du run MLflow (ex: "batch_003_150planets")
    n_planets  : nombre de planètes utilisées (pour le log MLflow)

    Returns
    -------
    val_score : float, score GLL sur la validation locale
    run_id    : str, identifiant du run MLflow
    """
    from src.features.build_features import prepare_train_data, prepare_test_data
    from src.models.evaluate import evaluate_predictions
    from src.ml.metrics import build_submission_dataframe

    bl  = params["baseline"]
    pre = params["preprocess"]

    # --- Features
    X_all, y_all, _, _  = prepare_train_data(df_train, drop_eccentricity=pre["drop_eccentricity"])
    X_kaggle, _, pids_kaggle = prepare_test_data(df_test, drop_eccentricity=pre["drop_eccentricity"])

    # --- Split
    X_train, X_val, y_train, y_val, idx_tr, idx_val = train_test_split(
        X_all, y_all, np.arange(len(df_train)),
        test_size=pre["test_size"], random_state=pre["random_state"],
    )
    pids_val = df_train["planet_id"].values[idx_val]

    # --- Standardisation (fit sur X_train uniquement)
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    X_tr_sc  = scaler_X.transform(X_train)
    X_val_sc = scaler_X.transform(X_val)
    X_kg_sc  = scaler_X.transform(X_kaggle)
    y_tr_sc  = scaler_y.transform(y_train)

    # --- Entraînement bootstrap
    alphas = np.logspace(bl["alphas_min"], bl["alphas_max"], bl["n_alphas"])

    preds_mean_sc, preds_std_sc = train_bootstrap(
        X_train_scaled = X_tr_sc,
        y_train_scaled = y_tr_sc,
        X_val_scaled   = X_val_sc,
        n_bootstrap    = bl["n_bootstrap"],
        alphas_grid    = alphas,
        cv             = bl["cv"],
        max_iter       = bl["max_iter"],
    )

    # --- Inversion scaler
    y_val_mean = scaler_y.inverse_transform(preds_mean_sc)
    y_val_std  = np.clip(preds_std_sc * scaler_y.scale_, 1e-9, None)

    # --- Évaluation GLL
    from src.ml.features import compute_naive_stats
    naive_mean, naive_sigma = compute_naive_stats(y_train)

    val_score = evaluate_predictions(
        y_true=y_val, y_pred_mean=y_val_mean, y_pred_std=y_val_std,
        planet_ids=pids_val, naive_mean=naive_mean, naive_sigma=naive_sigma,
    )

    # --- Log MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(params.get("mlflow", {}).get("experiment_name", "ariel-2025"))

    with mlflow.start_run(run_name=run_name or f"baseline_{n_planets or len(df_train)}planets") as run:
        mlflow.log_params({
            "model":       "MultiTaskLassoCV_bootstrap",
            "n_bootstrap": bl["n_bootstrap"],
            "cv":          bl["cv"],
            "n_planets":   n_planets or len(df_train),
            "n_train":     len(X_train),
            "n_val":       len(X_val),
        })
        mlflow.log_metrics({
            "val_gll_score": val_score,
            "n_planets":     n_planets or len(df_train),
            "naive_mean":    naive_mean,
            "naive_sigma":   naive_sigma,
        })

        # Prédictions Kaggle
        kg_mean_sc, kg_std_sc = train_bootstrap(
            X_train_scaled = scaler_X.transform(X_all),
            y_train_scaled = scaler_y.transform(y_all),
            X_val_scaled   = X_kg_sc,
            n_bootstrap    = bl["n_bootstrap"],
            alphas_grid    = alphas,
            cv             = bl["cv"],
            max_iter       = bl["max_iter"],
        )
        kg_mean = scaler_y.inverse_transform(kg_mean_sc)
        kg_std  = np.clip(kg_std_sc * scaler_y.scale_, 1e-9, None)

        df_sub  = build_submission_dataframe(pids_kaggle, kg_mean, kg_std)
        sub_path = Path("models/submission_baseline.csv")
        sub_path.parent.mkdir(exist_ok=True)
        df_sub.to_csv(sub_path, index=False)

        mlflow.log_artifact(str(sub_path))
        run_id = run.info.run_id

    logger.info(f"✅ Baseline | score={val_score:.6f} | run_id={run_id}")
    return val_score, run_id


# =============================================================================
# ENTRAÎNEMENT DL COMPLET
# =============================================================================

def train_ariel_dl(
    df_train:  pd.DataFrame,
    df_test:   pd.DataFrame,
    params:    dict,
    run_name:  str | None = None,
    n_planets: int        = None,
) -> tuple[float, str]:
    """
    Entraîne le MLP hétéroscédastique avec la loss GLL.

    Returns
    -------
    best_gll : float, meilleur score GLL sur la validation
    run_id   : str, identifiant du run MLflow
    """
    import torch
    from torch.utils.data import DataLoader

    from src.features.build_features import prepare_train_data, prepare_test_data
    from src.models.model import ArielHeteroscedasticMLP, GaussianNLLLoss, ArielDataset
    from src.models.evaluate import evaluate_predictions
    from src.ml.features import compute_naive_stats
    from src.ml.metrics import build_submission_dataframe

    m_cfg = params["model"]
    pre   = params["preprocess"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device : {device}")

    # --- Features
    X_all, y_all, _, _ = prepare_train_data(df_train, drop_eccentricity=pre["drop_eccentricity"])
    X_kaggle, _, pids_kaggle = prepare_test_data(df_test, drop_eccentricity=pre["drop_eccentricity"])

    # --- Split
    X_train, X_val, y_train, y_val, _, idx_val = train_test_split(
        X_all, y_all, np.arange(len(df_train)),
        test_size=pre["test_size"], random_state=pre["random_state"],
    )
    pids_val = df_train["planet_id"].values[idx_val]

    # --- Standardisation
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    naive_mean, naive_sigma = compute_naive_stats(y_train)

    X_tr_f32  = scaler_X.transform(X_train).astype(np.float32)
    X_val_f32 = scaler_X.transform(X_val).astype(np.float32)
    X_kg_f32  = scaler_X.transform(X_kaggle).astype(np.float32)
    y_tr_f32  = scaler_y.transform(y_train).astype(np.float32)
    y_val_f32 = scaler_y.transform(y_val).astype(np.float32)

    # --- DataLoaders
    bs = m_cfg["batch_size"]
    train_loader  = DataLoader(ArielDataset(X_tr_f32,  y_tr_f32),  batch_size=bs, shuffle=True)
    val_loader    = DataLoader(ArielDataset(X_val_f32, y_val_f32), batch_size=bs, shuffle=False)
    kaggle_loader = DataLoader(ArielDataset(X_kg_f32),             batch_size=bs, shuffle=False)

    # --- Modèle
    model = ArielHeteroscedasticMLP(
        n_features=X_tr_f32.shape[1], n_wavelengths=283,
        d_model=m_cfg["d_model"], n_layers=m_cfg["n_layers"],
        dropout=m_cfg["dropout"],
    ).to(device)

    loss_fn   = GaussianNLLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=m_cfg["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=m_cfg["n_epochs"], eta_min=m_cfg["learning_rate"] * 1e-2
    )

    # --- Boucle d'entraînement
    best_score   = -np.inf
    no_improve   = 0
    best_weights = None
    best_run_id  = None

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(params.get("mlflow", {}).get("experiment_name", "ariel-2025"))

    with mlflow.start_run(run_name=run_name or f"dl_{n_planets or len(df_train)}planets") as run:
        mlflow.log_params({
            "model":      "HeteroscedasticMLP",
            "d_model":    m_cfg["d_model"],
            "n_layers":   m_cfg["n_layers"],
            "dropout":    m_cfg["dropout"],
            "lr":         m_cfg["learning_rate"],
            "n_planets":  n_planets or len(df_train),
        })

        for epoch in range(1, m_cfg["n_epochs"] + 1):
            # Train
            model.train()
            train_loss = 0.0
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                mu, sigma = model(X_b)
                loss = loss_fn(mu, sigma, y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # Validation
            model.eval()
            all_mu, all_sigma = [], []
            with torch.no_grad():
                for X_b, _ in val_loader:
                    mu, sigma = model(X_b.to(device))
                    all_mu.append(mu.cpu().numpy())
                    all_sigma.append(sigma.cpu().numpy())

            y_pred_mean_sc = np.vstack(all_mu)
            y_pred_std_sc  = np.vstack(all_sigma)

            y_pred_mean = scaler_y.inverse_transform(y_pred_mean_sc)
            y_pred_std  = np.clip(y_pred_std_sc * scaler_y.scale_, 1e-9, None)

            val_score = evaluate_predictions(
                y_true=y_val, y_pred_mean=y_pred_mean, y_pred_std=y_pred_std,
                planet_ids=pids_val, naive_mean=naive_mean, naive_sigma=naive_sigma,
            )

            scheduler.step()

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_gll":    val_score,
            }, step=epoch)

            logger.info(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_gll={val_score:.6f}")

            # Early stopping
            if val_score > best_score:
                best_score   = val_score
                best_weights = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve   = 0
            else:
                no_improve += 1
                if no_improve >= m_cfg["patience"]:
                    logger.info(f"Early stopping à l'époque {epoch}")
                    break

        # Restaurer les meilleurs poids
        if best_weights:
            model.load_state_dict(best_weights)

        mlflow.log_metric("best_val_gll", best_score)

        # Prédictions Kaggle
        model.eval()
        kg_mu, kg_sigma = [], []
        with torch.no_grad():
            for batch in kaggle_loader:
                mu, sigma = model(batch[0].to(device))
                kg_mu.append(mu.cpu().numpy())
                kg_sigma.append(sigma.cpu().numpy())

        kg_mean = scaler_y.inverse_transform(np.vstack(kg_mu))
        kg_std  = np.clip(np.vstack(kg_sigma) * scaler_y.scale_, 1e-9, None)

        df_sub   = build_submission_dataframe(pids_kaggle, kg_mean, kg_std)
        sub_path = Path("models/submission_dl.csv")
        sub_path.parent.mkdir(exist_ok=True)
        df_sub.to_csv(sub_path, index=False)
        mlflow.log_artifact(str(sub_path))

        run_id = run.info.run_id

    logger.info(f"✅ DL | best_score={best_score:.6f} | run_id={run_id}")
    return best_score, run_id
