"""
train_dl.py — Entraînement Deep Learning Ariel Data Challenge 2025
===================================================================
Boucle d'entraînement complète pour le MLP hétéroscédastique.

Fonctionnalités :
    - Loss NLL gaussienne directement alignée sur la métrique GLL
    - Early stopping sur le score GLL de validation
    - Learning rate scheduler (CosineAnnealingLR)
    - Checkpointing du meilleur modèle
    - Tracking complet MLflow (params, métriques epoch par epoch, artefacts)
    - Inférence finale sur le test set Kaggle

Usage :
    python src/models/train_dl.py
    python src/models/train_dl.py --no-mlflow
    python src/models/train_dl.py --device cuda
"""

# === Standard library
import argparse
import json
import logging
import os
from pathlib import Path

# === Scientific stack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# === PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# === MLflow
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# === Dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# === Modules internes
from src.models.model      import ArielHeteroscedasticMLP, GaussianNLLLoss, ArielDataset
from src.ml.features       import build_feature_matrix, build_target_matrix, compute_naive_stats
from src.ml.metrics        import evaluate_on_validation, build_submission_dataframe

# =============================================================================
# LOGGING
# =============================================================================

os.makedirs("data/processed", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/processed/train_dl.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# CHARGEMENT DES PARAMÈTRES
# =============================================================================

def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# =============================================================================
# BOUCLE D'ENTRAÎNEMENT
# =============================================================================

def train_one_epoch(
    model: ArielHeteroscedasticMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: GaussianNLLLoss,
    device: torch.device,
    scaler_amp: torch.cuda.amp.GradScaler = None,
) -> float:
    """
    Exécute une époque d'entraînement complète.

    Utilise l'AMP (Automatic Mixed Precision) si un GradScaler est fourni —
    réduit la mémoire GPU de ~40% et accélère l'entraînement sur GPU récents.

    Parameters
    ----------
    model      : le modèle PyTorch
    loader     : DataLoader du train set
    optimizer  : optimiseur (Adam ou AdamW)
    loss_fn    : GaussianNLLLoss
    device     : cpu ou cuda
    scaler_amp : GradScaler pour AMP (None = pas d'AMP)

    Returns
    -------
    loss_mean : float, loss moyenne sur tous les batchs de l'époque
    """
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        if scaler_amp is not None:
            # Forward pass en demi-précision (float16)
            with torch.autocast(device_type=device.type):
                mu, sigma = model(X_batch)
                loss      = loss_fn(mu, sigma, y_batch)
            # Backward pass avec mise à l'échelle pour éviter l'underflow float16
            scaler_amp.scale(loss).backward()
            # Clipping du gradient pour stabiliser l'entraînement
            scaler_amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            mu, sigma = model(X_batch)
            loss      = loss_fn(mu, sigma, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_one_epoch(
    model: ArielHeteroscedasticMLP,
    loader: DataLoader,
    loss_fn: GaussianNLLLoss,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Évalue le modèle sur un DataLoader (validation ou test).

    @torch.no_grad() désactive le calcul du gradient — pas besoin
    pour l'inférence, économise de la mémoire et accélère le calcul.

    Returns
    -------
    val_loss    : float, loss NLL moyenne
    all_mu      : (n_samples, 283) — toutes les prédictions μ
    all_sigma   : (n_samples, 283) — tous les σ prédits
    """
    model.eval()
    total_loss = 0.0
    all_mu     = []
    all_sigma  = []

    for batch in loader:
        if len(batch) == 2:
            X_batch, y_batch = batch
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mu, sigma = model(X_batch)
            loss      = loss_fn(mu, sigma, y_batch)
            total_loss += loss.item()
        else:
            X_batch = batch[0].to(device)
            mu, sigma = model(X_batch)

        all_mu.append(mu.cpu().numpy())
        all_sigma.append(sigma.cpu().numpy())

    all_mu    = np.vstack(all_mu)
    all_sigma = np.vstack(all_sigma)
    val_loss  = total_loss / len(loader) if total_loss > 0 else 0.0

    return val_loss, all_mu, all_sigma


# =============================================================================
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """
    Arrête l'entraînement si le score de validation ne s'améliore plus.

    Surveille le score GLL (à maximiser) avec une patience configurable.
    Sauvegarde automatiquement le meilleur modèle (checkpoint).
    """

    def __init__(self, patience: int = 15, checkpoint_path: str = "models/best_model.pt"):
        """
        Parameters
        ----------
        patience        : nombre d'époques sans amélioration avant l'arrêt
        checkpoint_path : où sauvegarder le meilleur modèle
        """
        self.patience        = patience
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self.best_score      = -np.inf
        self.counter         = 0
        self.should_stop     = False

    def __call__(self, score: float, model: nn.Module):
        """
        Appeler à chaque époque avec le score de validation courant.

        Si le score s'améliore → sauvegarde du checkpoint, reset du compteur.
        Sinon → incrémente le compteur, déclenche l'arrêt si patience épuisée.
        """
        if score > self.best_score:
            self.best_score  = score
            self.counter     = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            logger.info(f"  ✅ Nouveau meilleur score : {score:.6f} → checkpoint sauvegardé")
        else:
            self.counter += 1
            logger.info(
                f"  ⏳ Pas d'amélioration ({self.counter}/{self.patience}) "
                f"| meilleur : {self.best_score:.6f}"
            )
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info("  🛑 Early stopping déclenché")


# =============================================================================
# MAIN
# =============================================================================

def main(use_mlflow: bool = True, device_str: str = "auto"):
    params   = load_params()
    m_cfg    = params["model"]
    pre_cfg  = params["preprocess"]
    data_cfg = params["data"]

    # --- Device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info(f"Device : {device}")

    PROCESSED_DIR = Path(data_cfg["processed_dir"])
    MODELS_DIR    = Path("models")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
    # =========================================================================
    logger.info("Chargement des données...")
    df_train = pd.read_csv(PROCESSED_DIR / "train_with_spectra.csv")
    df_test  = pd.read_csv(PROCESSED_DIR / "test_with_spectra.csv")

    X_all,    feat_cols = build_feature_matrix(df_train, drop_eccentricity=pre_cfg["drop_eccentricity"])
    y_all,    wl_cols   = build_target_matrix(df_train)
    X_kaggle, _         = build_feature_matrix(df_test,  drop_eccentricity=pre_cfg["drop_eccentricity"])

    # Split
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_all, y_all, np.arange(len(df_train)),
        test_size=pre_cfg["test_size"],
        random_state=pre_cfg["random_state"],
    )
    planet_ids_val    = df_train["planet_id"].values[idx_val]
    planet_ids_kaggle = df_test["planet_id"].values

    # Standardisation (fit sur X_train uniquement)
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    X_train_sc  = scaler_X.transform(X_train).astype(np.float32)
    X_val_sc    = scaler_X.transform(X_val).astype(np.float32)
    X_kaggle_sc = scaler_X.transform(X_kaggle).astype(np.float32)
    y_train_sc  = scaler_y.transform(y_train).astype(np.float32)
    y_val_sc    = scaler_y.transform(y_val).astype(np.float32)

    naive_mean, naive_sigma = compute_naive_stats(y_train)

    # DataLoaders
    batch_size  = m_cfg["batch_size"]
    train_ds    = ArielDataset(X_train_sc, y_train_sc)
    val_ds      = ArielDataset(X_val_sc,   y_val_sc)
    kaggle_ds   = ArielDataset(X_kaggle_sc)

    train_loader  = DataLoader(train_ds,  batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader    = DataLoader(val_ds,    batch_size=batch_size, shuffle=False, num_workers=0)
    kaggle_loader = DataLoader(kaggle_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info(f"Train : {len(X_train)} | Val : {len(X_val)} | Kaggle : {len(X_kaggle)}")

    # =========================================================================
    # 2. MODÈLE, LOSS, OPTIMISEUR, SCHEDULER
    # =========================================================================
    n_features = X_train_sc.shape[1]
    model = ArielHeteroscedasticMLP(
        n_features    = n_features,
        n_wavelengths = 283,
        d_model       = m_cfg["d_model"],
        n_layers      = m_cfg["n_layers"],
        dropout       = m_cfg["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Modèle : {n_params:,} paramètres entraînables")

    loss_fn   = GaussianNLLLoss(fgs_weight=1.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=m_cfg["learning_rate"],
        weight_decay=1e-4,      # régularisation L2 légère
    )

    # CosineAnnealingLR : réduit le LR progressivement en suivant une cosinus.
    # À la fin, le LR atteint eta_min. Favorise une convergence fine
    # vers la fin de l'entraînement sans avoir à choisir un schedule manuel.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=m_cfg["n_epochs"],
        eta_min=m_cfg["learning_rate"] * 1e-2,
    )

    # AMP (Automatic Mixed Precision) uniquement sur CUDA
    scaler_amp = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    early_stopping = EarlyStopping(
        patience=m_cfg["patience"],
        checkpoint_path=str(MODELS_DIR / "best_dl_model.pt"),
    )

    # =========================================================================
    # 3. BOUCLE D'ENTRAÎNEMENT
    # =========================================================================
    history = {"train_loss": [], "val_loss": [], "val_gll": [], "lr": []}

    mlflow_run = None
    if use_mlflow and MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
        mlflow.set_experiment("ariel-2025-deep-learning")
        mlflow_run = mlflow.start_run(run_name=f"HeteroscedasticMLP_d{m_cfg['d_model']}_L{m_cfg['n_layers']}")

        mlflow.log_params({
            "architecture":  m_cfg["architecture"],
            "d_model":       m_cfg["d_model"],
            "n_layers":      m_cfg["n_layers"],
            "dropout":       m_cfg["dropout"],
            "learning_rate": m_cfg["learning_rate"],
            "batch_size":    m_cfg["batch_size"],
            "n_epochs":      m_cfg["n_epochs"],
            "patience":      m_cfg["patience"],
            "n_params":      n_params,
            "n_features":    n_features,
            "device":        str(device),
        })

    logger.info("=" * 60)
    logger.info("Début de l'entraînement")
    logger.info("=" * 60)

    for epoch in range(1, m_cfg["n_epochs"] + 1):

        # --- Entraînement
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler_amp)

        # --- Évaluation
        val_loss, val_mu_sc, val_sigma_sc = evaluate_one_epoch(model, val_loader, loss_fn, device)

        # Inversion du scaler pour évaluation dans l'espace original
        val_mu    = scaler_y.inverse_transform(val_mu_sc)
        val_sigma = np.clip(val_sigma_sc * scaler_y.scale_, a_min=1e-9, a_max=None)

        # Score GLL officiel sur la validation
        val_gll = evaluate_on_validation(
            y_val_true=y_val,
            y_pred_mean=val_mu,
            y_pred_std=val_sigma,
            planet_ids_val=planet_ids_val,
            naive_mean=naive_mean,
            naive_sigma=naive_sigma,
        )

        # Learning rate courant
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # Logging
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_gll"].append(val_gll)
        history["lr"].append(current_lr)

        logger.info(
            f"Epoch {epoch:3d}/{m_cfg['n_epochs']} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_gll={val_gll:.6f} | "
            f"lr={current_lr:.2e}"
        )

        if mlflow_run:
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "val_gll":    val_gll,
                "lr":         current_lr,
            }, step=epoch)

        # Early stopping (surveille val_gll à maximiser)
        early_stopping(val_gll, model)
        if early_stopping.should_stop:
            logger.info(f"Entraînement arrêté à l'époque {epoch}")
            break

    # =========================================================================
    # 4. CHARGEMENT DU MEILLEUR MODÈLE ET INFÉRENCE KAGGLE
    # =========================================================================
    logger.info("Chargement du meilleur checkpoint...")
    model.load_state_dict(torch.load(early_stopping.checkpoint_path, map_location=device))
    best_gll = early_stopping.best_score
    logger.info(f"Meilleur score GLL validation : {best_gll:.6f}")

    # Inférence sur le test set Kaggle
    logger.info("Inférence sur le test set Kaggle...")
    _, kaggle_mu_sc, kaggle_sigma_sc = evaluate_one_epoch(model, kaggle_loader, loss_fn, device)

    kaggle_mu    = scaler_y.inverse_transform(kaggle_mu_sc)
    kaggle_sigma = np.clip(kaggle_sigma_sc * scaler_y.scale_, a_min=1e-9, a_max=None)

    df_submission = build_submission_dataframe(planet_ids_kaggle, kaggle_mu, kaggle_sigma)
    submission_path = MODELS_DIR / "submission_dl.csv"
    df_submission.to_csv(submission_path, index=False)
    logger.info(f"Soumission sauvegardée : {submission_path}")

    # =========================================================================
    # 5. MÉTRIQUES FINALES ET ARTEFACTS
    # =========================================================================
    metrics = {
        "val_gll_score": best_gll,
        "val_nll_loss":  float(min(history["val_loss"])),
        "n_epochs_run":  len(history["train_loss"]),
        "naive_mean":    naive_mean,
        "naive_sigma":   naive_sigma,
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Courbes d'apprentissage
    fig = _plot_training_curves(history)

    if mlflow_run:
        mlflow.log_metrics({
            "best_val_gll":  best_gll,
            "n_epochs_run":  len(history["train_loss"]),
        })
        mlflow.log_artifact("metrics.json")
        mlflow.log_artifact(str(submission_path))
        mlflow.log_figure(fig, "training_curves.png")
        mlflow.pytorch.log_model(model, "model")
        mlflow.end_run()
        logger.info("✅ Run MLflow finalisé")

    plt.close(fig)

    logger.info("=" * 60)
    logger.info(f"✅ Score GLL final : {best_gll:.6f}")
    logger.info("=" * 60)
    return best_gll


# =============================================================================
# UTILITAIRES
# =============================================================================

def _plot_training_curves(history: dict) -> plt.Figure:
    """Génère les courbes d'apprentissage (loss + GLL par époque)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss NLL
    axes[0].plot(epochs, history["train_loss"], label="Train NLL")
    axes[0].plot(epochs, history["val_loss"],   label="Val NLL")
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("NLL Loss")
    axes[0].set_title("Courbes de perte")
    axes[0].legend()

    # Score GLL
    axes[1].plot(epochs, history["val_gll"], color="green", label="Val GLL Score")
    axes[1].axhline(max(history["val_gll"]), color="red", linestyle="--",
                    label=f"Best : {max(history['val_gll']):.4f}")
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("Score GLL")
    axes[1].set_title("Score GLL sur validation")
    axes[1].legend()

    fig.tight_layout()
    return fig


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement Deep Learning Ariel 2025")
    parser.add_argument("--no-mlflow", action="store_true", help="Désactive MLflow")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device PyTorch (défaut: auto-détection)"
    )
    args = parser.parse_args()
    main(use_mlflow=not args.no_mlflow, device_str=args.device)
