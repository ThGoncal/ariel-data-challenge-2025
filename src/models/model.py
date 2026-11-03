"""
model.py — Modèle Deep Learning Ariel Data Challenge 2025
=========================================================
Architecture : MLP Hétéroscédastique avec tête probabiliste gaussienne.

Pourquoi ce choix ?
-------------------
La métrique GLL récompense deux choses simultanément :
    1. Une prédiction μ_pred proche de y_true
    2. Un σ_pred bien calibré (ni trop large, ni trop petit)

Un modèle hétéroscédastique prédit (μ, σ) pour chaque sortie
directement depuis l'entrée — σ varie selon la planète, ce qui
est physiquement correct (certaines planètes sont plus "faciles"
à caractériser que d'autres selon leurs paramètres stellaires).

Architecture :
    Input (289) → Encoder partagé → Tête μ (283) + Tête log_σ (283)

La séparation en deux têtes spécialisées permet à chaque branche
de se concentrer sur sa tâche : l'une minimise l'erreur de prédiction,
l'autre apprend à estimer la confiance.

Loss : Negative Log-Likelihood gaussienne (directement optimisée
       sur la même métrique que le challenge).
"""

# === Scientific stack
import numpy as np

# === PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# BLOCS DE BASE
# =============================================================================

class ResidualBlock(nn.Module):
    """
    Bloc résiduel pour MLP profond.

    Ajoute une connexion "skip" (raccourci) autour de deux couches linéaires.
    L'idée vient des ResNets (He et al., 2015) : plutôt que d'apprendre
    une transformation F(x), le bloc apprend la correction résiduelle F(x) + x.

    Avantages pour notre cas :
        - Permet des réseaux plus profonds sans dégradation du gradient
        - Initialisation plus stable : si F(x) ≈ 0, le bloc est une identité
        - Convergence plus rapide en pratique

    Architecture du bloc :
        x → LayerNorm → Linear → GELU → Dropout → Linear → Dropout → + x

    On utilise GELU plutôt que ReLU car il est plus doux aux bords
    (différentiable en 0) et tend à mieux fonctionner sur les données
    continues comme les spectres.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        """
        Parameters
        ----------
        dim     : dimension des features (entrée = sortie pour le skip)
        dropout : taux de dropout appliqué après chaque couche linéaire
        """
        super().__init__()

        self.norm    = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 2)   # expansion × 2 dans le bloc
        self.linear2 = nn.Linear(dim * 2, dim)   # retour à la dimension originale
        self.drop    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, dim)
        retour : (batch, dim)
        """
        residual = x                           # sauvegarde du raccourci

        out = self.norm(x)                     # normalisation avant transformation
        out = self.linear1(out)                # expansion
        out = F.gelu(out)                      # activation non-linéaire
        out = self.drop(out)
        out = self.linear2(out)                # contraction
        out = self.drop(out)

        return out + residual                  # addition du raccourci


# =============================================================================
# ENCODEUR PARTAGÉ
# =============================================================================

class SpectrumEncoder(nn.Module):
    """
    Encodeur partagé entre la tête μ et la tête σ.

    Transforme le vecteur de features (289 dimensions) en une représentation
    latente riche (d_model dimensions) via une projection d'entrée suivie
    de N blocs résiduels.

    Le partage des couches entre les deux têtes force l'encodeur à apprendre
    des représentations utiles à la fois pour la prédiction et pour
    l'estimation d'incertitude — les deux tâches partagent la même physique.
    """

    def __init__(self, n_features: int, d_model: int, n_layers: int, dropout: float):
        """
        Parameters
        ----------
        n_features : dimension de l'entrée (289 : 283 Rp + 6 astro)
        d_model    : dimension de l'espace latent
        n_layers   : nombre de blocs résiduels
        dropout    : taux de dropout
        """
        super().__init__()

        # Projection d'entrée : amène le vecteur de features à d_model
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Pile de blocs résiduels
        self.blocks = nn.ModuleList([
            ResidualBlock(d_model, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, n_features)
        retour : (batch, d_model)
        """
        out = self.input_proj(x)

        for block in self.blocks:
            out = block(out)

        return out


# =============================================================================
# MODÈLE PRINCIPAL : MLP HÉTÉROSCÉDASTIQUE
# =============================================================================

class ArielHeteroscedasticMLP(nn.Module):
    """
    MLP hétéroscédastique pour la prédiction de spectres d'exoplanètes.

    Prédit simultanément :
        - μ (mu_pred)     : spectre de transmission prédit (283 valeurs)
        - σ (sigma_pred)  : incertitude par longueur d'onde (283 valeurs)

    L'hétéroscédasticité signifie que σ dépend de l'entrée x :
    σ = σ(x). Cela permet au modèle d'exprimer plus de confiance sur
    certaines planètes/longueurs d'onde que sur d'autres, ce qui est
    physiquement motivé (une étoile plus brillante → meilleur SNR → σ plus petit).

    Deux têtes de sortie séparées après l'encodeur partagé :
        - mu_head    : Linear(d_model → n_wavelengths)
        - log_sigma_head : Linear(d_model → n_wavelengths)
          → exp() pour garantir σ > 0

    On prédit log(σ) plutôt que σ directement pour deux raisons :
        1. log(σ) peut prendre toutes les valeurs réelles → plus facile
           à optimiser sans contrainte
        2. exp(log_σ) garantit σ > 0 sans clip artificiel
    """

    def __init__(
        self,
        n_features: int = 289,
        n_wavelengths: int = 283,
        d_model: int = 256,
        n_layers: int = 4,
        dropout: float = 0.1,
        log_sigma_min: float = -10.0,   # σ_min = exp(-10) ≈ 4.5e-5
        log_sigma_max: float = 0.0,     # σ_max = exp(0)  = 1.0
    ):
        """
        Parameters
        ----------
        n_features    : dimension de l'entrée (283 Rp + 6 features astro)
        n_wavelengths : nombre de longueurs d'onde de sortie (283)
        d_model       : dimension de l'espace latent
        n_layers      : nombre de blocs résiduels dans l'encodeur
        dropout       : taux de dropout (entraînement uniquement)
        log_sigma_min : borne inf du clipping de log_σ (stabilité numérique)
        log_sigma_max : borne sup du clipping de log_σ
        """
        super().__init__()

        self.n_wavelengths = n_wavelengths
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max

        # Encodeur partagé entre les deux têtes
        self.encoder = SpectrumEncoder(
            n_features=n_features,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Tête de prédiction de la moyenne μ
        self.mu_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_wavelengths),
            nn.Softplus(),    # garantit μ > 0 (les spectres sont positifs)
        )

        # Tête de prédiction de l'incertitude log(σ)
        self.log_sigma_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_wavelengths),
        )

        # Initialisation des poids
        self._init_weights()

    def _init_weights(self):
        """
        Initialisation de Xavier uniform pour les couches linéaires.

        L'initialisation par défaut de PyTorch (Kaiming) est optimisée
        pour ReLU. Xavier est meilleure pour GELU/Softplus car elle
        tient compte de la variance des activations dans les deux sens
        (forward et backward).
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (batch, n_features) — features normalisées

        Returns
        -------
        mu        : (batch, n_wavelengths) — moyennes prédites, > 0
        sigma     : (batch, n_wavelengths) — écarts-types prédits, > 0
        """
        # Encodage partagé
        latent = self.encoder(x)                              # (batch, d_model)

        # Tête μ
        mu = self.mu_head(latent)                             # (batch, 283)

        # Tête log(σ) avec clipping pour stabilité numérique
        log_sigma = self.log_sigma_head(latent)               # (batch, 283)
        log_sigma = torch.clamp(
            log_sigma,
            min=self.log_sigma_min,
            max=self.log_sigma_max,
        )
        sigma = torch.exp(log_sigma)                          # (batch, 283)

        return mu, sigma


# =============================================================================
# LOSS : NEGATIVE LOG-LIKELIHOOD GAUSSIENNE
# =============================================================================

class GaussianNLLLoss(nn.Module):
    """
    Loss NLL gaussienne — directement alignée avec la métrique du challenge.

    La log-vraisemblance d'une observation y sous N(μ, σ²) est :
        log P(y | μ, σ) = -0.5 × [(y - μ)² / σ²  +  log(σ²)  +  log(2π)]

    On minimise la negative log-likelihood (NLL) :
        NLL = 0.5 × [(y - μ)² / σ²  +  log(σ²)]
              (le terme log(2π) est une constante, on l'ignore)

    Cette loss a une propriété remarquable : elle pénalise à la fois
    les erreurs de prédiction ET les mauvaises calibrations d'incertitude.

    Intuition :
        - Si σ est trop grand  : le terme log(σ²) augmente
        - Si σ est trop petit et μ est faux : le terme (y-μ)²/σ² explose
        - Optimum : σ capture exactement l'erreur résiduelle irréductible

    Pondération FGS1 / AIRS :
        Le canal FGS1 (indice 0) a un poids différent des canaux AIRS
        dans la métrique officielle. On applique la même pondération
        dans la loss pour aligner l'optimisation sur l'évaluation.
    """

    def __init__(self, fgs_weight: float = 1.0, reduction: str = "mean"):
        """
        Parameters
        ----------
        fgs_weight : poids du canal FGS1 (indice 0) vs canaux AIRS
        reduction  : "mean" (moyenne sur batch×wavelengths) ou "sum"
        """
        super().__init__()
        self.fgs_weight = fgs_weight
        self.reduction  = reduction

    def forward(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        mu     : (batch, n_wl) — moyennes prédites
        sigma  : (batch, n_wl) — écarts-types prédits, > 0
        y_true : (batch, n_wl) — vraies valeurs

        Returns
        -------
        loss : scalaire
        """
        # Terme d'erreur quadratique normalisé
        squared_error = ((y_true - mu) ** 2) / (sigma ** 2)    # (batch, n_wl)

        # Terme de régularisation sur σ (pénalise les σ trop grands)
        log_var       = 2 * torch.log(sigma)                    # log(σ²) = 2×log(σ)

        # NLL par élément
        nll           = 0.5 * (squared_error + log_var)         # (batch, n_wl)

        # Pondération FGS1 (canal 0) vs AIRS (canaux 1..282)
        if self.fgs_weight != 1.0:
            weights         = torch.ones_like(nll)
            weights[:, 0]   = self.fgs_weight
            nll             = nll * weights

        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        else:
            return nll


# =============================================================================
# DATASET PYTORCH
# =============================================================================

class ArielDataset(torch.utils.data.Dataset):
    """
    Dataset PyTorch pour le challenge Ariel.

    Encapsule les arrays numpy X et y en un Dataset itérable
    compatible avec DataLoader (batching, shuffling, multiprocessing).
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        device: str = "cpu",
    ):
        """
        Parameters
        ----------
        X      : (n_samples, n_features) — features normalisées
        y      : (n_samples, n_wavelengths) — targets (None pour le test set)
        device : "cpu" ou "cuda" — où stocker les tenseurs
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple:
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return (self.X[idx],)
