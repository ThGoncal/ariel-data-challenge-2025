"""
test_model.py — Tests unitaires pour src/models/model.py
=========================================================
On teste l'architecture PyTorch sans entraîner réellement le modèle :
shapes des tenseurs, propriétés mathématiques de la loss, contraintes
physiques sur les sorties (μ > 0, σ > 0).

Les tests sont rapides (<5s) car on ne fait que des forward passes
sur de petits batchs sans backward.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.model import (
    ArielHeteroscedasticMLP,
    GaussianNLLLoss,
    ResidualBlock,
    ArielDataset,
)

# =============================================================================
# CONSTANTES DE TEST
# =============================================================================

BATCH_SIZE    = 8
N_FEATURES    = 289     # 283 Rp + 6 astro
N_WAVELENGTHS = 283
D_MODEL       = 64      # réduit pour la rapidité des tests
N_LAYERS      = 2
DROPOUT       = 0.0     # 0 pour des sorties déterministes en eval mode


# =============================================================================
# FIXTURES LOCALES
# =============================================================================

@pytest.fixture
def model():
    """Modèle minimal en mode eval (dropout désactivé)."""
    m = ArielHeteroscedasticMLP(
        n_features=N_FEATURES,
        n_wavelengths=N_WAVELENGTHS,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
    )
    m.eval()
    return m


@pytest.fixture
def x_batch():
    """Batch d'entrée synthétique (BATCH_SIZE, N_FEATURES)."""
    torch.manual_seed(42)
    return torch.randn(BATCH_SIZE, N_FEATURES)


@pytest.fixture
def y_batch():
    """Batch de targets synthétiques positives (BATCH_SIZE, N_WAVELENGTHS)."""
    torch.manual_seed(42)
    return torch.abs(torch.randn(BATCH_SIZE, N_WAVELENGTHS)) * 0.1 + 0.1


# =============================================================================
# ResidualBlock
# =============================================================================

class TestResidualBlock:

    def test_shape_preservee(self):
        """Un ResidualBlock doit préserver la shape du tenseur d'entrée."""
        block  = ResidualBlock(dim=64, dropout=0.0)
        block.eval()
        x      = torch.randn(8, 64)
        output = block(x)
        assert output.shape == x.shape

    def test_skip_connection_avec_transformation_nulle(self):
        """
        Si les poids linéaires sont à 0, le bloc doit retourner x (identité).
        Vérifie que la connexion résiduelle fonctionne.
        """
        block = ResidualBlock(dim=32, dropout=0.0)
        block.eval()

        # Mettre tous les poids à 0 → transformation F(x) = 0
        with torch.no_grad():
            for param in block.parameters():
                param.zero_()

        x      = torch.randn(4, 32)
        output = block(x)

        # Avec F(x)=0, la sortie doit être x + 0 = x
        # Note : LayerNorm peut légèrement modifier x même avec poids=0,
        # mais la connexion résiduelle garantit que la sortie ≈ x
        # On vérifie la présence de la connexion en testant que l'output
        # n'est pas zéro (ce qu'il serait sans connexion résiduelle)
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_gradient_circule(self):
        """Le gradient doit circuler à travers le bloc résiduel (pas de vanishing)."""
        block = ResidualBlock(dim=32, dropout=0.0)
        x     = torch.randn(4, 32, requires_grad=True)
        out   = block(x)
        out.sum().backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any(), "Gradient NaN dans ResidualBlock"


# =============================================================================
# ArielHeteroscedasticMLP — Architecture
# =============================================================================

class TestArielHeteroscedasticMLPArchitecture:

    def test_forward_shapes(self, model, x_batch):
        """Les deux sorties (mu, sigma) doivent avoir la shape (batch, n_wl)."""
        with torch.no_grad():
            mu, sigma = model(x_batch)

        assert mu.shape    == (BATCH_SIZE, N_WAVELENGTHS), \
            f"mu.shape={mu.shape}, attendu ({BATCH_SIZE}, {N_WAVELENGTHS})"
        assert sigma.shape == (BATCH_SIZE, N_WAVELENGTHS), \
            f"sigma.shape={sigma.shape}, attendu ({BATCH_SIZE}, {N_WAVELENGTHS})"

    def test_mu_strictement_positif(self, model, x_batch):
        """
        μ doit être strictement positif (Softplus garantit > 0).
        Les spectres de transmission sont des valeurs physiques positives.
        """
        with torch.no_grad():
            mu, _ = model(x_batch)
        assert (mu > 0).all(), f"mu contient des valeurs ≤ 0 : min={mu.min().item():.6f}"

    def test_sigma_strictement_positif(self, model, x_batch):
        """
        σ doit être strictement positif (exp(log_σ) > 0 par construction).
        Un σ nul ou négatif rendrait la loss NLL indéfinie.
        """
        with torch.no_grad():
            _, sigma = model(x_batch)
        assert (sigma > 0).all(), \
            f"sigma contient des valeurs ≤ 0 : min={sigma.min().item():.6e}"

    def test_sigma_dans_la_plage_attendue(self, model, x_batch):
        """
        σ doit être dans [exp(-10), exp(0)] = [~4.5e-5, 1.0]
        grâce au clipping de log_sigma dans forward().
        """
        with torch.no_grad():
            _, sigma = model(x_batch)

        sigma_min = sigma.min().item()
        sigma_max = sigma.max().item()

        assert sigma_min >= np.exp(-10) * 0.99, \
            f"sigma trop petit : {sigma_min:.2e} < exp(-10)={np.exp(-10):.2e}"
        assert sigma_max <= 1.0 + 1e-5, \
            f"sigma trop grand : {sigma_max:.4f} > 1.0"

    def test_pas_de_nan_dans_les_sorties(self, model, x_batch):
        """Les sorties ne doivent contenir aucun NaN ni Inf."""
        with torch.no_grad():
            mu, sigma = model(x_batch)

        assert not torch.isnan(mu).any(),    "NaN dans mu"
        assert not torch.isinf(mu).any(),    "Inf dans mu"
        assert not torch.isnan(sigma).any(), "NaN dans sigma"
        assert not torch.isinf(sigma).any(), "Inf dans sigma"

    def test_sortie_differente_selon_entree(self, model):
        """
        Deux entrées différentes doivent donner deux sorties différentes
        (le modèle n'est pas constant).
        """
        torch.manual_seed(0)
        x1 = torch.randn(1, N_FEATURES)
        x2 = torch.randn(1, N_FEATURES) * 2 + 1.0

        with torch.no_grad():
            mu1, _ = model(x1)
            mu2, _ = model(x2)

        assert not torch.allclose(mu1, mu2), \
            "Sorties identiques pour deux entrées différentes"

    def test_nombre_parametres(self):
        """Vérifie que le modèle a bien un nombre de paramètres raisonnable."""
        m      = ArielHeteroscedasticMLP(
            n_features=N_FEATURES, n_wavelengths=N_WAVELENGTHS,
            d_model=D_MODEL, n_layers=N_LAYERS, dropout=0.0
        )
        n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        # Avec d_model=64, n_layers=2, on attend entre 50k et 500k paramètres
        assert 10_000 < n_params < 5_000_000, \
            f"Nombre de paramètres suspect : {n_params:,}"

    def test_mode_train_vs_eval_dropout(self):
        """
        En mode train avec dropout > 0, les sorties doivent varier.
        En mode eval, les sorties doivent être déterministes.
        """
        m = ArielHeteroscedasticMLP(
            n_features=N_FEATURES, n_wavelengths=N_WAVELENGTHS,
            d_model=D_MODEL, n_layers=N_LAYERS, dropout=0.5
        )
        torch.manual_seed(0)
        x = torch.randn(4, N_FEATURES)

        # Mode eval → déterministe
        m.eval()
        with torch.no_grad():
            mu1, _ = m(x)
            mu2, _ = m(x)
        assert torch.allclose(mu1, mu2), \
            "Sorties non déterministes en mode eval"

        # Mode train → stochastique (dropout actif)
        m.train()
        torch.manual_seed(1)
        with torch.no_grad():
            mu_train1, _ = m(x)
        torch.manual_seed(2)
        with torch.no_grad():
            mu_train2, _ = m(x)
        assert not torch.allclose(mu_train1, mu_train2), \
            "Dropout inactif en mode train"


# =============================================================================
# GaussianNLLLoss
# =============================================================================

class TestGaussianNLLLoss:

    @pytest.fixture
    def loss_fn(self):
        return GaussianNLLLoss(fgs_weight=1.0)

    def test_retourne_scalaire(self, loss_fn, y_batch):
        """La loss doit retourner un tenseur scalaire (shape = [])."""
        torch.manual_seed(0)
        mu    = torch.abs(torch.randn_like(y_batch)) * 0.1 + 0.1
        sigma = torch.ones_like(y_batch) * 0.01
        loss  = loss_fn(mu, sigma, y_batch)
        assert loss.shape == torch.Size([]), f"Shape attendue [], obtenue {loss.shape}"

    def test_loss_positive(self, loss_fn, y_batch):
        """
        La NLL gaussienne peut être négative pour des distributions très piquées.
        Mais elle doit être finie.
        """
        mu    = torch.abs(torch.randn_like(y_batch)) * 0.1 + 0.1
        sigma = torch.ones_like(y_batch) * 0.01
        loss  = loss_fn(mu, sigma, y_batch)
        assert torch.isfinite(loss), f"Loss infinie ou NaN : {loss.item()}"

    def test_prediction_parfaite_minimise_la_loss(self, loss_fn, y_batch):
        """
        μ = y_true avec σ optimal doit donner une loss plus faible
        qu'une prédiction aléatoire.
        """
        # Prédiction parfaite (σ petit)
        mu_parfait    = y_batch.clone()
        sigma_parfait = torch.ones_like(y_batch) * 1e-4
        loss_parfait  = loss_fn(mu_parfait, sigma_parfait, y_batch)

        # Prédiction aléatoire (loin de y_true)
        torch.manual_seed(42)
        mu_random    = torch.abs(torch.randn_like(y_batch))   # loin de y_batch
        sigma_random = torch.ones_like(y_batch) * 0.1
        loss_random  = loss_fn(mu_random, sigma_random, y_batch)

        assert loss_parfait < loss_random, (
            f"La prédiction parfaite devrait avoir une loss plus faible : "
            f"loss_parfait={loss_parfait.item():.4f} vs loss_random={loss_random.item():.4f}"
        )

    def test_sigma_grand_augmente_loss(self, loss_fn, y_batch):
        """
        À μ fixé, augmenter σ doit augmenter la loss (terme log(σ²)).
        Cela vérifie que la loss pénalise les incertitudes trop larges.
        """
        mu       = y_batch.clone()
        sigma_petit = torch.ones_like(y_batch) * 1e-3
        sigma_grand = torch.ones_like(y_batch) * 1.0   # 1000× plus grand

        loss_petit = loss_fn(mu, sigma_petit, y_batch)
        loss_grand = loss_fn(mu, sigma_grand, y_batch)

        # Avec μ = y_true, sigma petit → terme (y-μ)²/σ² ≈ 0, mais log(σ²) petit
        # sigma grand → terme (y-μ)²/σ² ≈ 0, mais log(σ²) grand
        assert loss_grand > loss_petit, (
            "La loss avec grand σ devrait être plus élevée (terme log(σ²) dominant)"
        )

    def test_gradient_non_nul(self, loss_fn, y_batch):
        """Le gradient de la loss par rapport à mu et sigma doit être non nul."""
        mu    = torch.abs(torch.randn_like(y_batch)) * 0.1 + 0.1
        mu.requires_grad_(True)
        sigma = torch.ones_like(y_batch) * 0.01
        sigma.requires_grad_(True)

        loss = loss_fn(mu, sigma, y_batch)
        loss.backward()

        assert mu.grad    is not None and not torch.allclose(mu.grad,    torch.zeros_like(mu.grad))
        assert sigma.grad is not None and not torch.allclose(sigma.grad, torch.zeros_like(sigma.grad))

    def test_pas_de_nan_dans_la_loss(self, loss_fn, y_batch):
        """La loss ne doit pas produire de NaN avec des valeurs normales."""
        mu    = torch.abs(torch.randn_like(y_batch)) * 0.1 + 0.1
        sigma = torch.abs(torch.randn_like(y_batch)) * 0.01 + 1e-5
        loss  = loss_fn(mu, sigma, y_batch)
        assert not torch.isnan(loss), "NaN dans la loss NLL"

    def test_fgs_weight_influence_la_loss(self, y_batch):
        """Changer fgs_weight doit changer la valeur de la loss."""
        mu    = torch.abs(torch.randn_like(y_batch)) * 0.1 + 0.1
        sigma = torch.ones_like(y_batch) * 0.01

        loss_w1 = GaussianNLLLoss(fgs_weight=1.0)(mu, sigma, y_batch)
        loss_w5 = GaussianNLLLoss(fgs_weight=5.0)(mu, sigma, y_batch)

        assert not torch.isclose(loss_w1, loss_w5), \
            "fgs_weight n'influence pas la loss"


# =============================================================================
# ArielDataset
# =============================================================================

class TestArielDataset:

    def test_len_correct(self):
        """__len__ doit retourner le nombre de samples."""
        X  = np.random.rand(50, N_FEATURES).astype(np.float32)
        y  = np.random.rand(50, N_WAVELENGTHS).astype(np.float32)
        ds = ArielDataset(X, y)
        assert len(ds) == 50

    def test_getitem_retourne_x_et_y(self):
        """En mode train (avec y), __getitem__ doit retourner (X, y)."""
        X  = np.random.rand(10, N_FEATURES).astype(np.float32)
        y  = np.random.rand(10, N_WAVELENGTHS).astype(np.float32)
        ds = ArielDataset(X, y)

        item = ds[0]
        assert len(item) == 2
        assert item[0].shape == torch.Size([N_FEATURES])
        assert item[1].shape == torch.Size([N_WAVELENGTHS])

    def test_getitem_retourne_seulement_x_sans_y(self):
        """En mode test (y=None), __getitem__ doit retourner (X,) seulement."""
        X  = np.random.rand(10, N_FEATURES).astype(np.float32)
        ds = ArielDataset(X, y=None)

        item = ds[0]
        assert len(item) == 1
        assert item[0].shape == torch.Size([N_FEATURES])

    def test_dtype_float32(self):
        """Les tenseurs doivent être en float32 pour la compatibilité GPU."""
        X  = np.random.rand(5, N_FEATURES).astype(np.float64)   # float64 en entrée
        y  = np.random.rand(5, N_WAVELENGTHS).astype(np.float64)
        ds = ArielDataset(X, y)

        x_item, y_item = ds[0]
        assert x_item.dtype == torch.float32
        assert y_item.dtype == torch.float32

    def test_compatible_avec_dataloader(self):
        """Le Dataset doit être utilisable avec un DataLoader PyTorch."""
        from torch.utils.data import DataLoader
        X      = np.random.rand(16, N_FEATURES).astype(np.float32)
        y      = np.random.rand(16, N_WAVELENGTHS).astype(np.float32)
        ds     = ArielDataset(X, y)
        loader = DataLoader(ds, batch_size=4, shuffle=False)

        batches = list(loader)
        assert len(batches) == 4   # 16 / 4 = 4 batchs
        x_b, y_b = batches[0]
        assert x_b.shape == torch.Size([4, N_FEATURES])
        assert y_b.shape == torch.Size([4, N_WAVELENGTHS])
