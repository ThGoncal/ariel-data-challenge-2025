"""
hetero.py — Modèles à incertitude σ(x) pour la métrique GLL (Ariel Data Challenge 2025)
=========================================================================================
Trois modèles, du plus simple au plus riche, tous avec la même interface tableaux :

    f(Xtr, ytr, Xte) -> Prediction(mu, sigma)      # mu, sigma : (n_test, 283)

  * ridge_const   : Ridge + σ constant par longueur d'onde  (= baseline v0.1.0)
  * ridge_scaled  : échelon A — μ du Ridge, σ_ij = c · σ_j · s(x_i) ; s(x) est un facteur
                    d'échelle PAR PLANÈTE appris sur les résidus hors-échantillon
  * mlp_hetero    : échelon B — ensemble de petits MLP à deux têtes (μ, log σ), entraînés en
                    minimisant la log-vraisemblance gaussienne (la métrique elle-même)

Principes de stabilité (peu de planètes : ~20 à ~1100) :
  * les résidus servant à calibrer σ sont HORS-ÉCHANTILLON (K-fold) : les résidus d'entraînement
    sont trop petits et donneraient des σ trop optimistes ;
  * le MLP part du Ridge (têtes initialisées à ~0) : au pas 0 il est identique à la baseline ;
  * log σ est borné autour de la valeur du Ridge (±2) : pas d'effondrement de σ ;
  * l'époque retenue est celle du meilleur NLL sur une portion mise de côté : le MLP ne peut pas
    finir pire que le point de départ sur cette portion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

EPS = 1e-12


@dataclass
class Prediction:
    mu: np.ndarray      # (n_test, n_wl)
    sigma: np.ndarray   # (n_test, n_wl)


def _scaled(Xtr: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sc = StandardScaler().fit(Xtr)                      # fit sur le train uniquement
    return sc.transform(Xtr), sc.transform(Xte)


def _oof_ridge(Ztr: np.ndarray, ytr: np.ndarray, alpha: float, n_splits: int, seed: int) -> np.ndarray:
    """Prédictions Ridge hors-échantillon (chaque planète prédite par un modèle qui ne l'a pas vue)."""
    k = max(2, min(n_splits, len(Ztr)))
    cv = KFold(n_splits=k, shuffle=True, random_state=seed)
    return cross_val_predict(Ridge(alpha=alpha), Ztr, ytr, cv=cv)


# --------------------------------------------------------------------------- baseline

def ridge_const(Xtr, ytr, Xte, alpha: float = 10.0) -> Prediction:
    """Baseline v0.1.0 : σ = écart-type des résidus d'entraînement, identique pour toutes les planètes."""
    Ztr, Zte = _scaled(Xtr, Xte)
    model = Ridge(alpha=alpha).fit(Ztr, ytr)
    sigma = np.maximum((ytr - model.predict(Ztr)).std(axis=0), 1e-6)
    mu = model.predict(Zte)
    return Prediction(mu, np.tile(sigma, (len(mu), 1)))


# --------------------------------------------------------------------------- échelon A

def ridge_scaled(Xtr, ytr, Xte, alpha: float = 10.0, scale_alpha: float = 30.0,
                 n_splits: int = 5, seed: int = 0, log_clip: float = 1.0) -> Prediction:
    """μ du Ridge ; σ_ij = c · σ_j · exp(t̂(x_i)).

    t_i = ½ log(moyenne_j z_ij²) mesure à quel point la planète i est « plus bruitée que la moyenne »
    (z = résidu hors-échantillon / σ_j). t̂(x) est une régression Ridge de t sur les features ;
    c est une calibration globale pour que la variance normalisée hors-échantillon vaille 1.
    """
    Ztr, Zte = _scaled(Xtr, Xte)
    oof = _oof_ridge(Ztr, ytr, alpha, n_splits, seed)
    res = ytr - oof
    sig_j = np.maximum(res.std(axis=0), EPS)
    z = res / sig_j
    t = 0.5 * np.log(np.mean(z ** 2, axis=1) + EPS)

    k = max(2, min(n_splits, len(Ztr)))
    cv = KFold(n_splits=k, shuffle=True, random_state=seed)
    t_oof = cross_val_predict(Ridge(alpha=scale_alpha), Ztr, t, cv=cv)
    s_oof = np.exp(np.clip(t_oof, -log_clip, log_clip))
    c = float(np.sqrt(np.mean((z / s_oof[:, None]) ** 2)))

    t_hat = Ridge(alpha=scale_alpha).fit(Ztr, t).predict(Zte)
    s_te = np.exp(np.clip(t_hat, -log_clip, log_clip))
    mu = Ridge(alpha=alpha).fit(Ztr, ytr).predict(Zte)
    sigma = np.maximum(c * s_te[:, None] * sig_j[None, :], 1e-6)
    return Prediction(mu, sigma)


# --------------------------------------------------------------------------- échelon B

def mlp_hetero(Xtr, ytr, Xte, alpha: float = 10.0, n_models: int = 5, hidden: int = 64,
               dropout: float = 0.2, weight_decay: float = 1e-3, lr: float = 3e-3,
               epochs: int = 400, patience: int = 40, holdout: float = 0.2,
               n_splits: int = 5, seed: int = 0) -> Prediction:
    """Ensemble de MLP (μ, log σ) entraînés sur la NLL gaussienne, μ = Ridge + correction apprise."""
    import torch
    import torch.nn as nn

    Ztr, Zte = _scaled(Xtr, Xte)
    y_mean, y_std = ytr.mean(axis=0), np.maximum(ytr.std(axis=0), EPS)
    yz = (ytr - y_mean) / y_std                                    # cibles normalisées par λ
    oof = _oof_ridge(Ztr, ytr, alpha, n_splits, seed)
    base_tr = (oof - y_mean) / y_std                               # point de départ de μ (hors-échantillon)
    base_te = (Ridge(alpha=alpha).fit(Ztr, ytr).predict(Zte) - y_mean) / y_std
    b = np.log(np.maximum(((ytr - oof) / y_std).std(axis=0), 1e-6))   # log σ de départ, par λ

    f32 = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
    Xt, Yt, Bt, Xe, Be, bt = f32(Ztr), f32(yz), f32(base_tr), f32(Zte), f32(base_te), f32(b)
    n, d_in, d_out = Xt.shape[0], Xt.shape[1], Yt.shape[1]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
            self.mu, self.ls = nn.Linear(hidden, d_out), nn.Linear(hidden, d_out)
            for head in (self.mu, self.ls):                        # départ ≈ Ridge à σ constant
                nn.init.normal_(head.weight, std=1e-3)
                nn.init.zeros_(head.bias)

        def forward(self, x, base):
            h = self.body(x)
            return base + self.mu(h), bt + 2.0 * torch.tanh(self.ls(h))   # log σ borné : b ± 2

    def nll(mu, logsig, y):
        return (0.5 * ((y - mu) / torch.exp(logsig)) ** 2 + logsig).mean()

    mus, sigs = [], []
    for m in range(n_models):
        g = torch.Generator().manual_seed(seed + m)
        torch.manual_seed(seed + m)
        perm = torch.randperm(n, generator=g)
        n_ho = max(1, int(round(holdout * n))) if n >= 5 else 0
        ho, tr = perm[:n_ho], perm[n_ho:]
        net = Net()
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

        def ho_loss():
            net.eval()
            with torch.no_grad():
                return float(nll(*net(Xt[ho], Bt[ho]), Yt[ho])) if n_ho else 0.0

        best, best_state, wait = ho_loss(), {k: v.clone() for k, v in net.state_dict().items()}, 0
        for _ in range(epochs):
            net.train()
            opt.zero_grad()
            loss = nll(*net(Xt[tr], Bt[tr]), Yt[tr])
            loss.backward()
            opt.step()
            cur = ho_loss()
            if cur < best - 1e-6:
                best, wait = cur, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            mu, ls = net(Xe, Be)
        mus.append(mu.numpy())
        sigs.append(np.exp(ls.numpy()))

    mus, sigs = np.stack(mus), np.stack(sigs)
    mu_z = mus.mean(axis=0)
    var_z = (sigs ** 2 + mus ** 2).mean(axis=0) - mu_z ** 2        # variance du mélange (ensemble)
    return Prediction(y_mean + y_std * mu_z, np.maximum(y_std * np.sqrt(np.maximum(var_z, EPS)), 1e-6))


MODELS = {"ridge-const": ridge_const, "ridge-scaled": ridge_scaled, "mlp-hetero": mlp_hetero}
