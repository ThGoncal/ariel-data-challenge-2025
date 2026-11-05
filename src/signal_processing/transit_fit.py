"""
transit_fit.py — Profondeur de transit par longueur d'onde, avec fenêtre adaptée à chaque planète
===================================================================================================
Remplace l'hypothèse de calcul_Rp_median (« le transit occupe ±10 % autour du milieu de la série »)
par une mesure sur la courbe de lumière blanche (somme de toutes les longueurs d'onde, très haut
rapport signal/bruit), puis estime la profondeur de chaque longueur d'onde.

Entrées : F (n_time, n_wl), flux sommé sur l'axe spatial, calibré et binné (c'est ce que sauvegarde
scripts/check_transit_window.py) ; sorties : profondeur (≈ (Rp/Rs)², au facteur d'assombrissement
du bord près), PAS sa racine.

Méthodes de extract_depths :
  "fixe"             fenêtre ±10 % autour du milieu (reproduit calcul_Rp_median, sans la racine)
  "adaptee"          fenêtre détectée sur la courbe blanche, médianes dans / hors transit
  "adaptee_detrend"  idem, après division par une ligne de base polynomiale (degré `deg`) ajustée
                     sur les frames hors transit, canal par canal
  "gabarit"          idem, puis ajustement du gabarit de transit de la courbe blanche :
                     d_j = Σ_t T_t·y_jt / Σ_t T_t²   (utilise toutes les frames du transit, pas la médiane)
"""
from __future__ import annotations

import numpy as np

METHODS = ("fixe", "adaptee", "adaptee_detrend", "gabarit")


# --------------------------------------------------------------------------- détection de la fenêtre

def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    pad = np.pad(x, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(pad, np.ones(w) / w, mode="valid")


def _run(s: np.ndarray, level: float, i0: int) -> tuple[int, int]:
    """Plus grand intervalle contigu autour de i0 où s < level."""
    a = b = i0
    while a > 0 and s[a - 1] < level:
        a -= 1
    while b < len(s) - 1 and s[b + 1] < level:
        b += 1
    return a, b


def transit_window(flux: np.ndarray, edge: float = 0.10) -> dict:
    """Mesure le transit sur une courbe de lumière blanche et compare fenêtre fixe / fenêtre adaptée.

    flux : (n_time,) flux total. Les `edge` (10 %) premières et dernières frames servent de référence
    hors transit pour la détection.
    """
    flux = np.asarray(flux, dtype=float)
    n = len(flux)
    k = max(int(edge * n), 5)
    # tendance : droite passant par la médiane du premier et du dernier bord (une dérive plus grande que la
    # profondeur du transit ferait trouver le « minimum » au bord de la série, pas dans le transit)
    xl, xr = (k - 1) / 2.0, n - 1 - (k - 1) / 2.0
    ml, mr = float(np.median(flux[:k])), float(np.median(flux[-k:]))
    f = flux / (ml + (mr - ml) * (np.arange(n) - xl) / (xr - xl))
    w = max(5, n // 100)
    s = _smooth(f, w)
    noise = 1.4826 * float(np.median(np.abs(np.r_[f[:k], f[-k:]] - 1.0)))
    c = max(int(0.01 * n), 2)

    # --- méthode actuelle (calcul_Rp_median) appliquée à la courbe blanche
    mid, win = n // 2, max(n // 10, 1)
    depth_fixed = 1.0 - np.median(f[mid - win: mid + win]) / np.median(np.r_[f[: mid - win], f[mid + win:]])

    i0 = int(np.argmin(s))
    lo, hi = max(i0 - c, 0), min(i0 + c + 1, n)
    depth_bottom = 1.0 - float(np.median(s[lo:hi]))
    out = {"n": n, "noise": noise, "depth_bottom": depth_bottom, "depth_fixed": float(depth_fixed),
           "window_frac": 2 * win / n, "detected": False}
    if depth_bottom < 5 * noise / np.sqrt(w) + 1e-12:
        return out                                            # pas de transit détectable

    s50, e50 = _run(s, 1.0 - 0.5 * depth_bottom, i0)
    has10 = 0.1 * depth_bottom > 3 * noise / np.sqrt(w)
    s10, e10 = _run(s, 1.0 - 0.1 * depth_bottom, i0) if has10 else (s50, e50)
    dur50, dur10 = (e50 - s50 + 1) / n, (e10 - s10 + 1) / n

    # fenêtre adaptée : fond du transit (sans les bords) contre le hors-transit élargi d'une marge
    m_in = max(int(0.15 * (e50 - s50)), 1)
    m_out = max(int(0.02 * n), 2) + (0 if has10 else int(0.5 * (e50 - s50)))
    a_in, b_in = min(s50 + m_in, e50), max(e50 - m_in, s50)
    outside = np.r_[f[: max(s10 - m_out, 0)], f[min(e10 + m_out + 1, n):]]
    depth_adapt = 1.0 - np.median(f[a_in: b_in + 1]) / np.median(outside) if len(outside) else np.nan

    # la médiane résiste tant que moins de ~40 % des frames « hors transit » sont en transit (F_out)
    # et que plus de ~60 % des frames de la fenêtre sont en transit (F_in)
    wf = out["window_frac"]
    contamination_out = max(dur10 - wf, 0.0) / (1.0 - wf)
    verdict = "F_out contaminé : transit trop long" if contamination_out > 0.4 else \
              "F_in dilué : transit trop court" if dur50 < 0.6 * wf else "fenêtre acceptable"
    out.update(detected=True, dur50=dur50, dur10=dur10, center_offset=((s50 + e50) / 2 - n / 2) / n,
               s50=s50, e50=e50, s10=s10, e10=e10, a_in=a_in, b_in=b_in, m_out=m_out, has10=bool(has10),
               depth_adapt=float(depth_adapt), verdict_fixed=verdict)
    return out


# --------------------------------------------------------------------------- briques d'extraction

def oot_mask(n: int, st: dict) -> np.ndarray:
    """Frames hors transit (marge incluse). Repli : 10 % de chaque bord si le transit n'est pas détecté
    ou si trop peu de frames restent."""
    mask = np.zeros(n, dtype=bool)
    if st.get("detected"):
        mask[: max(st["s10"] - st["m_out"], 0)] = True
        mask[min(st["e10"] + st["m_out"] + 1, n):] = True
    if mask.sum() < max(10, int(0.05 * n)):
        k = max(int(0.1 * n), 5)
        mask[:] = False
        mask[:k] = True
        mask[-k:] = True
    return mask


def detrended(F: np.ndarray, mask: np.ndarray, deg: int) -> np.ndarray:
    """F / ligne de base, où la ligne de base est un polynôme de degré `deg` (en temps) ajusté, canal par
    canal, sur les frames `mask`. Hors transit, le résultat vaut ≈ 1."""
    F = np.asarray(F, dtype=float)
    med = np.median(F[mask], axis=0)
    Fn = F / med
    t = np.linspace(-1.0, 1.0, F.shape[0])
    V = np.vander(t, deg + 1, increasing=True)
    coef, *_ = np.linalg.lstsq(V[mask], Fn[mask], rcond=None)
    return Fn / (V @ coef)


def depth_fixed_window(F: np.ndarray) -> np.ndarray:
    """Reproduit calcul_Rp_median (fenêtre ±10 % autour du milieu), en profondeur : max(1 − F_in/F_out, 0)."""
    n = F.shape[0]
    mid, win = n // 2, max(n // 10, 1)
    F_in = np.median(F[mid - win: mid + win], axis=0)
    F_out = np.median(np.concatenate([F[: mid - win], F[mid + win:]]), axis=0)
    return np.maximum(1.0 - F_in / F_out, 0.0)


def depth_window_median(F: np.ndarray, st: dict, deg: int | None = None) -> np.ndarray:
    """Médianes dans le fond du transit détecté / hors transit ; deg=None : sans ligne de base."""
    mask = oot_mask(F.shape[0], st)
    sl = slice(st["a_in"], st["b_in"] + 1)
    if deg is None:
        return 1.0 - np.median(F[sl], axis=0) / np.median(F[mask], axis=0)
    return 1.0 - np.median(detrended(F, mask, deg)[sl], axis=0)


def transit_template(white: np.ndarray, st: dict, deg: int) -> np.ndarray:
    """Forme du transit (0 hors transit, 1 au fond) mesurée sur la courbe blanche."""
    n = len(white)
    mask = oot_mask(n, st)
    x = 1.0 - detrended(np.asarray(white, dtype=float)[:, None], mask, deg)[:, 0]
    xs = _smooth(x, max(5, n // 100))
    bottom = float(np.median(xs[st["a_in"]: st["b_in"] + 1]))
    T = np.clip(xs / max(bottom, 1e-12), 0.0, 1.3)
    T[mask] = 0.0
    return T


def depth_template(F: np.ndarray, white: np.ndarray, st: dict, deg: int) -> np.ndarray:
    """d_j = Σ_t T_t·(1 − F_jt/b_jt) / Σ_t T_t² : profondeur du fond du transit, toutes frames pondérées."""
    T = transit_template(white, st, deg)
    Y = 1.0 - detrended(F, oot_mask(F.shape[0], st), deg)
    return (T[:, None] * Y).sum(axis=0) / max(float((T ** 2).sum()), 1e-12)


# --------------------------------------------------------------------------- point d'entrée

def extract_depths(airs: np.ndarray, fgs: np.ndarray, method: str = "gabarit", deg: int = 1) -> np.ndarray:
    """Profondeurs [FGS1, AIRS_1..AIRS_n] (n_wl + 1,). airs : (n_time, n_wl), fgs : (n_time_fgs,).

    Chaque instrument a sa propre courbe blanche et sa propre fenêtre. Si le transit n'est pas détecté,
    repli sur la fenêtre fixe."""
    if method not in METHODS:
        raise ValueError(f"méthode inconnue : {method} (attendu : {METHODS})")
    parts = []
    for F in (np.asarray(fgs, dtype=float)[:, None], np.asarray(airs, dtype=float)):
        white = F.sum(axis=1)
        st = transit_window(white)
        if method == "fixe" or not st["detected"]:
            d = depth_fixed_window(F)
        elif method == "adaptee":
            d = depth_window_median(F, st, None)
        elif method == "adaptee_detrend":
            d = depth_window_median(F, st, deg)
        else:
            d = depth_template(F, white, st, deg)
        parts.append(d)
    return np.concatenate(parts)
