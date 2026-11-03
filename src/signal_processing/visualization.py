"""
visualization.py — Visualisation des spectres Ariel Data Challenge 2025
========================================================================
Responsabilité : afficher les résultats du traitement du signal.
Séparé des autres modules pour que l'import de Plotly soit optionnel
(Plotly n'est pas nécessaire sur un serveur d'entraînement headless).
"""

# === Scientific stack
import numpy as np

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def affichage_Rp(
    wl: np.ndarray,
    Rp: np.ndarray,
    planet_id: int | str,
    title: str = None,
) -> None:
    """
    Affiche le spectre de transmission d'une exoplanète (Rp/Rs vs longueur d'onde).

    Parameters
    ----------
    wl        : ndarray (n_wl,) — longueurs d'onde en microns
    Rp        : ndarray (n_wl,) — rayon relatif Rp/Rs
    planet_id : identifiant de la planète
    title     : titre personnalisé (optionnel)
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly n'est pas installé. Lancez : pip install plotly"
        )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wl,
        y=Rp,
        mode="lines+markers",
        marker=dict(size=4),
        name="Rp/Rs mesuré",
    ))

    fig.update_layout(
        title=title or f"Spectre de transmission — planète {planet_id}",
        xaxis_title="Longueur d'onde (μm)",
        yaxis_title="Rp/Rs",
        template="plotly_white",
        width=900,
        height=500,
    )
    fig.show()


def affichage_light_curve(
    flux: np.ndarray,
    planet_id: int | str,
    wl_index: int = 0,
    title: str = None,
) -> None:
    """
    Affiche la courbe de lumière d'une planète pour une longueur d'onde donnée.

    Parameters
    ----------
    flux      : ndarray (n_wl, n_time) — flux par longueur d'onde et par temps
    planet_id : identifiant de la planète
    wl_index  : index de la longueur d'onde à afficher (défaut : 0)
    title     : titre personnalisé (optionnel)
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly n'est pas installé. Lancez : pip install plotly"
        )

    lc = flux[wl_index, :]
    t  = np.arange(len(lc))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=lc,
        mode="lines",
        name=f"λ index {wl_index}",
    ))

    fig.update_layout(
        title=title or f"Courbe de lumière — planète {planet_id} (λ index {wl_index})",
        xaxis_title="Frame (temps binné)",
        yaxis_title="Flux (unités instrumentales)",
        template="plotly_white",
        width=900,
        height=400,
    )
    fig.show()
