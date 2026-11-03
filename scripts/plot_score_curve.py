"""Trace la courbe « score GLL vs nombre de planètes » depuis state.json -> docs/score_curve.png."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--state", default="data/processed/incremental/state.json")
ap.add_argument("--out", default="docs/score_curve.png")
a = ap.parse_args()

st = json.loads(Path(a.state).read_text())
h = [r for r in st["history"] if r["score"] is not None]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot([r["n_train"] for r in h], [r["score"] for r in h], "o-", alpha=.5, label="score GLL (batch)")
ax.plot([r["n_train"] for r in h], [r["smoothed"] for r in h], "-", lw=2, label="score lissé")
ax.set_xlabel("planètes d'entraînement traitées")
ax.set_ylabel("score GLL normalisé (validation fixe)")
ax.set_title(f"Apprentissage incrémental — arrêt : {st['stop_reason'] or 'en cours'}", fontsize=9)
ax.grid(alpha=.3); ax.legend()
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout(); fig.savefig(a.out, dpi=150)
print("écrit", a.out)
