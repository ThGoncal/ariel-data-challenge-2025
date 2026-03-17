#!/bin/bash
set -e

echo "🚀 Setup MLOps Template"

# Python env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# Pre-commit hooks
pre-commit install

# DVC
dvc init --no-scm 2>/dev/null || true

# Dossiers nécessaires
mkdir -p artifacts mlruns

echo "✅ Setup terminé !"
echo "👉 Lance 'make stack-up' pour démarrer la stack"
