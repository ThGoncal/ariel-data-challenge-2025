# NeurIPS — Ariel Data Challenge 2025 · pipeline MLOps reproductible

Extraction du spectre de transmission d'exoplanètes (mission ESA Ariel, données simulées) et
prédiction **μ ± σ** de Rp/Rs sur 283 longueurs d'onde, avec une chaîne MLOps complète.
Métrique : Gaussian Log-Likelihood normalisée (0 = baseline naïve, 1 = oracle).

**Version courante : v0.2.0** — extraction à fenêtre adaptative, modèle champion enregistré dans MLflow,
pipeline reproductible avec DVC, inférence sur le jeu de test.

## Résultat

| | v0.1.0 | v0.2.0 |
|---|---|---|
| Score GLL, validation fixe (20 planètes) | 0,19 | **0,380** (intervalle à 90 % : 0,330 – 0,420) |
| Score GLL, validation croisée (5 plis × 3 répétitions, 160 planètes) | 0,158 | **0,337** (écart-type entre plis : 0,098) |

Le score a presque doublé **sans changer de modèle** : le gain vient entièrement de la qualité du spectre extrait
(voir « Ce qui a fait progresser le score »). Le modèle reste un Ridge (α = 10) sur Rp² et les paramètres
stellaires/orbitaux, avec un σ constant par longueur d'onde.

### Combien de planètes faut-il ?

Courbe d'apprentissage (`scripts/learning_curve.py`, validation croisée, mêmes plis de test pour toutes les tailles) :

| Planètes d'entraînement | 20 | 40 | 60 | 80 | 100 | 128 |
|---|---|---|---|---|---|---|
| Score moyen | 0,253 | 0,297 | 0,317 | 0,323 | 0,329 | 0,337 |

Chaque doublement des données rapporte environ moitié moins que le précédent ; le dernier palier (+0,008 ± 0,008)
n'est pas distinguable de zéro. Avec ce modèle, le nombre de planètes n'est plus le facteur principal
(160 sur 1100 sont utilisées). Le `alpha` est gardé fixe pour toutes les tailles, ce qui sous-estime un peu le
gain possible avec plus de données.

![Score GLL vs nombre de planètes (boucle incrémentale v0.1.0)](docs/score_curve.png)

## Ce qui a fait progresser le score

1. **Fenêtre de transit adaptée à chaque planète.** La fenêtre fixe (±10 % autour du transit) était
   contaminée pour les transits longs et **sous-estimait la profondeur de 10 à 24 %**. La fenêtre est désormais
   détectée sur la courbe de lumière blanche ; la dispersion du niveau d'une planète à l'autre passe de 0,100 à
   0,018–0,026 (`src/signal_processing/transit_fit.py`).
2. **Canaux de bord.** Les ~10 premiers canaux AIRS ont un flux très faible (0,2 à 4,5 % de la médiane) :
   leur profondeur est dominée par le bruit. Ils sont remplacés par la moyenne des 10 canaux suivants.
3. **Profondeur plutôt que Rp comme variable d'entrée** (Rp²), et un σ calculé sur les résidus d'entraînement.
4. **Une seule définition de l'extraction** (`extract_spectrum`) pour l'entraînement et l'inférence : pas de
   décalage entre ce qui est appris et ce qui est servi.

Les comparaisons sont faites en **validation croisée appariée** (mêmes plis pour toutes les configurations) ;
les tableaux et figures sont dans `reports/extraction/`.

## Résultats négatifs et mises en garde

- **La validation fixe de 20 planètes peut tromper.** Un gabarit de transit avec ligne de base parabolique
  paraissait meilleur sur ces 20 planètes mais ne l'était pas en validation croisée appariée : il n'est pas retenu.
- **Le plateau de la v0.1.0 n'était pas une saturation des données.** La boucle incrémentale s'était arrêtée à
  140 planètes parce que le score stagnait ; la cause principale était la qualité du spectre (fenêtre fixe), pas la
  quantité de données — corriger l'extraction a doublé le score à nombre de planètes comparable.
- **Cause des canaux de bord non établie.** Une sur-soustraction du fond est une hypothèse, non prouvée : le
  remède (remplacement) est validé, l'explication ne l'est pas.
- **Réglage sur les mêmes planètes.** Les paramètres d'extraction ont été choisis sur ces 160 planètes : le gain en
  validation croisée est apparié, mais n'est pas indépendant de ce choix.

## Limites connues

- Seulement **160 planètes sur 1100** (~15 %) utilisées pour l'entraînement.
- Quelques planètes aberrantes dominent encore certains plis (scores de 0,125 à 0,436 selon le pli).
- Modèle de base : Ridge, σ constant. Le MLP hétéroscédastique (`src/models/model.py`) n'a pas été
  réévalué avec la nouvelle extraction.
- Le jeu de test public se réduit à **une planète sans labels** : aucun score n'est calculable en local ; le
  contrôle de la chaîne d'inférence se fait sur des planètes d'entraînement (voir plus bas).
- La boucle incrémentale pilotée par le score (v0.1.0, `src/pipelines/incremental.py`, 17 tests) reste dans le
  dépôt mais n'est **pas rebranchée** sur l'extraction v0.2.0. Le point d'arrêt est aujourd'hui mesuré a posteriori
  par la courbe d'apprentissage. L'orchestration Airflow (`airflow/dags/ariel_incremental.py`) et le
  `docker-compose.yml` n'ont pas été validés de bout en bout.
- Les tests hérités du gabarit de départ (`tests/unit/test_train.py`, `tests/integration/`) ne sont pas à jour et
  sont ignorés par la CI.

## Le problème d'ingénierie

Le dataset pèse **~264 Go** (1100 planètes × ~240 Mo) : il ne tient pas sur un poste de travail, et à ~1,5 Mo/s
le télécharger entièrement prendrait environ **40 h**. Le projet a donc été construit autour de cette contrainte :

```
brut (par planète) ─► traitement du signal (Numba) ─► courbes de lumière sauvegardées (DVC, quelques Mo)
courbes ─► extraction du spectre (params.yaml) ─► spectres (DVC) ─► modèle champion ─► MLflow
```

La v0.1.0 a d'abord appris de façon **incrémentale** (télécharger, traiter, supprimer le brut, réentraîner,
mesurer sur une validation fixe, s'arrêter quand le score stagne). La v0.2.0 conserve le principe « ne garder que
les courbes » : le brut est traité une fois, puis l'extraction peut être refaite à volonté, hors ligne, sur des
courbes de quelques Mo au lieu de 240 Mo.

## Pipeline reproductible (v0.2.0)

Deux étapes DVC (`dvc.yaml`), paramétrées par `params.yaml` :

| Étape | Rôle | Sortie |
|---|---|---|
| `spectra` | courbes sauvegardées → spectres Rp (méthode, degré, canaux de bord) selon le découpage figé `configs/split_v1.json` (20 validation / 140 entraînement) | `data/processed/spectra_v2` |
| `train_champion` | entraîne le Ridge, mesure sur la validation fixe (intervalle bootstrap) et en validation croisée, sauvegarde le modèle | `models/champion_v2`, `reports/champion/metrics.json` |

Le modèle est enregistré dans le registre MLflow (DagsHub) sous `ariel-ridge-adaptive-depth`, alias `champion`.
Chaque version porte l'empreinte des courbes utilisées (`curves_sha256`) et les paramètres d'extraction.

## Inférence sur le jeu de test

`src/pipelines/predict_test.py` : brut d'une planète → même traitement et même extraction que l'entraînement →
modèle champion → fichier au format `sample_submission.csv`, vérifié (colonnes, identifiants, valeurs ≥ 0,
écarts-types > 0).

Comme le test n'a pas de labels, le script propose un **contrôle de cohérence** (`--check`) : il refait la chaîne
depuis le brut sur des planètes d'entraînement et vérifie qu'elle redonne les spectres et les prédictions de
l'entraînement. Résultat sur 3 planètes (2 de validation, 1 d'entraînement) : écart relatif des spectres nul,
des prédictions < 1e-6. Le script refuse de tourner si l'extraction de `params.yaml` diffère de celle du modèle.

## Traitement du signal

CDS (end − start), correction ADC (`/= gain`, `+= offset`), dark, flat, pixels morts, hot pixels
(σ-clipping MAD), non-linéarité (polynôme de Horner), binning temporel (Numba `@njit`, `prange` + joblib), puis
courbes de lumière sommées sur l'espace. L'extraction de la profondeur par canal utilise la fenêtre adaptative
décrite plus haut. Code dans `src/signal_processing/`.

## Stack

| Outil | Rôle |
|---|---|
| uv, pytest, ruff | environnement, tests unitaires (205), lint |
| DVC / DagsHub | courbes de lumière, spectres, modèles et métriques versionnés ; étapes reproductibles |
| MLflow (DagsHub) | suivi des expériences, registre de modèles (alias `champion`) |
| Kaggle API | téléchargement par planète (v0.1.0) |
| Airflow, Docker / compose | orchestration et conteneurs (non validés de bout en bout) |
| GitHub Actions | CI : lint et tests unitaires (`.github/workflows/ci.yml`) |

## Démarrage

```bash
uv sync
uv run python -m pytest tests/unit -q -m "not slow" --deselect tests/unit/test_train.py

# reproduire le modèle champion (nécessite l'accès DVC/DagsHub)
dvc pull
dvc repro --single-item spectra
dvc repro --single-item train_champion

# inférence : contrôle de cohérence, puis soumission
uv run python -m src.pipelines.predict_test --check --raw-root <dossier brut> --folder train --ids <id> <id>
uv run python -m src.pipelines.predict_test --raw-root <dossier brut> --folder test

# courbe d'apprentissage
uv run python -m scripts.learning_curve
```

Identifiants : `~/.kaggle/kaggle.json` ou variables `KAGGLE_USERNAME` / `KAGGLE_KEY`, et identifiants DagsHub
(`.env`, non versionné).

`dvc repro` sans argument relancerait aussi la boucle de téléchargement de la v0.1.0 : utiliser `--single-item`.

## Structure

```
src/data/ (collecte Kaggle) · src/signal_processing/ (calibration, extraction : transit_fit, spectra)
src/ml/ (features, métrique) · src/models/ (champion, MLP, baselines)
src/pipelines/ (build_spectra, train_champion, predict_test, incremental)
scripts/ (études d'extraction, retraitement, courbe d'apprentissage) · reports/ · configs/ · airflow/ · docker/ · tests/
```

## Prochaines étapes

- Retraiter les 1100 planètes depuis les données brutes locales (~1 min par planète) et réévaluer, notamment avec
  un modèle plus flexible (MLP, σ variable) qui ne tient pas avec 140 planètes.
- Diagnostiquer les planètes aberrantes.
- Rebrancher la boucle pilotée par le score sur l'extraction v0.2.0.
- Corriger à la source les noms de colonnes d'incertitude du modèle enregistré (`sigma_i` attendu par la
  soumission officielle, `wl_i_std` actuellement).
- Nettoyer les tests hérités du gabarit et valider Docker Compose et Airflow de bout en bout.
