from kaggle.api.kaggle_api_extended import KaggleApi
import os
import time

# Initialisation
api = KaggleApi()
api.authenticate()  # Utilise kaggle.json

# Liste des fichiers à télécharger (à la racine)
root_files = [
    "adc_info.csv",
    "axis_info.parquet",
    "sample_submission.csv",
    "test_star_info.csv",
    "train.csv",
    "train_star_info.csv",
    "wavelengths.csv"
]

# Dossier de destination
LOCAL_ROOT = "data/raw/"
os.makedirs(LOCAL_ROOT, exist_ok=True)

# Télécharger chaque fichier
for file in root_files:
    try:
        print(f"Téléchargement de {file}...")
        # ✅ Utilise competition_download_file (singulier) pour un fichier spécifique
        api.competition_download_file(
            "ariel-data-challenge-2025",
            file,  # Nom du fichier à télécharger
            LOCAL_ROOT  # Dossier de destination
        )
        print(f"✅ {file} téléchargé avec succès !")
        time.sleep(3)  # Pause pour éviter l'erreur 429
    except Exception as e:
        print(f"❌ Erreur pour {file} : {e}")
        time.sleep(10)