# from kaggle.api.kaggle_api_extended import KaggleApi
#
# api = KaggleApi()
# api.authenticate()
#
# # Méthode pour l'API 2.0+
# files = api.competition_list_files("ariel-data-challenge-2025")
# print(f"Fichiers disponibles : {files}")
#
# # Initialisation
# all_files = []
# next_page_token = None
# page = 0
#
# print("Récupération de la liste complète des fichiers...")
#
# while True:
#     page += 1
#     print(f"Récupération de la page {page}...")
#
#     # Appel à l'API avec pagination
#     result = api.competition_list_files(
#         "ariel-data-challenge-2025",
#         page_token=next_page_token  # Passe None pour la première page
#     )
#
#     # Ajouter les fichiers de cette page
#     all_files.extend(result.files)
#     print(f"  + {len(result.files)} fichiers récupérés (total: {len(all_files)})")
#
#     # Vérifier s'il y a une page suivante
#     next_page_token = result.next_page_token
#     if not next_page_token:
#         print("✅ Liste complète récupérée !")
#         break
#
# # Afficher le nombre total de fichiers
# print(f"\nNombre total de fichiers : {len(all_files)}")
#
# # Afficher les 10 premiers fichiers
# print("\n10 premiers fichiers :")
# for file in all_files[:10]:
#     print(f"  - {file.name}")

from kaggle.api.kaggle_api_extended import KaggleApi
import time

api = KaggleApi()
api.authenticate()

# Initialisation
all_files = []
page = 1
has_more = True

print("Récupération de la liste complète des fichiers...")

while has_more:
    print(f"Récupération de la page {page}...")

    try:
        # ✅ Utilise competition_list_files avec le paramètre page
        result = api.competition_list_files(
            "ariel-data-challenge-2025",
            page=page  # ✅ Pagination via page (pas page_token)
        )

        # Vérifier si result est une liste ou un objet
        if hasattr(result, 'files'):  # Si c'est un objet ApiListDataFilesResponse
            files = result.files
            has_more = len(files) > 0  # On suppose qu'il n'y a pas de next_page_token
        else:  # Si c'est une liste directe
            files = result
            has_more = len(files) > 0

        all_files.extend(files)
        print(f"  + {len(files)} fichiers récupérés (total: {len(all_files)})")

        # Passer à la page suivante
        page += 1

        # ✅ Pause pour éviter l'erreur 429
        time.sleep(3)

    except Exception as e:
        print(f"❌ Erreur sur la page {page} : {e}")
        has_more = False  # Arrêter en cas d'erreur

# Afficher le nombre total de fichiers
print(f"\nNombre total de fichiers : {len(all_files)}")

# Afficher les 10 premiers fichiers
print("\n10 premiers fichiers :")
for file in all_files[:10]:
    # Si file est un objet, utiliser file.name, sinon file est déjà un nom
    file_name = file.name if hasattr(file, 'name') else file
    print(f"  - {file_name}")