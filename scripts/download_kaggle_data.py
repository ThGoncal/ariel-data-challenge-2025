"""Téléchargement des données brutes du concours Ariel 2025 avec Selenium (Firefox).

Pourquoi : 264 Go ne tiennent pas sur le disque du PC, et l'API Kaggle renvoyait des erreurs 429.
Le script se connecte à Kaggle, parcourt l'arborescence « Data Explorer » et télécharge les .parquet
planète par planète. Reprenable grâce au journal downloaded_files.log.
Configuration : .env (KAGGLE_USERNAME = e-mail du compte, KAGGLE_PASSWORD) et ARIEL_DATA_DIR.
Fragile par nature : dépend des sélecteurs XPath de l'interface web de Kaggle.
"""
import os
import time
import shutil
import pandas as pd
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from dotenv import load_dotenv
from tqdm import tqdm

# --- CONFIGURATION GLOBALE ---
load_dotenv()
KAGGLE_USER = os.getenv("KAGGLE_USERNAME")
KAGGLE_PASS = os.getenv("KAGGLE_PASSWORD")
KAGGLE_DATA_URL = "https://www.kaggle.com/competitions/ariel-data-challenge-2025/data"

# Dossier de destination (disque externe) : variable ARIEL_DATA_DIR, sinon ./ariel_data_2025
DATA_DIR = os.getenv("ARIEL_DATA_DIR", "ariel_data_2025")
BASE_DOWNLOAD_DIR = os.path.normpath(os.path.join(DATA_DIR, "downloads_temp"))
FINAL_ROOT_DOWNLOAD_DIR = os.path.normpath(os.path.join(DATA_DIR, "downloads"))
LOG_FILE = os.path.join(DATA_DIR, "downloaded_files.log")

os.makedirs(DATA_DIR, exist_ok=True)
# Prépare les répertoires de téléchargement si nécessaire
os.makedirs(BASE_DOWNLOAD_DIR, exist_ok=True)
os.makedirs(FINAL_ROOT_DOWNLOAD_DIR, exist_ok=True)

DOWNLOAD_DIR_FOR_PREF = os.path.abspath(BASE_DOWNLOAD_DIR)

# --- FONCTIONS UTILITAIRES POUR LE LOG ET LE TÉLÉCHARGEMENT ---

def load_downloaded_files():
    """Charge les chemins des fichiers déjà téléchargés depuis le journal."""
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f)
    return set()


def save_downloaded_file(file_path):
    """Ajoute le chemin d'un fichier téléchargé au journal."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(file_path + "\n")


def wait_for_download_and_move(expected_file_name, source_dir, target_dir, timeout=240):
    """
    Attend qu'un fichier soit complètement téléchargé, puis le déplace.
    """
    full_source_path = os.path.join(source_dir, expected_file_name)
    start_time = time.time()

    while True:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Le fichier '{expected_file_name}' n'a pas été téléchargé dans les {timeout} secondes.")

        if os.path.exists(full_source_path) and os.path.getsize(full_source_path) > 0:
            initial_size = os.path.getsize(full_source_path)
            time.sleep(2)  # Pause pour s'assurer que la taille est stable
            current_size = os.path.getsize(full_source_path)
            if current_size == initial_size:
                break
        time.sleep(1)
    os.makedirs(target_dir, exist_ok=True)
    full_target_path = os.path.join(target_dir, expected_file_name)
    shutil.move(full_source_path, full_target_path)


# --- FONCTIONS GÉNÉRIQUES DE TÉLÉCHARGEMENT AVEC TQDM ---

def download_files_at_level(driver, wait, downloaded_files_log, full_relative_folder_path, parent_xpath,
                            base_target_path, dataset_type):
    """
    Détecte et télécharge tous les fichiers .parquet visibles sous un chemin XPath donné.
    Utilise tqdm pour afficher la progression des téléchargements de fichiers.
    """
    files_xpath = f"{parent_xpath}/../following-sibling::ul//p[contains(text(), '.parquet')]"

    try:
        wait.until(EC.presence_of_all_elements_located((By.XPATH, files_xpath)))
        file_elements = driver.find_elements(By.XPATH, files_xpath)
    except Exception:
        return  # Aucun fichier trouvé

    for file_element in tqdm(file_elements, desc=f"  DL {full_relative_folder_path}", unit="f", leave=False):
        file_name = file_element.text.strip()
        log_entry = os.path.join(dataset_type, full_relative_folder_path, file_name)

        if log_entry in downloaded_files_log:
            continue

        try:
            driver.execute_script("arguments[0].click();", file_element)

            xpath_download = "//span[@aria-label='Download']"
            download_button = wait.until(EC.element_to_be_clickable((By.XPATH, xpath_download)))
            driver.execute_script("arguments[0].click();", download_button)

            wait_for_download_and_move(file_name, BASE_DOWNLOAD_DIR, base_target_path)
            save_downloaded_file(log_entry)

        except Exception as e:
            tqdm.write(f"      ❌ Échec du téléchargement de {file_name} : {e}")


def process_subfolders(driver, wait, downloaded_files_log, full_relative_parent_path, parent_xpath, dataset_type):
    """
    Détecte et traite les sous-dossiers sous un chemin XPath donné.
    Utilise tqdm pour afficher la progression des sous-dossiers.
    """
    # Remonter au parent de la flèche, puis chercher le UL frère
    subfolder_elements_xpath = (
        f"{parent_xpath}/../following-sibling::ul//div[@data-testid='tree-item-arrow-right' and .//span[contains(text(), 'arrow_right')]]")

    try:
        subfolder_elements = driver.find_elements(By.XPATH, subfolder_elements_xpath)
    except Exception:
        return  # Pas de sous-dossiers

    for sub_folder_element in tqdm(subfolder_elements, desc=f"  Proc. subfolders in {full_relative_parent_path}",
                                   unit="folder", leave=False):
        sub_folder_name = sub_folder_element.get_attribute("title").replace("Expand ", "")

        current_subfolder_full_relative_path = os.path.join(full_relative_parent_path, sub_folder_name)
        target_subfolder_path = os.path.join(FINAL_ROOT_DOWNLOAD_DIR, dataset_type, current_subfolder_full_relative_path)
        os.makedirs(target_subfolder_path, exist_ok=True)

        try:
            driver.execute_script("arguments[0].click();", sub_folder_element)
            time.sleep(3)

            subfolder_xpath = f"//div[@title='Expand {sub_folder_name}']"
            download_files_at_level(driver, wait, downloaded_files_log, current_subfolder_full_relative_path,
                                    subfolder_xpath, target_subfolder_path, dataset_type)

            collapse_folder_xpath = (
                f"//div[@data-testid='tree-item-arrow-right' "
                f"and @title='Expand {sub_folder_name}' "
                f"and .//span[contains(text(), 'arrow_drop_down')]]"
            )
            collapse_button = wait.until(EC.element_to_be_clickable((By.XPATH, collapse_folder_xpath)))
            driver.execute_script("arguments[0].click();", collapse_button)
            time.sleep(1)
        except Exception as e_folder:
            tqdm.write(f"    ❌ Échec du traitement du sous-dossier '{sub_folder_name}' : {e_folder}")


# --- FONCTION PRINCIPALE D'EXÉCUTION ---
def main():
    # Initialisation du WebDriver
    service = Service()
    firefox_options = Options()
    firefox_options.set_preference("browser.download.dir", DOWNLOAD_DIR_FOR_PREF)
    firefox_options.set_preference("browser.download.folderList", 2)
    firefox_options.set_preference("browser.helperApps.neverAsk.saveToDisk",
                                   "application/octet-stream,application/x-parquet,application/x-download")
    firefox_options.add_argument("--headless")  # Exécuter en mode headless

    driver = None  # Initialiser driver à None
    try:
        driver = webdriver.Firefox(service=service, options=firefox_options)
        driver.get(KAGGLE_DATA_URL)

        wait = WebDriverWait(driver, 30)  # Wait générique pour les éléments UI

        # --- PROCESSUS DE CONNEXION KAGGLE ---
        print("\n--- Début du processus de connexion à Kaggle ---")
        wait_for_login_elements = WebDriverWait(driver, 20)  # Attente spécifique pour les éléments de connexion

        # 1. OK, Got it. (premier pop-up de consentement aux cookies)
        try:
            ok_got_it_xpath_1 = "//div[contains(text(), 'OK, Got it.')]"
            ok_got_it_button_1 = wait_for_login_elements.until(
                EC.element_to_be_clickable((By.XPATH, ok_got_it_xpath_1)))
            driver.execute_script("arguments[0].click();", ok_got_it_button_1)
            print("  Clic sur 'OK, Got it.' (cookies)")
        except Exception as e:
            print(
                f"  Avertissement: Impossible de cliquer sur le premier 'OK, Got it.' (peut être déjà fermé ou absent): {e}")

        # 2. Sign In (bouton principal sur la page Kaggle)
        sign_in_button_xpath_main = "//span[contains(text(), 'Sign In')]"
        sign_in_button_main = wait_for_login_elements.until(
            EC.element_to_be_clickable((By.XPATH, sign_in_button_xpath_main)))
        driver.execute_script("arguments[0].click();", sign_in_button_main)
        print("  Clic sur 'Sign In' principal")

        # 3. Sign In with Email (sur le pop-up de connexion)
        sign_in_email_xpath = "//span[contains(text(), 'Sign in with Email')]"
        sign_in_email_button = wait_for_login_elements.until(
            EC.element_to_be_clickable((By.XPATH, sign_in_email_xpath)))
        driver.execute_script("arguments[0].click();", sign_in_email_button)
        print("  Clic sur 'Sign in with Email'")

        # 4. Saisir l'adresse e-mail
        email_field = wait_for_login_elements.until(EC.visibility_of_element_located((By.NAME, "email")))
        email_field.send_keys(KAGGLE_USER)
        print("  Adresse e-mail saisie.")

        # 5. Saisir le mot de passe
        password_field = wait_for_login_elements.until(EC.visibility_of_element_located((By.NAME, "password")))
        password_field.send_keys(KAGGLE_PASS)
        print("  Mot de passe saisi.")

        # 6. Sign In (bouton final du formulaire de connexion)
        sign_in_button_form_xpath = "//button[@type='submit']/span[contains(text(), 'Sign In')]"
        sign_in_button_form = wait_for_login_elements.until(
            EC.element_to_be_clickable((By.XPATH, sign_in_button_form_xpath)))
        driver.execute_script("arguments[0].click();", sign_in_button_form)
        print("  Clic sur 'Sign In' du formulaire.")

        # 7. OK, Got it. (deuxième pop-up après connexion, si présent)
        try:
            ok_got_it_xpath_2 = "//span[contains(text(), 'OK, Got it.')]"
            ok_got_it_button_2 = wait_for_login_elements.until(
                EC.element_to_be_clickable((By.XPATH, ok_got_it_xpath_2)))
            driver.execute_script("arguments[0].click();", ok_got_it_button_2)
            print("  Clic sur 'OK, Got it.' (post-connexion)")
        except Exception as e:
            print(
                f"  Avertissement: Impossible de cliquer sur le deuxième 'OK, Got it.' (peut être déjà fermé ou absent): {e}")

        print("✅ Connecté à Kaggle")
        time.sleep(5)  # Laisser le temps à la page de se charger complètement après la connexion

        # --- Chargement du journal des fichiers déjà téléchargés ---
        downloaded_files_log = load_downloaded_files()
        print(f"Chargé {len(downloaded_files_log)} entrées depuis le journal des téléchargements.")

        # --- TÉLÉCHARGEMENT DES FICHIERS À LA RACINE DU DATA EXPLORER ---
        print("\n--- Début du téléchargement des fichiers à la racine ---")
        root_file_list = ['adc_info.csv', 'axis_info.parquet', 'sample_submission.csv', 'test_star_info.csv',
                          'train.csv', 'train_star_info.csv', 'wavelengths.csv']

        target_root_download_path = os.path.join(FINAL_ROOT_DOWNLOAD_DIR)
        os.makedirs(target_root_download_path, exist_ok=True)

        for file_name in tqdm(root_file_list, desc="  DL root files", unit="file", leave=True):
            log_entry = file_name
            if log_entry in downloaded_files_log:
                continue
            try:
                file_to_download_xpath = f"//p[contains(text(), '{file_name}')]"
                file_to_download = wait.until(EC.element_to_be_clickable((By.XPATH, file_to_download_xpath)))
                driver.execute_script("arguments[0].click();", file_to_download)

                xpath_download = "//span[@aria-label='Download']"
                download_button = wait.until(EC.element_to_be_clickable((By.XPATH, xpath_download)))
                driver.execute_script("arguments[0].click();", download_button)

                wait_for_download_and_move(file_name, BASE_DOWNLOAD_DIR, target_root_download_path)
                save_downloaded_file(log_entry)

            except Exception as e:
                tqdm.write(f"  ❌ Échec du téléchargement/organisation de {file_name} : {e}")
                continue
        print("✅ Fichiers à la racine téléchargés.")

        # --- DÉPLIEMENT DES DOSSIERS PRINCIPAUX ('train', 'more...') ---
        print("\n--- Dépliement des dossiers principaux pour accès aux données ---")

        # 1. Déplier le dossier "train"
        expand_train_folder_xpath = (
            "//div[@data-testid='tree-item-arrow-right' "
            "and @title='Expand train' "
            "and .//span[contains(text(), 'arrow_right')]]"
        )
        expand_train_folder = wait.until(EC.element_to_be_clickable((By.XPATH, expand_train_folder_xpath)))
        driver.execute_script("arguments[0].click();", expand_train_folder)
        print("  ✅ Dossier 'train' déplié.")
        time.sleep(3)  # Laisser le temps à l'UI de charger

        # 2. Déplier tous les éléments " more ..." du dossier "train" pour charger tous les planet_id
        while True:
            try:
                more_button_xpath = "//span[contains(text(), 'more_horiz')]"
                more_button = wait.until(EC.element_to_be_clickable((By.XPATH, more_button_xpath)))
                driver.execute_script("arguments[0].click();", more_button)
                time.sleep(3)  # Laisser le temps aux nouveaux éléments de se charger
            except:  # Si le bouton 'more' n'est plus là (TimeoutException)
                print(
                    "  Tous les éléments 'more...' du dossier 'train' ont été chargés ou le bouton n'est pas présent.")
                break
                # NOUVEAU : Pause après le dépliement complet du dossier 'train'
        print("  ✅ Tous les éléments 'more...' du dossier 'train' dépliés.")
        print("  Attente de quelques secondes pour le chargement complet de l'arborescence...")
        time.sleep(5)  # Pause supplémentaire après le dépliement complet du dossier train

        # --- BOUCLE PRINCIPALE POUR LE DOSSIER 'TRAIN' ---
        # train_star_info.csv est maintenant téléchargé et lu
        file_path_train = os.path.join(FINAL_ROOT_DOWNLOAD_DIR, "train_star_info.csv")
        df_folders_train = pd.read_csv(file_path_train)
        folder_names_to_download_train = df_folders_train['planet_id'].astype(str).tolist()
        print(f"\nChargé {len(folder_names_to_download_train)} dossiers 'train' depuis train_star_info.csv.")

        # tqdm pour la boucle principale des dossiers TRAIN
        for planet_id in tqdm(folder_names_to_download_train, desc="Processing TRAIN folders", unit="folder"):
            # L'élément planet_id doit être trouvé sous le dossier 'train' déjà déplié

            expand_planet_id_folder_xpath = (
                f"//div[@data-testid='tree-item-arrow-right' "
                f"and @title='Expand {planet_id}' "
                f"and .//span[contains(text(), 'arrow_right')]]"
            )
            try:
                expand_planet_id_folder = wait.until(
                    EC.element_to_be_clickable((By.XPATH, expand_planet_id_folder_xpath)))
                driver.execute_script("arguments[0].click();", expand_planet_id_folder)
                time.sleep(3)  # Laisser le temps à l'UI de charger

                target_base_path = os.path.join(FINAL_ROOT_DOWNLOAD_DIR, 'train', planet_id)
                os.makedirs(target_base_path, exist_ok=True)

                # Chemin XPath du dossier planet_id déplié, relatif au conteneur principal 'train'
                planet_id_collapsed_xpath = f"//div[@title='Expand {planet_id}']"

                download_files_at_level(driver, wait, downloaded_files_log, planet_id, planet_id_collapsed_xpath,
                                        target_base_path, 'train')
                process_subfolders(driver, wait, downloaded_files_log, planet_id, planet_id_collapsed_xpath, 'train')

                # Replier le dossier planet_id
                collapse_folder_xpath_planet_id = (
                    f"//div[@data-testid='tree-item-arrow-right' "
                    f"and @title='Expand {planet_id}' "
                    f"and .//span[contains(text(), 'arrow_drop_down')]]"
                )
                collapse_button = wait.until(EC.element_to_be_clickable((By.XPATH, collapse_folder_xpath_planet_id)))
                driver.execute_script("arguments[0].click();", collapse_button)
                time.sleep(1)

            except Exception as e:
                tqdm.write(f"  ❌ Échec de dépliement/traitement du dossier principal '{planet_id}' : {e}")

        # --- TRAITEMENT DU DOSSIER 'TEST' ---
        test_folder_id = '1103775'
        print(f"\n--- Début du traitement du dossier principal TEST : {test_folder_id} ---")

        # S'assurer que le dossier parent 'test' est déplié
        expand_test_folder_parent_xpath = (
            "//div[@data-testid='tree-item-arrow-right' "
            "and @title='Expand test' "
            "and .//span[contains(text(), 'arrow_right')]]"
        )
        try:
            expand_test_folder_parent = wait.until(
                EC.element_to_be_clickable((By.XPATH, expand_test_folder_parent_xpath)))
            driver.execute_script("arguments[0].click();", expand_test_folder_parent)
            print("  ✅ Dossier parent 'test' déplié.")
            time.sleep(3)
        except Exception as e:
            print(f"  Avertissement: Impossible de déplier le dossier parent 'test' (peut être déjà déplié): {e}")

        # Déplier le dossier spécifique '1103775' sous 'test'

        expand_test_folder_xpath = (
            f"//div[@data-testid='tree-item-arrow-right' "
            f"and @title='Expand {test_folder_id}' "
            f"and .//span[contains(text(), 'arrow_right')]]"
        )
        try:
            expand_test_folder = wait.until(EC.element_to_be_clickable((By.XPATH, expand_test_folder_xpath)))
            driver.execute_script("arguments[0].click();", expand_test_folder)
            print(f"  ✅ Dossier principal 'TEST' ({test_folder_id}) déplié.")
            time.sleep(3)

            target_base_path_test = os.path.join(FINAL_ROOT_DOWNLOAD_DIR, 'test', test_folder_id)
            os.makedirs(target_base_path_test, exist_ok=True)

            #test_folder_collapsed_xpath = f"//div[@title='Collapse {test_folder_id}']"
            test_folder_collapsed_xpath = f"//div[@title='Expand {test_folder_id}']"

            download_files_at_level(driver, wait, downloaded_files_log, test_folder_id, test_folder_collapsed_xpath,
                                    target_base_path_test, 'test')
            process_subfolders(driver, wait, downloaded_files_log, test_folder_id, test_folder_collapsed_xpath, 'test')

            # Replier le dossier principal TEST
            collapse_folder_xpath_test_id = (
                f"//div[@data-testid='tree-item-arrow-right' "
                f"and @title='Expand {test_folder_id}' "
                f"and .//span[contains(text(), 'arrow_drop_down')]]"
            )
            collapse_button = wait.until(EC.element_to_be_clickable((By.XPATH, collapse_folder_xpath_test_id)))
            driver.execute_script("arguments[0].click();", collapse_button)
            print(f"  Dossier '{test_folder_id}' replié.")
            time.sleep(1)

        except Exception as e:
            print(f"  ❌ Échec du traitement du dossier principal 'TEST' ({test_folder_id}) : {e}")

        print("\n--- Script de téléchargement complet terminé ! ---")

    finally:
        if driver:
            driver.quit()  # S'assurer que le driver est toujours fermé


# Point d'entrée principal du programme
if __name__ == "__main__":
    main()