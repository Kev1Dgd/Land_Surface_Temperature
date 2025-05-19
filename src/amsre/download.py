from earthaccess import login, search_data, download
import os
import time

def authenticate():
    """Authenticates the user to Earthdata."""
    print("🔐 Authentification Earthdata...")
    return login()

from earthaccess import login, search_data, download
import os

def authenticate():
    """Authenticates the user to Earthdata."""
    print("🔐 Authentification Earthdata...")
    return login()

def download_amsre_ae_l2a(date="2005-07-01", output_dir="data/raw/amsre"):
    """Download AE_L2A files for a given date within a bounding box"""
    print(f"\n🔍 Search AMSR-E AE_L2A data for {date}")

    results = search_data(
        short_name="AE_L2A",
        temporal=(date, date),
        cloud_hosted=True
    )

    os.makedirs(output_dir, exist_ok=True)

    max_retries = 3
    delay = 5  # secondes d'attente entre les tentatives

    for attempt in range(1, max_retries + 1):
        try:
            print(f"📥 Tentative n°{attempt} de téléchargement...")
            files = download(results, local_path=output_dir)
            print(f"✅ {len(files)} fichiers téléchargés dans {output_dir}")
            return files
        except Exception as e:
            print(f"⚠️ Erreur lors du téléchargement : {e}")
            if attempt < max_retries:
                print(f"🔁 Nouvelle tentative dans {delay} secondes...")
                time.sleep(delay)
            else:
                print("❌ Échec après 3 tentatives. Abandon.")
                raise