import pandas as pd

def get_downloaded_fluxnet_sites(csv_path: str) -> list:
    """
    Charge le fichier CSV de suivi et retourne la liste des sites dont le statut est 'downloaded'.
    """
    try:
        df = pd.read_csv(csv_path)
        downloaded_sites = df[df["status"] == "downloaded"]["site_id"].tolist()
        print(f"✅ {len(downloaded_sites)} sites marqués comme 'downloaded'.")
        return downloaded_sites
    except Exception as e:
        print(f"❌ Erreur lors du chargement de {csv_path} : {e}")
        return []
