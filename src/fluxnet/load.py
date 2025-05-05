import pandas as pd

def load_fluxnet_data(file_path):
    """Charge les données Fluxnet depuis un fichier CSV."""
    try:
        # Charger le fichier CSV
        df = pd.read_csv(file_path)
        print(f"🔄 Données Fluxnet chargées depuis {file_path}")
        return df
    except Exception as e:
        print(f"❌ Erreur lors du chargement du fichier {file_path}: {e}")
        return None