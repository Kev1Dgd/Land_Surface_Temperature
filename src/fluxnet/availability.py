import pandas as pd
import os

def load_fluxnet_availability(data_dir="data/processed/fluxnet"):
    """Charge le fichier de disponibilité des données FLUXNET."""
    for file in os.listdir(data_dir):
        if file.startswith("data-availability") and file.endswith(".csv"):
            file_path = os.path.join(data_dir, file)
            try:
                df = pd.read_csv(file_path)
                print(f"🔄 Données de disponibilité chargées depuis {file_path}")
                return df
            except Exception as e:
                print(f"❌ Erreur lors du chargement de {file}: {e}")
    print("⚠️ Aucun fichier de disponibilité FLUXNET trouvé.")
    return None
