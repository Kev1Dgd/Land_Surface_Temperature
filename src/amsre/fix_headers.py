import os
from glob import glob

def fix_amsre_headers():
    # Chemin vers les fichiers AMSRE quotidiens
    amsre_folder = "data/processed/amsre"
    header = "date,latitude,longitude,brightness_temp_37v,brightness_temp_19v\n"

    # Parcours de tous les fichiers merged_amsre_data_*.csv dans les sous-dossiers
    for root, dirs, files in os.walk(amsre_folder):
        for file in files:
            if file.startswith("merged_amsre_data_") and file.endswith(".csv"):
                file_path = os.path.join(root, file)

                # Lire le contenu existant
                with open(file_path, "r") as f:
                    content = f.read()

                # Vérifier si le fichier contient déjà l'en-tête
                if not content.startswith("date,latitude"):
                    print(f"✍️ Ajout de l'en-tête à {file_path}")
                    with open(file_path, "w") as f:
                        f.write(header + content)
                else:
                    print(f"✅ En-tête déjà présent dans {file_path}")