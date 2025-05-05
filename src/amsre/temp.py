from pyhdf.SD import SD, SDC

# Fonction pour explorer la structure du fichier HDF
def inspect_hdf4_structure(file_path):
    try:
        f = SD(file_path, SDC.READ)
        print(f"Contenu du fichier HDF4 : {file_path}")
        print("Variables disponibles :")
        for i, sds_name in enumerate(f.datasets()):
            print(f"{i+1}. {sds_name}")
    except Exception as e:
        print(f"Erreur lors de l'exploration du fichier {file_path}: {e}")

# Traiter les fichiers AMSR-E
def traiter_amsr_e_files():
    fichiers = [
        "data/raw/amsre/AMSR_E_L2A_BrightnessTemperatures_V13_200412312344_D.hdf",
        "data/raw/amsre/AMSR_E_L2A_BrightnessTemperatures_V13_200501010033_A.hdf",
        "data/raw/amsre/AMSR_E_L2A_BrightnessTemperatures_V13_200501010123_D.hdf",
        "data/raw/amsre/AMSR_E_L2A_BrightnessTemperatures_V13_200501010212_A.hdf",
        # Ajoutez ici d'autres fichiers si nécessaire
    ]

    # Explorez chaque fichier
    for fichier in fichiers:
        inspect_hdf4_structure(fichier)
