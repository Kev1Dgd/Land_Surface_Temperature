import numpy as np
import os


def clean_lst_data(lst_data):
    """Replace invalid or missing values in LST data."""
    lst_data[lst_data == 0] = np.nan
    return lst_data

def check_and_create_file(output_file, create_func, input_dir):
    if os.path.exists(output_file):
        print(f"Le fichier {output_file} existe déjà. Aucune nouvelle génération effectuée.")
        return  # Ne rien faire si le fichier existe déjà
    else:
        print(f"📂 Création du fichier {output_file}...")
        create_func(input_dir, output_file)  # Appelle la fonction de création avec les arguments passés
