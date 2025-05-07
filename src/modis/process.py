import numpy as np
import pandas as pd
from pyhdf.SD import SD, SDC

def lat_lon_from_projection(x_dim, y_dim, upper_left, lower_right):
    """
    Calculer les coordonnées géographiques (latitude, longitude) de chaque pixel
    en fonction des dimensions du grid et des coordonnées des coins du fichier.
    """
    # Extraire les informations des coins
    x_min, y_max = upper_left
    x_max, y_min = lower_right
    
    # Créer les coordonnées en x et y
    x_coords = np.linspace(x_min, x_max, x_dim)
    y_coords = np.linspace(y_max, y_min, y_dim)
    
    # Créer une grille de coordonnées (X, Y)
    lon_grid, lat_grid = np.meshgrid(x_coords, y_coords)
    
    return lat_grid, lon_grid

def extract_lst_from_hdf(hdf_file):
    """Extraire les données LST, latitude et longitude à partir d'un fichier HDF."""
    try:
        hdf = SD(hdf_file, SDC.READ)
        
        # Extraire les données LST
        lst_data = hdf.select('LST_Day_1km')[:]
        
        # Extraire les informations de projection
        upper_left = hdf.attributes()['UpperLeftPointMtrs']
        lower_right = hdf.attributes()['LowerRightMtrs']
        
        # Extraire latitude et longitude en utilisant la projection
        lat_data, lon_data = lat_lon_from_projection(lst_data.shape[0], lst_data.shape[1], upper_left, lower_right)
        
        return lst_data, lat_data, lon_data
    except Exception as e:
        print(f"Erreur lors de l'extraction des données : {e}")
        return None, None, None

def prepare_lst_data(lst_data, lat_data, lon_data):
    """Transformer les données LST en un DataFrame."""
    data_list = []
    for i in range(len(lat_data)):
        for j in range(len(lon_data[0])):
            data_list.append([lat_data[i][j], lon_data[i][j], lst_data[i, j]])
    
    df = pd.DataFrame(data_list, columns=['Latitude', 'Longitude', 'LST'])
    df['LST_Celsius'] = df['LST'] - 273.15  # Conversion en Celsius
    return df

def save_lst_to_csv(df, output_file):
    """Sauvegarder les données LST dans un fichier CSV."""
    try:
        df.to_csv(output_file, index=False)
        print(f"Fichier CSV sauvegardé : {output_file}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du fichier CSV : {e}")

def process_hdf_to_csv(hdf_file, output_file):
    """Fonction principale pour traiter un fichier HDF en CSV."""
    print(f"Traitement du fichier HDF : {hdf_file}")
    lst_data, lat_data, lon_data = extract_lst_from_hdf(hdf_file)
    if lst_data is None or lat_data is None or lon_data is None:
        return
    df = prepare_lst_data(lst_data, lat_data, lon_data)
    save_lst_to_csv(df, output_file)
