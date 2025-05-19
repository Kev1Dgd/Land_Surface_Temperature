import xarray as xr
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

def load_land_cover_map(nc_path, target_lat, target_lon, method="nearest"):
    """
    Charge la carte land cover (IGBP) et interpole sur la grille cible (MODIS ou AMSRE).
    
    :param nc_path: chemin du fichier NetCDF (.nc4)
    :param target_lat: np.array ou pd.Series des latitudes à interpoler
    :param target_lon: np.array ou pd.Series des longitudes à interpoler
    :param method: méthode d'interpolation ("nearest", "linear" ou "cubic")
    :return: land cover interpolé (np.array)
    """
    print(f"📦 Chargement de la carte land cover depuis : {nc_path}")
    ds = xr.open_dataset(nc_path)

    # Extraire les données
    lc = ds["land_cover_class"].squeeze()  # 2D
    lats = ds["latitude"].values
    lons = ds["longitude"].values
    values = lc.values

    # Créer une grille des coordonnées sources
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    points = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))
    values_flat = values.ravel()

    # Préparer les points cibles
    target_points = np.column_stack((target_lat, target_lon))

    print("🧮 Interpolation des classes d’occupation du sol...")
    land_cover_interp = griddata(points, values_flat, target_points, method=method)

    return land_cover_interp
