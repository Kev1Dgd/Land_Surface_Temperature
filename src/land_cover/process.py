import os
import xarray as xr
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

def load_land_cover_map(nc_path, target_lat, target_lon, method="nearest"):
    print(f"📦 Load land cover map from : {nc_path}")
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

    print("🧮 Interpolation of land use classes...")
    land_cover_interp = griddata(points, values_flat, target_points, method=method)

    return land_cover_interp

def convert_land_cover_nc_to_csv(nc_path, output_csv_path):
    if os.path.exists(output_csv_path):
        print(f"⚠️ The CSV file already exists: {output_csv_path}. Conversion ignored.")
        return
    ds = xr.open_dataset(nc_path)
    var_name = list(ds.data_vars)[0]
    land_cover = ds[var_name]
    df = land_cover.to_dataframe().reset_index()
    df.to_csv(output_csv_path, index=False)
    print(f"✅ CSV saved at : {output_csv_path}")
