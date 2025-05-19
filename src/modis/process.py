import os
import xarray as xr
import pandas as pd

def process_nc_to_csv_light(nc_file, output_csv, variable_name="LST_Day_1km", day_index=0):
    """
    Traite un fichier NetCDF volumineux en ne chargeant qu'un seul pas temporel pour éviter les problèmes de mémoire.
    Si le fichier CSV existe déjà, il ne refait pas le traitement.
    """
    if os.path.exists(output_csv):
        print(f"⚠️ Fichier déjà existant, on saute : {output_csv}")
        return

    try:
        print(f"📂 Lecture du fichier NetCDF : {nc_file}")
        ds = xr.open_dataset(nc_file)

        if variable_name not in ds:
            raise ValueError(f"La variable '{variable_name}' n'existe pas dans le fichier.")

        # Sélection d’un seul jour pour réduire la charge mémoire
        if "time" in ds.dims:
            lst = ds[variable_name].isel(time=day_index)
        else:
            lst = ds[variable_name]

        # Application des attributs si présents
        if "scale_factor" in lst.attrs:
            lst = lst * lst.attrs["scale_factor"]
        if "add_offset" in lst.attrs:
            lst = lst + lst.attrs["add_offset"]

        # Conversion en DataFrame
        df = lst.to_dataframe(name="LST_Kelvin").reset_index()
        df = df.dropna(subset=["LST_Kelvin"])
        df["LST_Celsius"] = df["LST_Kelvin"] - 273.15

        df.to_csv(output_csv, index=False)
        print(f"✅ CSV sauvegardé : {output_csv}")
    except Exception as e:
        print(f"❌ Erreur : {e}")
