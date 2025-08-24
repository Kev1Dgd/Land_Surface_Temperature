import os
import pandas as pd
import glob
import numpy as np
from datetime import datetime

from src.modis.plot import plot_modis_lst_map


def process_nc_to_csv_light(ds, output_csv, day_index, variable_name="LST_Day_1km"):
    if os.path.exists(output_csv):
        print(f"✅ Already existing file, skip : {output_csv}")
        return

    try:
        if variable_name not in ds:
            raise ValueError(f"The variable '{variable_name}' does not exist in the file.")
        
        if "time" in ds.dims:
            lst = ds[variable_name].isel(time=day_index)
        else:
            lst = ds[variable_name]
        
        if "scale_factor" in lst.attrs:
            lst = lst * lst.attrs["scale_factor"]
        if "add_offset" in lst.attrs:
            lst = lst + lst.attrs["add_offset"]
        
        df = lst.to_dataframe(name="LST_Kelvin").reset_index()
        df = df.dropna(subset=["LST_Kelvin"])
        df = df[["lat", "lon", "LST_Kelvin"]]

        lon_min, lat_min, lon_max, lat_max = -12.984, 35.290, 38.018, 64.090
        df = df[(df["lon"] >= lon_min) & (df["lon"] <= lon_max) &
                (df["lat"] >= lat_min) & (df["lat"] <= lat_max)]
        
        df["lat"] = np.round(df["lat"] * 4) / 4
        df["lon"] = np.round(df["lon"] * 4) / 4
        
        df = df.groupby(["lat", "lon"], as_index=False).mean()
        df["LST_Celsius"] = df["LST_Kelvin"] - 273.15
        
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"✅ CSV sauvegardé : {output_csv}")

    except Exception as e:
        print(f"❌ Erreur : {e}")


def process_all_modis_csv(input_folder="data/processed/modis", output_folder="outputs/modis/dates"):
    csv_files = sorted(glob.glob(os.path.join(input_folder, "modis_lst_*.csv")))
    os.makedirs(output_folder, exist_ok=True)

    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            # Extraire la date du nom de fichier
            filename = os.path.basename(csv_file)
            date_str = filename.replace("modis_lst_", "").replace(".csv", "")
            date = datetime.strptime(date_str, "%Y-%m-%d")

            df["date"] = date
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Erreur avec le fichier {csv_file} : {e}")
            continue

    if not dfs:
        print("Aucun fichier CSV chargé.")
        return

    full_df = pd.concat(dfs, ignore_index=True)

    # Grouper par date, lat, lon et moyenne
    grouped = full_df.groupby(["date", "lat", "lon"], as_index=False).mean()

    # Pour chaque date, créer la carte
    for date, group_df in grouped.groupby("date"):
        date_str = date.strftime("%Y-%m-%d")
        plot_modis_lst_map(group_df, date_str, output_dir=output_folder)