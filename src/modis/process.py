import os
import xarray as xr
import pandas as pd
import glob

from src.modis.plot import plot_modis_lst_map

def process_nc_to_csv_light(nc_file, output_csv, variable_name="LST_Day_1km", day_index=0):
    if os.path.exists(output_csv):
        print(f"⚠️ Existing file, skip : {output_csv}")
        return

    try:
        print(f"📂 Reading the NetCDF file : {nc_file}")
        ds = xr.open_dataset(nc_file)

        if variable_name not in ds:
            raise ValueError(f"The variable '{variable_name}' does not exist in the.")

        # Select a single day to reduce memory load
        if "time" in ds.dims:
            lst = ds[variable_name].isel(time=day_index)
        else:
            lst = ds[variable_name]

        # Apply attributes if present
        if "scale_factor" in lst.attrs:
            lst = lst * lst.attrs["scale_factor"]
        if "add_offset" in lst.attrs:
            lst = lst + lst.attrs["add_offset"]

        # DataFrame conversion
        df = lst.to_dataframe(name="LST_Kelvin").reset_index()
        df = df.dropna(subset=["LST_Kelvin"])
        df["LST_Celsius"] = df["LST_Kelvin"] - 273.15

        df.to_csv(output_csv, index=False)
        print(f"✅ CSV sauvegardé : {output_csv}")
    except Exception as e:
        print(f"❌ Erreur : {e}")


def process_all_modis_csv(input_folder="data/processed/modis", output_folder="outputs/modis/dates"):
    csv_files = sorted(glob(os.path.join(input_folder, "*.csv")))
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        date = pd.to_datetime(df["time"].iloc[0]).strftime("%Y-%m-%d")
        plot_modis_lst_map(df, date, output_dir=output_folder)