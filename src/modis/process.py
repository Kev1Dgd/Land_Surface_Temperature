from pyhdf.SD import SD, SDC
import numpy as np
import pandas as pd
import os
from .utils import convert_kelvin_to_celsius, clean_lst_data

def load_modis_data(file_path):
    """Load and scale LST data from a MODIS .hdf file."""
    try:
        hdf = SD(file_path, SDC.READ)
        lst_dataset = hdf.select('LST_Day_1km')
        lst_data = lst_dataset[:].astype(np.float32)
        lst_data = lst_data * 0.02 - 273.15  # MODIS scaling and Kelvin to °C
        lst_data[lst_data == 0] = np.nan
        return lst_data
    except Exception as e:
        print(f"PyHDF Error while loading {file_path}: {e}")
        return None

def save_lst_to_csv(lst_data, output_file):
    """Save LST array to CSV format."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pd.DataFrame(lst_data).to_csv(output_file, index=False)
    print(f"✅ Saved LST to {output_file}")

def process_modis_files(input_dir="data/raw/modis", output_dir="data/processed/modis"):
    """Process all .hdf files in the directory and save cleaned data to CSV."""
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".hdf"):
            file_path = os.path.join(input_dir, file_name)
            print(f"📄 Processing {file_path}")
            lst_data = load_modis_data(file_path)
            if lst_data is not None:
                lst_data = clean_lst_data(lst_data)
                print(f"🌡️ Min: {np.nanmin(lst_data):.2f}°C, Max: {np.nanmax(lst_data):.2f}°C")
                csv_name = f"lst_{os.path.splitext(file_name)[0]}.csv"
                save_lst_to_csv(lst_data, os.path.join(output_dir, csv_name))
            else:
                print(f"⚠️ Skipped {file_name} due to read error.")