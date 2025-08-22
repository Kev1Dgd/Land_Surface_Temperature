import numpy as np
import pandas as pd
import os
from netCDF4 import Dataset

DEBUG = True

def debug_print(msg):
    if DEBUG:
        print(f"[DEBUG] {msg}")

def extract_bt(file_path, frequency):
    try:
        if not file_path.endswith(".hdf"):
            debug_print(f"Ignored non-HDF file: {file_path}")
            return None, None, None

        ds = Dataset(file_path, mode='r')

        freq_str = "18.7" if frequency == 19 else "36.5"
        tb_var_names = [f"{freq_str}V_Res.{i}_TB" for i in range(1, 6)] + [f"{freq_str}V_Res.5B_TB"]

        tb, tb_var = None, None
        for name in tb_var_names:
            if name in ds.variables:
                tb_var = ds.variables[name]
                tb = tb_var[:]
                break
        if tb is None:
            raise KeyError(f"No TB {frequency} GHz V variable found.")

        scale = tb_var.getncattr("SCALE FACTOR") if "SCALE FACTOR" in tb_var.ncattrs() else 1.0
        offset = tb_var.getncattr("OFFSET") if "OFFSET" in tb_var.ncattrs() else 0.0

        tb_corrected = tb * scale + offset
        tb_corrected = np.where(tb_corrected > 6550, np.nan, tb_corrected)

        lat = ds.variables["Latitude"][:]
        lon = ds.variables["Longitude"][:]
        ds.close()

        lat = np.where(lat > 90, lat - 180, lat)
        lon = np.where(lon > 180, lon - 360, lon)
        lat = np.round(lat, 2)
        lon = np.round(lon, 2)

        return lat, lon, tb_corrected

    except Exception as e:
        debug_print(f"Error : {e}")
        return None, None, None
    

def combine_amsre_files(files, date, frequency, output_dir="data/processed/amsre"):
    lon_min, lat_min, lon_max, lat_max = -12.984, 35.290, 38.018, 64.090
    resolution = 0.25

    freq_str = f"{frequency}GHz"
    date_output_dir = os.path.join(output_dir, date)
    os.makedirs(date_output_dir, exist_ok=True)

    output_path_ascending = os.path.join(date_output_dir, f"amsre_combined_{freq_str}_{date}_ascending.csv")
    output_path_descending = os.path.join(date_output_dir, f"amsre_combined_{freq_str}_{date}_descending.csv")

    if os.path.exists(output_path_ascending) and os.path.exists(output_path_descending):
        print(f"⏭️ The {freq_str} combined files already exist.")
        return output_path_ascending, output_path_descending

    all_data_ascending = []
    all_data_descending = []

    for file_path in files:
        lat, lon, bt = extract_bt(file_path, frequency)

        if lat is not None and bt is not None:
            if lat.shape[1] == 2 * bt.shape[1] and lat.shape[0] == bt.shape[0]:
                lat = (lat[:, ::2] + lat[:, 1::2]) / 2
                lon = (lon[:, ::2] + lon[:, 1::2]) / 2
            elif lat.shape != bt.shape:
                debug_print("❌ Non-trivial dimensional incompatibility, file ignored")
                continue

            bt_flat = bt.flatten()
            lat_flat = lat.flatten()
            lon_flat = lon.flatten()

            valid_coords = (lat_flat >= lat_min) & (lat_flat <= lat_max) & (lon_flat >= lon_min) & (lon_flat <= lon_max)
            valid = ~np.isnan(bt_flat) & valid_coords

            data = pd.DataFrame({
                "latitude": lat_flat[valid],
                "longitude": lon_flat[valid],
                f"brightness_temp_{frequency}v": bt_flat[valid],
                "pass_type": "ascending" if '_A' in file_path else "descending"
            })

            if '_A' in file_path:
                all_data_ascending.append(data)
            elif '_D' in file_path:
                all_data_descending.append(data)
            
            del lat, lon, bt, bt_flat, lat_flat, lon_flat, data

    for data_list, output_path in zip(
        [all_data_ascending, all_data_descending],
        [output_path_ascending, output_path_descending]
    ):
        if data_list:
            df = pd.concat(data_list, ignore_index=True)
            df["latitude"] = (df["latitude"] / resolution).round() * resolution
            df["longitude"] = (df["longitude"] / resolution).round() * resolution
            df = df.groupby(["latitude", "longitude", "pass_type"], as_index=False).mean(numeric_only=True)
            df.to_csv(output_path, index=False)
            print(f"✅ {df['pass_type'].iloc[0].capitalize()} {freq_str} CSV saved in {output_path}")

    return output_path_ascending, output_path_descending

def merge_amsre_csvs_per_frequency(date, base_dir="data/processed/amsre"):
    output_dir=f"data/processed/amsre/{date}"
    freq_list = [19, 37]
    os.makedirs(output_dir, exist_ok=True)
    
    for freq in freq_list:
        freq_str = f"{freq}GHz"
        date_dir = os.path.join(base_dir, date)
        
        asc_path = os.path.join(date_dir, f"amsre_combined_{freq_str}_{date}_ascending.csv")
        desc_path = os.path.join(date_dir, f"amsre_combined_{freq_str}_{date}_descending.csv")
        
        if not (os.path.exists(asc_path) and os.path.exists(desc_path)):
            print(f"⚠️ Missing CSVs for {freq_str} on {date}. Skipping merge.")
            continue

        try:
            df_asc = pd.read_csv(asc_path)
            df_desc = pd.read_csv(desc_path)
            
            merged_df = pd.concat([df_asc, df_desc], ignore_index=True)
            output_path = os.path.join(output_dir, f"amsre_merged_{freq_str}_{date}.csv")
            merged_df.to_csv(output_path, index=False)
            
            print(f"✅ Merged {freq_str} CSV for {date} saved at {output_path}")
        except Exception as e:
            print(f"❌ Error merging CSVs for {freq_str} on {date}: {e}")
