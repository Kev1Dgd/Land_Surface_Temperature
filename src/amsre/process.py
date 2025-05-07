import numpy as np
import pandas as pd
import os
from netCDF4 import Dataset

DEBUG = True  # Set to False to disable debug prints

def debug_print(msg):
    if DEBUG:
        print(f"[DEBUG] {msg}")


def extract_bt_19ghz(file_path):
    try:
        if not file_path.endswith(".hdf"):
            debug_print(f"Ignored non-HDF file: {file_path}")
            return None, None, None, None

        ds = Dataset(file_path, mode='r')
        tb_var_names = [
            '18.7V_Res.1_TB', '18.7V_Res.2_TB', '18.7V_Res.3_TB',
            '18.7V_Res.4_TB', '18.7V_Res.5A_TB', '18.7V_Res.5B_TB'
        ]

        tb = None
        tb_var = None
        for name in tb_var_names:
            if name in ds.variables:
                tb_var = ds.variables[name]
                tb = tb_var[:]
                break
        if tb is None:
            raise KeyError("No TB 19 GHz V variable found.")

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

        return lat, lon, tb_corrected, None

    except Exception as e:
        debug_print(f"Error : {e}")
        return None, None, None, None

def combine_amsre_files_19ghz(files, date, output_dir="data/processed/amsre"):
    date_output_dir = os.path.join(output_dir, date)
    os.makedirs(date_output_dir, exist_ok=True)

    output_path_ascending = os.path.join(date_output_dir, f"amsre_combined_19GHz_{date}_ascending.csv")
    output_path_descending = os.path.join(date_output_dir, f"amsre_combined_19GHz_{date}_descending.csv")

    if os.path.exists(output_path_ascending) and os.path.exists(output_path_descending):
        print(f"⏭️ The 19GHz combined files already exist.")
        return output_path_ascending, output_path_descending

    all_data_ascending = []
    all_data_descending = []

    for file_path in files:
        lat, lon, bt_19v, hour = extract_bt_19ghz(file_path)

        if lat is not None and bt_19v is not None:

            if lat.shape[1] == 2 * bt_19v.shape[1] and lat.shape[0] == bt_19v.shape[0]:
                lat = (lat[:, ::2] + lat[:, 1::2]) / 2
                lon = (lon[:, ::2] + lon[:, 1::2]) / 2
            elif lat.shape != bt_19v.shape:
                debug_print("❌ Non-trivial dimensional incompatibility, file ignored")
                continue

            bt_flat = bt_19v.flatten()
            lat_flat = lat.flatten()
            lon_flat = lon.flatten()
            valid = ~np.isnan(bt_flat)

            if '_A' in file_path:
                pass_type = "ascending"
                data = pd.DataFrame({
                    "latitude": lat_flat[valid],
                    "longitude": lon_flat[valid],
                    "brightness_temp_19v": bt_flat[valid],
                    "pass_type": pass_type
                })
                all_data_ascending.append(data)
            elif '_D' in file_path:
                pass_type = "descending"
                data = pd.DataFrame({
                    "latitude": lat_flat[valid],
                    "longitude": lon_flat[valid],
                    "brightness_temp_19v": bt_flat[valid],
                    "pass_type": pass_type
                })
                all_data_descending.append(data)

    if all_data_ascending:
        df_ascending = pd.concat(all_data_ascending, ignore_index=True)
        df_ascending.to_csv(output_path_ascending, index=False)
        print(f"✅ Ascending 19GHz CSV saved in {output_path_ascending}")

    if all_data_descending:
        df_descending = pd.concat(all_data_descending, ignore_index=True)
        df_descending.to_csv(output_path_descending, index=False)
        print(f"✅ Descending 19GHz CSV saved in {output_path_descending}")

    return output_path_ascending, output_path_descending

def extract_bt_37ghz(file_path):
    try:
        if not file_path.endswith(".hdf"):
            debug_print(f"Ignored non-HDF file: {file_path}")
            return None, None, None, None

        ds = Dataset(file_path, mode='r')
        tb_var_names = [
            '36.5V_Res.1_TB', '36.5V_Res.2_TB', '36.5V_Res.3_TB',
            '36.5V_Res.4_TB', '36.5V_Res.5A_TB', '36.5V_Res.5B_TB'
        ]

        tb = None
        tb_var = None
        for name in tb_var_names:
            if name in ds.variables:
                tb_var = ds.variables[name]
                tb = tb_var[:]
                break
        if tb is None:
            raise KeyError("No TB 37 GHz V variable found.")

        scale = tb_var.getncattr("SCALE FACTOR") if "SCALE FACTOR" in tb_var.ncattrs() else 1.0
        offset = tb_var.getncattr("OFFSET") if "OFFSET" in tb_var.ncattrs() else 0.0

        tb_corrected = tb * scale + offset
        tb_corrected = np.where(tb_corrected > 6550, np.nan, tb_corrected)

        lat = ds.variables["Latitude"][:]
        lon = ds.variables["Longitude"][:]

        ds.close()

        # Ajustement mineur
        lat = np.where(lat > 90, lat - 180, lat)
        lon = np.where(lon > 180, lon - 360, lon)
        lat = np.round(lat, 2)
        lon = np.round(lon, 2)

        return lat, lon, tb_corrected, None

    except Exception as e:
        debug_print(f"Error : {e}")
        return None, None, None, None


def combine_amsre_files_37ghz(files, date, output_dir="data/processed/amsre"):
    # Create a sub-folder for each date
    date_output_dir = os.path.join(output_dir, date)
    os.makedirs(date_output_dir, exist_ok=True)

    # Define the output paths in this folder
    output_path_ascending = os.path.join(date_output_dir, f"amsre_combined_37GHz_{date}_ascending.csv")
    output_path_descending = os.path.join(date_output_dir, f"amsre_combined_37GHz_{date}_descending.csv")

    if os.path.exists(output_path_ascending) and os.path.exists(output_path_descending):
        print(f"⏭️ The combined files already exist.")
        return output_path_ascending, output_path_descending

    all_data_ascending = []
    all_data_descending = []

    for file_path in files:        
        lat, lon, bt_37v, hour = extract_bt_37ghz(file_path)

        if lat is not None and bt_37v is not None:

            if lat.shape[1] == 2 * bt_37v.shape[1] and lat.shape[0] == bt_37v.shape[0]:
                lat = (lat[:, ::2] + lat[:, 1::2]) / 2
                lon = (lon[:, ::2] + lon[:, 1::2]) / 2
            elif lat.shape != bt_37v.shape:
                debug_print("❌ Non-trivial dimensional incompatibility, file ignored")
                continue

            bt_flat = bt_37v.flatten()
            lat_flat = lat.flatten()
            lon_flat = lon.flatten()
            valid = ~np.isnan(bt_flat)

            # Check the file name to determine whether it is an ascending or descending passage
            if '_A' in file_path:
                pass_type = "ascending"
                data = pd.DataFrame({
                    "latitude": lat_flat[valid],
                    "longitude": lon_flat[valid],
                    "brightness_temp_37v": bt_flat[valid],
                    "pass_type": pass_type
                })
                all_data_ascending.append(data)
            elif '_D' in file_path:
                pass_type = "descending"
                data = pd.DataFrame({
                    "latitude": lat_flat[valid],
                    "longitude": lon_flat[valid],
                    "brightness_temp_37v": bt_flat[valid],
                    "pass_type": pass_type
                })
                all_data_descending.append(data)

    # Save bottom-up and top-down data in CSV files
    if all_data_ascending:
        df_ascending = pd.concat(all_data_ascending, ignore_index=True)
        df_ascending.to_csv(output_path_ascending, index=False)
        print(f"✅ Ascending CSV file saved in {output_path_ascending}")

    if all_data_descending:
        df_descending = pd.concat(all_data_descending, ignore_index=True)
        df_descending.to_csv(output_path_descending, index=False)
        print(f"✅ Descending CSV file saved in {output_path_ascending}")

    return output_path_ascending, output_path_descending