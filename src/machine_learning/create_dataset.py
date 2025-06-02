import os
import pandas as pd
from glob import glob
from tqdm import tqdm
from src.land_cover.process import load_land_cover_map
from datetime import datetime
import re

def _load_and_prepare_csv(path, lat_col="lat", lon_col="lon", date_col=None, date_fmt=None):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df[lat_col] = df[lat_col].round(2)
    df[lon_col] = df[lon_col].round(2)

    if date_col and date_fmt:
        df["date"] = pd.to_datetime(df[date_col], format=date_fmt).dt.strftime("%Y-%m-%d")
    elif date_col:
        df["date"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
    
    return df

def merge_daily_datasets(modis_folder="data/processed/modis",
                         amsre_file_global="data/processed/machine_learning/merged_amsre_data.csv",
                         land_cover_path="data/raw/land_cover/968_Land_Cover_Class_0.25degree.nc4",
                         output_folder="data/processed/machine_learning/dates"):

    os.makedirs(output_folder, exist_ok=True)
    modis_files = sorted(glob(os.path.join(modis_folder, "*.csv")))

    # Load the global AMSRE file once
    if not os.path.exists(amsre_file_global):
        raise FileNotFoundError(f"Missing global AMSRE file: {amsre_file_global}")
    df_amsre_global = _load_and_prepare_csv(amsre_file_global, lat_col="lat", lon_col="lon", date_col="date")

    date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")  # regex date yyyy-mm-dd

    for modis_file in tqdm(modis_files, desc="Merging MODIS + AMSRE + Land Cover files"):
        try:
            # Extract date from filename
            filename = os.path.basename(modis_file)
            match = date_pattern.search(filename)
            if not match:
                tqdm.write(f"⚠️ Ignored file (no date found): {filename}")
                continue
            date_str = match.group(0)
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")

            # Check if output file already exists
            output_path = os.path.join(output_folder, f"{date_str}.csv")
            if os.path.exists(output_path):
                tqdm.write(f"⏭️ Already exists, skipped: {output_path}")
                continue

            # Load MODIS without expecting date column
            df_modis = _load_and_prepare_csv(modis_file, lat_col="lat", lon_col="lon")
            df_modis["date"] = date_str

            if "LST_Kelvin" not in df_modis.columns:
                tqdm.write(f"⚠️ Missing column 'LST_Kelvin' in {modis_file}")
                continue

            if "LST_Celsius" not in df_modis.columns:
                df_modis["LST_Celsius"] = df_modis["LST_Kelvin"] - 273.15

            # Filter AMSRE global for current date
            df_amsre = df_amsre_global[df_amsre_global["date"] == date_str]
            if df_amsre.empty:
                tqdm.write(f"⚠️ No AMSRE data for date: {date_str}")
                continue

            # Merge MODIS + AMSRE
            df = pd.merge(df_modis, df_amsre, on=["lat", "lon", "date"], how="inner")
            if df.empty:
                tqdm.write(f"❌ No matching records for {date_str}")
                continue

            # Add land cover
            df["land_cover_class"] = load_land_cover_map(
                nc_path=land_cover_path,
                target_lat=df["lat"].values,
                target_lon=df["lon"].values,
                method="nearest"
            )

            # Final columns
            final_cols = [
                "lat",
                "lon",
                "LST_Kelvin",
                "LST_Celsius",
                "brightness_temp_mean",
                "land_cover_class"
            ]
            df = df[final_cols]

            # Save
            df.to_csv(output_path, index=False)
            tqdm.write(f"\n✅ Merge completed: {output_path}")

        except Exception as e:
            tqdm.write(f"❌ Error processing {modis_file}: {e}")



def concat_amsre_files(input_dir="data/processed/amsre", output_file="data/processed/machine_learning/merged_amsre_data.csv"):
    if os.path.exists(output_file):
        print(f"⚠️ Final file already exists: {output_file}")
        print("→ No recalculation performed.")
        return

    date_folders = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d)) and re.match(r"\d{4}-\d{2}-\d{2}", d)
    ])

    all_dfs = []

    for date_folder in date_folders:
        folder_path = os.path.join(input_dir, date_folder)

        file_19 = os.path.join(folder_path, f"amsre_merged_19GHz_{date_folder}.csv")
        file_37 = os.path.join(folder_path, f"amsre_merged_37GHz_{date_folder}.csv")

        if not os.path.exists(file_19) or not os.path.exists(file_37):
            print(f"⚠️ Missing merged files for date {date_folder}, skipped")
            continue

        try:
            df_19 = pd.read_csv(file_19)
            df_37 = pd.read_csv(file_37)

            # Rename columns to unify
            df_19 = df_19.rename(columns={"latitude": "lat", "longitude": "lon"})
            df_37 = df_37.rename(columns={"latitude": "lat", "longitude": "lon"})

            # Merge on lat, lon
            df = pd.merge(df_19, df_37, on=["lat", "lon"], suffixes=("_19", "_37"))

            # Calculate mean brightness temp
            df["brightness_temp_mean"] = (df["brightness_temp_19v"] + df["brightness_temp_37v"]) / 2
            df["date"] = date_folder

            # Keep only necessary columns
            df_final = df[["lat", "lon", "date", "brightness_temp_mean"]]

            # Remove duplicates by averaging if any
            df_final = df_final.groupby(["lat", "lon", "date"], as_index=False).mean(numeric_only=True)

            all_dfs.append(df_final)

            print(f"✅ Successful merge for {date_folder} ({len(df_final)} rows)")

        except Exception as e:
            print(f"❌ Error merging date {date_folder}: {e}")

    if not all_dfs:
        print("\n❌ No files could be merged.")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv(output_file, index=False)
    print(f"\n✅ Final file saved: {output_file} ({len(df_all)} rows)")
