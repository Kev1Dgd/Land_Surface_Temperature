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
                "brightness_temp_19GHz",
                "brightness_temp_37GHz",
                "land_cover_class"
            ]
            df = df[final_cols]

            # Save
            df.to_csv(output_path, index=False)
            tqdm.write(f"\n✅ Merge completed: {output_path}")

        except Exception as e:
            tqdm.write(f"❌ Error processing {modis_file}: {e}")



def concat_amsre_files(
    input_dir_vertical="data/processed/amsre/vertical_polarization",
    input_dir_horizontal="data/processed/amsre/horizontal_polarization",
    output_file="data/processed/machine_learning/merged_amsre_data.csv"
):
    import os
    import re
    import pandas as pd

    if os.path.exists(output_file):
        print(f"⚠️ Final file already exists: {output_file}")
        print("→ No recalculation performed.")
        return

    dates_vert = {d for d in os.listdir(input_dir_vertical) if re.match(r"\d{4}-\d{2}-\d{2}", d)}
    dates_horiz = {d for d in os.listdir(input_dir_horizontal) if re.match(r"\d{4}-\d{2}-\d{2}", d)}
    date_folders = sorted(list(dates_vert & dates_horiz))

    all_dfs = []

    for date_folder in date_folders:
        path_vert = os.path.join(input_dir_vertical, date_folder)
        path_horiz = os.path.join(input_dir_horizontal, date_folder)

        file_19v = os.path.join(path_vert, f"amsre_merged_19GHz_vertical_{date_folder}.csv")
        file_37v = os.path.join(path_vert, f"amsre_merged_37GHz_vertical_{date_folder}.csv")
        file_19h = os.path.join(path_horiz, f"amsre_merged_19GHz_horizontal_{date_folder}.csv")
        file_37h = os.path.join(path_horiz, f"amsre_merged_37GHz_horizontal_{date_folder}.csv")

        if not all(os.path.exists(f) for f in [file_19v, file_19h, file_37v, file_37h]):
            print(f"⚠️ Missing files for {date_folder}, skipped.")
            continue

        try:
            df_19v = pd.read_csv(file_19v)
            df_37v = pd.read_csv(file_37v)
            df_19h = pd.read_csv(file_19h)
            df_37h = pd.read_csv(file_37h)

            # Supprimer les colonnes inutiles pour éviter les collisions
            for df in [df_19v, df_37v, df_19h, df_37h]:
                for col in ["pass_type", "orbit", "time"]:  # adapte selon ton cas
                    if col in df.columns:
                        df.drop(columns=[col], inplace=True)

            df_19v = df_19v.rename(columns={
                "latitude": "lat",
                "longitude": "lon",
                "brightness_temp_19v": "brightness_temp_19GHz_v"
            })
            df_37v = df_37v.rename(columns={
                "latitude": "lat",
                "longitude": "lon",
                "brightness_temp_37v": "brightness_temp_37GHz_v"
            })
            df_19h = df_19h.rename(columns={
                "latitude": "lat",
                "longitude": "lon",
                "brightness_temp_19h": "brightness_temp_19GHz_h"
            })
            df_37h = df_37h.rename(columns={
                "latitude": "lat",
                "longitude": "lon",
                "brightness_temp_37h": "brightness_temp_37GHz_h"
            })

            # Fusion sans duplication
            df = df_19v.merge(df_37v, on=["lat", "lon"], how="outer") \
                       .merge(df_19h, on=["lat", "lon"], how="outer") \
                       .merge(df_37h, on=["lat", "lon"], how="outer")

            df["date"] = date_folder

            df_final = df[[
                "lat", "lon", "date",
                "brightness_temp_19GHz_v", "brightness_temp_19GHz_h",
                "brightness_temp_37GHz_v", "brightness_temp_37GHz_h"
            ]]

            df_final = df_final.groupby(["lat", "lon", "date"], as_index=False).mean(numeric_only=True)
            all_dfs.append(df_final)

            print(f"✅ Successfully merged for {date_folder} ({len(df_final)} rows)")

        except Exception as e:
            print(f"❌ Error processing {date_folder}: {e}")

    if not all_dfs:
        print("\n❌ No data merged.")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv(output_file, index=False)
    print(f"\n✅ Final dataset saved: {output_file} ({len(df_all)} rows)")