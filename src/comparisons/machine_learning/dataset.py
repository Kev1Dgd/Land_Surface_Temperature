import os
import pandas as pd
import re
from glob import glob
from tqdm import tqdm

from src.land_cover.process import load_land_cover_map


def concat_full_amsre_spatial(input_dir_vertical, input_dir_horizontal, output_file):
                              
    lon_min, lat_min, lon_max, lat_max = -12.984, 35.290, 38.018, 64.090

    if os.path.exists(output_file):
        print(f"⚠️ Final file already exists: {output_file}")
        print("→ No recalculation performed.")
        return

    dates_vert = {d for d in os.listdir(input_dir_vertical) if re.match(r"\d{4}-\d{2}-\d{2}", d)}
    dates_horiz = {d for d in os.listdir(input_dir_horizontal) if re.match(r"\d{4}-\d{2}-\d{2}", d)}
    date_folders = sorted(list(dates_vert & dates_horiz))

    all_dfs = []

    print(f"📅 Processing {len(date_folders)} dates...\n")

    for date_folder in date_folders:
        try:
            def load_split(file_path, freq, pol):
                df = pd.read_csv(file_path)

                if 'pass_type' in df.columns:
                    orbit_col = 'pass_type'
                elif 'orbit' in df.columns:
                    orbit_col = 'orbit'
                else:
                    raise ValueError("Missing orbit/pass_type column in file")

                df = df.rename(columns={
                    "latitude": "lat",
                    "longitude": "lon",
                    f"brightness_temp_{freq}{pol}": "brightness_temp"
                })

                df = df[["lat", "lon", orbit_col, "brightness_temp"]]
                df = df[(df["lat"] >= lat_min) & (df["lat"] <= lat_max) &
                        (df["lon"] >= lon_min) & (df["lon"] <= lon_max)]

                df_asc = df[df[orbit_col] == "ascending"].copy()
                df_desc = df[df[orbit_col] == "descending"].copy()

                df_asc = df_asc.drop(columns=[orbit_col]).rename(
                    columns={"brightness_temp": f"brightness_temp_{freq}{pol}_asc"}
                )
                df_desc = df_desc.drop(columns=[orbit_col]).rename(
                    columns={"brightness_temp": f"brightness_temp_{freq}{pol}_desc"}
                )

                return df_asc, df_desc

            path_vert = os.path.join(input_dir_vertical, date_folder)
            path_horiz = os.path.join(input_dir_horizontal, date_folder)

            files = {
                "19v": os.path.join(path_vert, f"amsre_merged_19GHz_vertical_{date_folder}.csv"),
                "37v": os.path.join(path_vert, f"amsre_merged_37GHz_vertical_{date_folder}.csv"),
                "19h": os.path.join(path_horiz, f"amsre_merged_19GHz_horizontal_{date_folder}.csv"),
                "37h": os.path.join(path_horiz, f"amsre_merged_37GHz_horizontal_{date_folder}.csv"),
            }

            if not all(os.path.exists(f) for f in files.values()):
                print(f"⚠️ Missing files for {date_folder}, skipped.")
                continue

            dfs = []
            for key, path in files.items():
                freq = key[:2]
                pol = key[2]
                asc, desc = load_split(path, freq, pol)
                dfs.extend([asc, desc])

            # Fusion des 8 sous-ensembles
            df_merged = dfs[0]
            for df in dfs[1:]:
                df_merged = df_merged.merge(df, on=["lat", "lon"], how="outer")

            df_merged["date"] = date_folder
            cols = ["lat", "lon", "date"] + sorted([c for c in df_merged.columns if c.startswith("brightness_temp")])
            df_merged = df_merged[cols]

            all_dfs.append(df_merged)
            print(f"✅ {date_folder}: {len(df_merged)} rows merged")

        except Exception as e:
            print(f"❌ Error on {date_folder}: {e}")

    if not all_dfs:
        print("\n❌ No data merged.")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv(output_file, index=False)
    print(f"\n✅ Final dataset saved: {output_file} ({len(df_all)} rows)")


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


def merge_daily_full_datasets(
    modis_folder="data/processed/modis",
    amsre_file_global="data/processed/comparisons/machine_learning/merged_full_amsre_data.csv",
    land_cover_path="data/raw/land_cover/968_Land_Cover_Class_0.25degree.nc4",
    output_folder="data/processed/comparisons/machine_learning/dates"
):
    os.makedirs(output_folder, exist_ok=True)
    modis_files = sorted(glob(os.path.join(modis_folder, "*.csv")))

    if not os.path.exists(amsre_file_global):
        raise FileNotFoundError(f"Missing global AMSRE file: {amsre_file_global}")
    df_amsre_global = _load_and_prepare_csv(amsre_file_global, lat_col="lat", lon_col="lon", date_col="date")

    date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")

    for modis_file in tqdm(modis_files, desc="Merging MODIS + AMSRE + Land Cover files"):
        try:
            filename = os.path.basename(modis_file)
            match = date_pattern.search(filename)
            if not match:
                tqdm.write(f"⚠️ Ignored file (no date found): {filename}")
                continue
            date_str = match.group(0)

            output_path = os.path.join(output_folder, f"{date_str}.csv")
            if os.path.exists(output_path):
                tqdm.write(f"⏭️ Already exists, skipped: {output_path}")
                continue

            df_modis = _load_and_prepare_csv(modis_file, lat_col="lat", lon_col="lon")
            df_modis["date"] = date_str

            if "LST_Kelvin" not in df_modis.columns:
                tqdm.write(f"⚠️ Missing column 'LST_Kelvin' in {modis_file}")
                continue

            if "LST_Celsius" not in df_modis.columns:
                df_modis["LST_Celsius"] = df_modis["LST_Kelvin"] - 273.15

            df_amsre = df_amsre_global[df_amsre_global["date"] == date_str]
            if df_amsre.empty:
                tqdm.write(f"⚠️ No AMSRE data for date: {date_str}")
                continue

            df = pd.merge(df_modis, df_amsre, on=["lat", "lon", "date"], how="inner")
            if df.empty:
                tqdm.write(f"❌ No matching records for {date_str}")
                continue

            # Chargement de la carte land cover (supposé que ta fonction est définie ailleurs)
            df["land_cover_class"] = load_land_cover_map(
                nc_path=land_cover_path,
                target_lat=df["lat"].values,
                target_lon=df["lon"].values,
                method="nearest"
            )

            final_cols = [
                "lat",
                "lon",
                "LST_Kelvin",
                "LST_Celsius",
                "brightness_temp_19v_asc",
                "brightness_temp_19v_desc",
                "brightness_temp_37v_asc",
                "brightness_temp_37v_desc",
                "brightness_temp_19h_asc",
                "brightness_temp_19h_desc",
                "brightness_temp_37h_asc",
                "brightness_temp_37h_desc",
                "land_cover_class"
            ]
            
            # Certaines colonnes AMSRE peuvent manquer selon les données, on vérifie
            final_cols = [col for col in final_cols if col in df.columns]

            df = df[final_cols]

            df.to_csv(output_path, index=False)
            tqdm.write(f"\n✅ Merge completed: {output_path}")

        except Exception as e:
            tqdm.write(f"❌ Error processing {modis_file}: {e}")
        

