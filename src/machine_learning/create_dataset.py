import os
import pandas as pd
from glob import glob
from tqdm import tqdm
from src.land_cover.process import load_land_cover_map
from datetime import datetime, timedelta

def _load_and_prepare_csv(path, lat_col="lat", lon_col="lon", date_col=None, date_fmt=None):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={lat_col: "latitude", lon_col: "longitude"})
    df["latitude"] = df["latitude"].round(1)
    df["longitude"] = df["longitude"].round(1)

    if date_col and date_fmt:
        df["date"] = pd.to_datetime(df[date_col], format=date_fmt).dt.strftime("%Y-%m-%d")
    elif date_col:
        df["date"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
    
    return df

def merge_daily_datasets(modis_folder="data/processed/modis",
                         amsre_folder="data/processed/amsre",
                         land_cover_path="data/raw/land_cover/968_Land_Cover_Class_0.25degree.nc4",
                         output_folder="data/processed/machine_learning"):
    os.makedirs(output_folder, exist_ok=True)
    modis_files = sorted(glob(os.path.join(modis_folder, "*.csv")))

    for modis_file in tqdm(modis_files, desc="Merge MODIS + AMSRE + Land Cover files"):
        try:
            # Extract date from file name
            doy_str = os.path.basename(modis_file).split("_")[-1].replace(".csv", "")
            doy = int(doy_str)
            date_obj = datetime(2005, 1, 1) + timedelta(days=doy)
            date_str = date_obj.strftime("%Y-%m-%d")

            # Loading MODIS
            df_modis = _load_and_prepare_csv(modis_file, lat_col="lat", lon_col="lon", date_col="time")
            if "LST_Kelvin" not in df_modis.columns:
                tqdm.write(f"⚠️ Column 'LST_Kelvin' missing in {modis_file}")
                continue

            # Add LST_Celsius
            df_modis["LST_Celsius"] = df_modis["LST_Kelvin"] - 273.15

            # Loading AMSRE
            amsre_file = os.path.join(amsre_folder, date_str, f"merged_amsre_data_{date_str}.csv")
            if not os.path.exists(amsre_file):
                tqdm.write(f"⚠️ Missing AMSRE file : {amsre_file}")
                continue
            df_amsre = _load_and_prepare_csv(amsre_file, lat_col="latitude", lon_col="longitude", date_col="date")

            # Merger MODIS + AMSRE
            df = pd.merge(df_modis, df_amsre, on=["latitude", "longitude", "date"], how="inner")
            if df.empty:
                tqdm.write(f"❌ No correspondence for {date_str}")
                continue

            # Add Land Cover
            df["land_cover_class"] = load_land_cover_map(
                nc_path=land_cover_path,
                target_lat=df["latitude"].values,
                target_lon=df["longitude"].values,
                method="nearest"
            )

            # Final columns
            final_cols = [
                "latitude",
                "longitude",
                "LST_Kelvin",
                "LST_Celsius",
                "brightness_temp_37v",
                "brightness_temp_19v",
                "land_cover_class"
            ]
            df = df[final_cols]

            # Save
            output_path = os.path.join(output_folder, f"{date_str}.csv")
            df.to_csv(output_path, index=False)
            tqdm.write(f"✅ Merger completed : {output_path}")

        except Exception as e:
            tqdm.write(f"❌ Error for {modis_file} : {e}")
