import os
import pandas as pd
from glob import glob
from tqdm import tqdm
from src.land_cover.process import load_land_cover_map
import datetime

def merge_daily_datasets(
    modis_folder="data/processed/modis",
    amsre_folder="data/processed/amsre",
    land_cover_path="data/raw/land_cover/968_Land_Cover_Class_0.25degree.nc4",
    output_folder="data/processed/merged"
):
    os.makedirs(output_folder, exist_ok=True)

    # Récupère toutes les dates dispo
    modis_files = sorted(glob(os.path.join(modis_folder, "*.csv")))

    for modis_file in tqdm(modis_files, desc="Fusion des fichiers MODIS + AMSRE + Land Cover"):
        try:
            filename = os.path.basename(modis_file)
            doy_str = filename.split("_")[-1].replace(".csv", "")  # Ex: "0", "123", etc.
            doy = int(doy_str)

            # Convertit DOY en date réelle (en supposant 2005 comme année de référence)
            year = 2005
            date_obj = datetime.datetime(year, 1, 1) + datetime.timedelta(days=doy)
            date_str = date_obj.strftime("%Y-%m-%d")  # Ex: "2005-01-01"

            # Charger MODIS
            df_modis = pd.read_csv(modis_file)
            df_modis.columns = df_modis.columns.str.strip()
            df_modis = df_modis.rename(columns={"lat": "latitude", "lon": "longitude"})

            # Arrondi des coordonnées MODIS
            df_modis["latitude"] = df_modis["latitude"].round(1)
            df_modis["longitude"] = df_modis["longitude"].round(1)

            # Charger AMSRE
            amsre_file = os.path.join(amsre_folder, date_str, f"merged_amsre_data_{date_str}.csv")
            if not os.path.exists(amsre_file):
                print(f"⚠️ Fichier AMSRE manquant pour {date_str}, on passe.")
                continue
            df_amsre = pd.read_csv(amsre_file)

            # Arrondi des coordonnées AMSRE
            df_amsre["latitude"] = df_amsre["latitude"].round(1)
            df_amsre["longitude"] = df_amsre["longitude"].round(1)

            print("🔍 MODIS columns :", df_modis.columns.tolist())
            print("🔍 AMSRE columns :", df_amsre.columns.tolist())

            # Harmonisation des dates
            df_modis["date"] = pd.to_datetime(df_modis["time"]).dt.strftime("%Y-%m-%d")
            df_amsre["date"] = pd.to_datetime(df_amsre["date"]).dt.strftime("%Y-%m-%d")


            # Fusion
            df = pd.merge(df_modis, df_amsre, on=["latitude", "longitude", "date"], how="inner")

            if df.empty:
                print(f"❌ Aucune correspondance de grille pour {date_str}")
                continue

            # Land Cover
            land_cover_classes = load_land_cover_map(
                nc_path=land_cover_path,
                target_lat=df["latitude"].values,
                target_lon=df["longitude"].values,
                method="nearest"
            )
            df["land_cover_class"] = land_cover_classes

            # DoY
            df["DoY"] = doy + 1  # facultatif, si tu veux garder 1-indexé

            # Sauvegarde
            output_path = os.path.join(output_folder, f"merged_dataset_{date_str}.csv")
            df.to_csv(output_path, index=False)
            print(f"✅ Fusion terminée : {output_path}")

        except Exception as e:
            print(f"❌ Erreur pour {modis_file} : {e}")
