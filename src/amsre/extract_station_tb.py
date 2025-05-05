import os
from pathlib import Path
import numpy as np
import pandas as pd
from netCDF4 import Dataset

from src.amsre.stations import find_nearest_station

PROCESSED_DIR = "data/processed/amsre/stations"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def extract_tb_for_stations(hdf_files, date, tolerance=0.3):
    print(f"\n📡 Traitement AMSR-E station pour la date : {date}...")

    results = []

    for file_path in hdf_files:
        try:
            ds = Dataset(file_path, 'r')
            lat = ds.variables["Latitude"][:]
            lon = ds.variables["Longitude"][:]

            tb_var = None
            for key in ds.variables:
                if "37 GHz" in key and "(V-pol)" in key:
                    tb_var = ds.variables[key]
                    break

            if tb_var is None:
                print(f"⏭️ Pas de TB 37GHz V trouvée dans {file_path}")
                continue

            tb = tb_var[:]
            tb = np.where(tb > 6550, np.nan, tb)
            scale = tb_var.getncattr("SCALE FACTOR") if "SCALE FACTOR" in tb_var.ncattrs() else 1.0
            tb = tb * scale

            matches = find_nearest_station(lat, lon, tolerance=tolerance)
            for m in matches:
                i, j = m['i'], m['j']
                results.append({
                    "file": os.path.basename(file_path),
                    "station": m["station"],
                    "station_lat": m["station_lat"],
                    "station_lon": m["station_lon"],
                    "pixel_lat": m["lat"],
                    "pixel_lon": m["lon"],
                    "distance": m["distance"],
                    "TB": tb[i, j]
                })

            ds.close()
        except Exception as e:
            print(f"[ERREUR] {file_path} -> {e}")
            continue

    if results:
        df = pd.DataFrame(results)
        output_path = Path(PROCESSED_DIR) / f"amsre_tb_station_{date}.csv"
        df.to_csv(output_path, index=False)
        print(f"✅ Données extraites et enregistrées : {output_path}")
        return str(output_path)
    else:
        print(f"❌ Aucune station correspondante trouvée pour les fichiers du {date}")
        return None
