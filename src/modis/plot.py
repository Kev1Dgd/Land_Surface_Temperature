import os
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from tqdm import tqdm
from glob import glob


def plot_modis_lst_map(df, date, cmap="coolwarm", output_dir="outputs/modis/dates"):
    output_file = os.path.join(output_dir, date, f"modis_lst_map_{date}.png")
    
    if os.path.exists(output_file):
        print(f"⏭️ Existing map, skip : {output_file}")
        return

    print(f"🗺️ MODIS map generation for {date}...")

    df["lat_bin"] = df["lat"].round(4)
    df["lon_bin"] = df["lon"].round(4)

    df_grouped = df.groupby(["lat_bin", "lon_bin"]).agg({
        "LST_Celsius": "mean"
    }).reset_index()

    # Create a folder for the date
    date_output_dir = os.path.join(output_dir, date)
    os.makedirs(date_output_dir, exist_ok=True)

    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)

    scatter = ax.scatter(
        df_grouped["lon_bin"], df_grouped["lat_bin"],
        c=df_grouped["LST_Celsius"], cmap=cmap, s=10,
        transform=ccrs.PlateCarree(), alpha=0.8,
        vmin=-40, vmax=50
    )

    plt.title(f"MODIS Land Surfate Temperature (LST) – {date}")
    plt.colorbar(scatter, label="LST (°C)", orientation="vertical", shrink=0.7, pad=0.05)

    plt.tight_layout()
    plt.savefig(output_file, dpi=500)
    plt.close()

    print(f"✅ MODIS map saved in : {output_file}")


def plot_temp_mean_Kelvin(input_dir="data/processed/modis",
                          csv_dir="data/processed/modis/mean_temp_2005_Kelvin.csv",
                          output_file="outputs/modis/mean_temp_2005_Kelvin.png"):
    lon_min, lat_min, lon_max, lat_max = -12.984, 35.290, 38.018, 64.090
    res = 0.25

    if os.path.exists(csv_dir):
        print(f"⏭️ CSV already exists, direct reading : {csv_dir}")
        modis_mean_df = pd.read_csv(csv_dir)
    else:
        lat_bins = np.arange(lat_min, lat_max + res, res)
        lon_bins = np.arange(lon_min, lon_max + res, res)
        sum_dict = {}
        count_dict = {}

        modis_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".csv")])
        print("📥 MODIS files processing...")

        for file in tqdm(modis_files):
            path = os.path.join(input_dir, file)
            try:
                df = pd.read_csv(path, usecols=["lat", "lon", "LST_Kelvin"])
                df.dropna(inplace=True)
                df = df[(df["lat"] >= lat_min) & (df["lat"] <= lat_max) &
                        (df["lon"] >= lon_min) & (df["lon"] <= lon_max)]
                df["lat_bin"] = (df["lat"] // res) * res
                df["lon_bin"] = (df["lon"] // res) * res
                grouped = df.groupby(["lat_bin", "lon_bin"])["LST_Kelvin"].agg(["sum", "count"]).reset_index()

                for _, row in grouped.iterrows():
                    key = (row["lat_bin"], row["lon_bin"])
                    sum_dict[key] = sum_dict.get(key, 0) + row["sum"]
                    count_dict[key] = count_dict.get(key, 0) + row["count"]
            except Exception as e:
                print(f"⚠️ Error with {file} : {e}")

        mean_data = [[lat, lon, sum_dict[(lat, lon)] / count_dict[(lat, lon)]]
                     for (lat, lon) in sum_dict]
        modis_mean_df = pd.DataFrame(mean_data, columns=["lat", "lon", "LST_Kelvin_mean"])
        os.makedirs(os.path.dirname(csv_dir), exist_ok=True)
        modis_mean_df.to_csv(csv_dir, index=False)
        print(f"✅ MODIS average saved in : {csv_dir}")

    # 🗺️ Map generation with contours
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")

    lat_vals = np.sort(modis_mean_df["lat"].unique())
    lon_vals = np.sort(modis_mean_df["lon"].unique())
    lat_vals_centered = lat_vals + res / 2
    lon_vals_centered = lon_vals + res / 2

    Z = modis_mean_df.pivot(index="lat", columns="lon", values="LST_Kelvin_mean").values

    img = ax.imshow(Z,origin="lower", cmap="hot",extent=[lon_vals_centered.min(), lon_vals_centered.max(), lat_vals_centered.min(), lat_vals_centered.max()],transform=ccrs.PlateCarree())
    
    plt.colorbar(img, ax=ax, label="Average MODIS temperature (K)", orientation="vertical", shrink=0.7, pad=0.05)
    plt.title("MODIS mean temperature - 2005 (K)")
    plt.tight_layout()
    plt.savefig(output_file, dpi=600)
    plt.close()
    print(f"🖼️ Map (°K) saved at : {output_file}")


    
def plot_temp_mean_Celsius(input_dir="data/processed/modis",
                           csv_dir="data/processed/modis/mean_temp_2005_Celsius.csv",
                           output_file="outputs/modis/mean_temp_2005_Celsius.png"):
    lon_min, lat_min, lon_max, lat_max = -12.984, 35.290, 38.018, 64.090
    res = 0.25

    if os.path.exists(csv_dir):
        print(f"⏭️ CSV Celsius already exists, direct reading : {csv_dir}")
        modis_mean_df = pd.read_csv(csv_dir)
    else:
        sum_dict = {}
        count_dict = {}

        modis_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".csv") and "mean_temp" not in f])
        print("📥 MODIS file processing for mean temperature in °C...")

        for file in tqdm(modis_files):
            path = os.path.join(input_dir, file)
            try:
                df = pd.read_csv(path, usecols=["lat", "lon", "LST_Kelvin"])
                df.dropna(inplace=True)
                df = df[(df["lat"] >= lat_min) & (df["lat"] <= lat_max) &
                        (df["lon"] >= lon_min) & (df["lon"] <= lon_max)]
                df["lat_bin"] = (df["lat"] // res) * res
                df["lon_bin"] = (df["lon"] // res) * res
                grouped = df.groupby(["lat_bin", "lon_bin"])["LST_Kelvin"].agg(["sum", "count"]).reset_index()

                for _, row in grouped.iterrows():
                    key = (row["lat_bin"], row["lon_bin"])
                    sum_dict[key] = sum_dict.get(key, 0) + row["sum"]
                    count_dict[key] = count_dict.get(key, 0) + row["count"]
            except Exception as e:
                print(f"⚠️ Error with {file} : {e}")

        mean_data = [[lat, lon, (sum_dict[(lat, lon)] / count_dict[(lat, lon)]) - 273.15]
                     for (lat, lon) in sum_dict]
        modis_mean_df = pd.DataFrame(mean_data, columns=["lat", "lon", "LST_Celsius_mean"])
        os.makedirs(os.path.dirname(csv_dir), exist_ok=True)
        modis_mean_df.to_csv(csv_dir, index=False)
        print(f"✅ MODIS average (°C) saved : {csv_dir}")

    # 🗺️ Contoured map
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")

    lat_vals = np.sort(modis_mean_df["lat"].unique())
    lon_vals = np.sort(modis_mean_df["lon"].unique())
    lat_centered = lat_vals + res / 2
    lon_centered = lon_vals + res / 2

    Z = modis_mean_df.pivot(index="lat", columns="lon", values="LST_Celsius_mean").values

    img = ax.imshow(Z,
                    origin="lower",
                    cmap="hot",
                    extent=[lon_centered.min(), lon_centered.max(), lat_centered.min(), lat_centered.max()],
                    transform=ccrs.PlateCarree())

    plt.colorbar(img, ax=ax, label="MODIS mean temperature (°C)", orientation="vertical", shrink=0.7, pad=0.05)
    plt.title("MODIS mean temperature - 2005 (°C)")
    plt.tight_layout()
    plt.savefig(output_file, dpi=600)
    plt.close()
    print(f"🖼️ Map (°C) saved at : {output_file}")