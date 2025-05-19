import os
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from glob import glob

def plot_modis_lst_map(df, date, cmap="coolwarm", output_dir="outputs/modis/dates"):
    output_file = os.path.join(output_dir, date, f"modis_lst_map_{date}.png")
    
    # ✅ NE PAS REGÉNÉRER si la carte existe déjà
    if os.path.exists(output_file):
        print(f"⏭️ Carte déjà existante, saut : {output_file}")
        return

    print(f"🗺️ Génération de la carte MODIS pour {date}...")

    df["lat_bin"] = df["lat"].round(4)
    df["lon_bin"] = df["lon"].round(4)

    df_grouped = df.groupby(["lat_bin", "lon_bin"]).agg({
        "LST_Celsius": "mean"
    }).reset_index()

    # Créer un dossier pour la date
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

    plt.title(f"Température de surface MODIS – {date}")
    plt.colorbar(scatter, label="LST (°C)", orientation="vertical", shrink=0.7, pad=0.05)

    plt.tight_layout()
    plt.savefig(output_file, dpi=500)
    plt.close()

    print(f"✅ Carte MODIS sauvegardée : {output_file}")


def process_all_modis_csv(input_folder="data/processed/modis", output_folder="outputs/modis/dates"):
    csv_files = sorted(glob(os.path.join(input_folder, "*.csv")))
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        date = pd.to_datetime(df["time"].iloc[0]).strftime("%Y-%m-%d")
        plot_modis_lst_map(df, date, output_dir=output_folder)

    
