import os
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

def plot_difference_map_explicit(modis_csv, amsre_csv, modis_col, amsre_col, output_path, title, color_label, cmap="coolwarm"):
    
    # Settings
    res = 0.25  
    lon_min, lat_min, lon_max, lat_max = -12.984, 35.290, 38.018, 64.090

    # Path to cached difference CSV
    diff_csv_path = os.path.join("data", "processed", "differences", os.path.splitext(os.path.basename(output_path))[0] + ".csv")
    os.makedirs(os.path.dirname(diff_csv_path), exist_ok=True)

    # Load cached difference if available
    if os.path.exists(diff_csv_path):
        print(f"📄 Using cached difference data from: {diff_csv_path}")
        merged = pd.read_csv(diff_csv_path).rename(columns={"latitude": "lat", "longitude": "lon"})
    else:
        # Data loading
        df_modis = pd.read_csv(modis_csv)
        df_amsre = pd.read_csv(amsre_csv)

        # Merging and difference computation
        merged = pd.merge(df_modis, df_amsre, on=["lat", "lon"], how="inner")
        merged["diff"] = merged[modis_col] - merged[amsre_col]

        # Export difference CSV
        diff_df = merged.rename(columns={"lat": "latitude", "lon": "longitude"})[["latitude", "longitude", "diff"]]
        diff_df.to_csv(diff_csv_path, index=False)
        print(f"✅ Difference CSV saved at: {diff_csv_path}")

    # Creating explicit grid
    lat_vals = np.sort(merged["lat"].unique())
    lon_vals = np.sort(merged["lon"].unique())
    lat_vals_centered = lat_vals + res / 2
    lon_vals_centered = lon_vals + res / 2

    # Z grid for imshow
    Z = merged.pivot(index="lat", columns="lon", values="diff").values

    # Plot
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")

    img = ax.imshow(Z,
                    origin="lower",
                    cmap=cmap,
                    extent=[
                        lon_vals_centered.min(),
                        lon_vals_centered.max(),
                        lat_vals_centered.min(),
                        lat_vals_centered.max()
                    ],
                    transform=ccrs.PlateCarree())

    plt.colorbar(img, ax=ax, label=color_label, orientation="vertical", shrink=0.7, pad=0.05)
    plt.title(title)
    plt.tight_layout()

    # Save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=600)
    plt.close()
