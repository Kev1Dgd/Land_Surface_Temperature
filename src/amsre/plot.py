import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

def plot_bt_map(df, date, pass_type, freq_label, title=None, cmap="viridis", output_dir="outputs/amsre/dates"):
    print(f"🗺️ Génération de la carte pour pass_type = {pass_type}...")

    # If you want the combined card (all the files)
    if pass_type == "combined":
        df_filtered = df  # No filtering by type of passage
    else:
        df_filtered = df[df["pass_type"] == pass_type]  # Filtering by specific passage

    df_filtered["lat_bin"] = df_filtered["latitude"].round(4)
    df_filtered["lon_bin"] = df_filtered["longitude"].round(4)

    df_grouped = df_filtered.groupby(["lat_bin", "lon_bin"]).agg({
        f"brightness_temp_{freq_label[:2]}v": "mean"
    }).reset_index()

    df_grouped.rename(columns={"lat_bin": "latitude", "lon_bin": "longitude"}, inplace=True)

    # Create a folder by date in outputs/amsre
    date_output_dir = os.path.join(output_dir, date)
    os.makedirs(date_output_dir, exist_ok=True)

    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    vmin = 130
    vmax = 300

    scatter = ax.scatter(
        df_grouped["longitude"], df_grouped["latitude"],
        c=df_grouped[f"brightness_temp_{freq_label[:2]}v"], cmap=cmap, s=10,
        transform=ccrs.PlateCarree(), alpha=0.7,
        vmin=vmin, vmax=vmax
    )

    # If no title specified, define a default title
    if not title:
        if pass_type == "combined":
            title = f"Brightness temperature {freq_label} – {date}"
        else:
            title = f"Brightness temperature {freq_label} – {pass_type} - {date}"

    plt.title(title)
    plt.colorbar(scatter, label=f"TB {freq_label} (K)", orientation="vertical", shrink=0.7, pad=0.05)
    ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)

    # Nom de fichier sans 'ascending' ou 'descending' si c'est combiné
    if pass_type == "combined":
        output_file = os.path.join(date_output_dir, f"tb_{freq_label}_map_{date}.png")
    else:
        output_file = os.path.join(date_output_dir, f"tb_{freq_label}_map_{date}_{pass_type}.png")

    plt.tight_layout()
    plt.savefig(output_file, dpi=600)
    plt.close()

    print(f"✅ Map saved in {output_file}")

def plot_temp_estimated_map(df, date, pass_type, freq_label, a, b, cmap="viridis", output_dir="outputs/amsre/dates"):
    print(f"🗺️ Génération de la carte de température estimée pour pass_type = {pass_type}...")

    if pass_type == "combined":
        df_filtered = df
    else:
        df_filtered = df[df["pass_type"] == pass_type]

    df_filtered["lat_bin"] = df_filtered["latitude"].round(4)
    df_filtered["lon_bin"] = df_filtered["longitude"].round(4)

    # Appliquer la régression
    df_filtered["estimated_temp"] = a * df_filtered[f"brightness_temp_{freq_label[:2]}v"] + b -273.15

    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    sc = ax.scatter(df_filtered["lon_bin"], df_filtered["lat_bin"], c=df_filtered["estimated_temp"],
                    cmap=cmap, s=10, transform=ccrs.PlateCarree())

    plt.colorbar(sc, ax=ax, orientation='vertical', label='Température estimée (°C)')
    title = f"Température estimée ({freq_label}) - {pass_type} - {date}"
    plt.title(title)

    os.makedirs(f"{output_dir}/{date}", exist_ok=True)
    output_path = f"{output_dir}/{date}/temp_by_reg_{freq_label}_map_{date}_{pass_type}.png"
    plt.savefig(output_path, dpi=500)
    plt.close()
    print(f"✅ Carte de température estimée sauvegardée : {output_path}")
