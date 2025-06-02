import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import os
from tqdm import tqdm


def plot_bt_map(df, date, pass_type, freq_label, title=None, cmap="viridis", output_dir="outputs/amsre/dates/"):
    print(f"🗺️ Generation of the map for pass_type = {pass_type}...")

    # If you want the combined card (all the files)
    if pass_type == "combined":
        df_filtered = df  # No filtering by type of passage
    else:
        df_filtered = df[df["pass_type"] == pass_type]  # Filtering by specific passage

    df_filtered = df_filtered.copy()
    df_filtered.loc[:, "lat_bin"] = df_filtered["latitude"].round(4)
    df_filtered.loc[:, "lon_bin"] = df_filtered["longitude"].round(4)

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

    if pass_type == "combined":
        output_file = os.path.join(date_output_dir,"brightness_temperature", freq_label, f"tb_{freq_label}_map_{date}.png")
    else:
        output_file = os.path.join(date_output_dir,"brightness_temperature", freq_label, f"tb_{freq_label}_map_{date}_{pass_type}.png")

    plt.tight_layout()
    plt.savefig(output_file, dpi=600)
    plt.close()
    del df_filtered, df_grouped, fig, ax, scatter
    print(f"✅ Map saved in {output_file}")


def plot_temp_estimated_map(df, date, pass_type, freq_label, a, b, cmap="viridis", output_dir="outputs/amsre/dates"):
    print(f"🗺️ Génération de la carte de température estimée pour pass_type = {pass_type}...")

    # 🔍 Filtrage selon le type de passage
    if pass_type != "combined":
        df = df[df["pass_type"] == pass_type]

    # 🎯 Estimation de la température (sans regroupement en grille)
    brightness_column = f"brightness_temp_{freq_label[:2]}v"
    df = df.assign(
        estimated_temp=a * df[brightness_column] + b
    )

    # 💾 Chemin du CSV
    csv_dir = os.path.join("data/processed/amsre", date)
    output_csv_path = os.path.join(csv_dir, f"amsre_calculated_temp_reg_{date}_{freq_label}.csv")

    # ✅ Sauvegarde CSV seulement s’il n’existe pas déjà
    if pass_type == "combined":
        if os.path.exists(output_csv_path):
            print(f"⏩ CSV déjà existant, aucune sauvegarde : {output_csv_path}")
        else:
            os.makedirs(csv_dir, exist_ok=True)
            df[["latitude", "longitude", brightness_column, "estimated_temp"]].to_csv(output_csv_path, index=False)
            print(f"✅ CSV sauvegardé : {output_csv_path}")

    # 🗺️ Affichage de la carte
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    sc = ax.scatter(
        df["longitude"],
        df["latitude"],
        c=df["estimated_temp"],
        cmap=cmap,
        s=10,
        transform=ccrs.PlateCarree()
    )

    plt.colorbar(sc, ax=ax, orientation='vertical', label='Température estimée (°K)')
    ax.set_title(f"Température estimée ({freq_label}) - {pass_type} - {date}")

    output_path = os.path.join(output_dir, date, "estimated_temperature", freq_label, f"temp_by_reg_{freq_label}_map_{date}_{pass_type}.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=500)
    plt.close()

    print(f"✅ Carte sauvegardée : {output_path}")


def plot_temp_mean_amsre_Kelvin(
    freq_label,
    input_dir="data/processed/amsre",
    csv_out="data/processed/amsre/mean_temp_2005_{}_Kelvin.csv",
    map_out="outputs/amsre/mean_temp_2005_{}_Kelvin.png",
    temp_column="estimated_temp"
):
    freq_label = str(freq_label)
    csv_out = csv_out.format(freq_label)
    map_out = map_out.format(freq_label)

    lon_min, lat_min, lon_max, lat_max = -12.984, 35.290, 38.018, 64.090
    res = 0.25

    if os.path.exists(csv_out):
        print(f"⏭️ Average CSV already exists : {csv_out}")
        df_mean = pd.read_csv(csv_out)
    else:
        print(f"\n📊 Calculation of average AMSRE temperature ({freq_label}) à partir des fichiers estimés...")
        sum_dict, count_dict = {}, {}

        for date_folder in tqdm(sorted(os.listdir(input_dir))):
            file_path = os.path.join(input_dir, date_folder, f"amsre_calculated_temp_reg_{date_folder}_{freq_label}ghz.csv")
            if not os.path.exists(file_path):
                continue

            try:
                df = pd.read_csv(file_path, usecols=["latitude", "longitude", temp_column])
                df.dropna(inplace=True)

                df = df[
                    (df["latitude"] >= lat_min) & (df["latitude"] <= lat_max) &
                    (df["longitude"] >= lon_min) & (df["longitude"] <= lon_max)
                ]

                df["lat_bin"] = (df["latitude"] // res) * res
                df["lon_bin"] = (df["longitude"] // res) * res

                grouped = df.groupby(["lat_bin", "lon_bin"])[temp_column].agg(["sum", "count"]).reset_index()
                for _, g in grouped.iterrows():
                    key = (g["lat_bin"], g["lon_bin"])
                    sum_dict[key] = sum_dict.get(key, 0) + g["sum"]
                    count_dict[key] = count_dict.get(key, 0) + g["count"]

            except Exception as e:
                print(f"⚠️ File error in {file_path} : {e}")

        mean_data = [[lat, lon, sum_dict[(lat, lon)] / count_dict[(lat, lon)]] for (lat, lon) in sum_dict]
        df_mean = pd.DataFrame(mean_data, columns=["lat", "lon", "temp_K_mean"])
        os.makedirs(os.path.dirname(csv_out), exist_ok=True)
        df_mean.to_csv(csv_out, index=False)
        print(f"✅ Kelvin csv saved in : {csv_out}")

    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    pivot = df_mean.pivot(index="lat", columns="lon", values="temp_K_mean")
    lons = pivot.columns.values
    lats = pivot.index.values
    mesh = ax.pcolormesh(lons, lats, pivot.values, cmap="hot", shading="auto", transform=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)

    plt.colorbar(mesh, label=f"Température AMSRE {freq_label}GHz (K)", orientation="vertical", shrink=0.7, pad=0.05)
    plt.title(f"Mean temperature AMSRE {freq_label}GHz - 2005 (K)")
    plt.tight_layout()

    os.makedirs(os.path.dirname(map_out), exist_ok=True)
    plt.savefig(map_out, dpi=600)
    plt.close()
    print(f"🖼️ Kelvin map saved in : {map_out}")


def plot_temp_mean_amsre_Celsius(
    freq="19GHz",
    csv_kelvin="data/processed/amsre/mean_temp_2005_{}_Kelvin.csv",
    csv_out="data/processed/amsre/mean_temp_2005_{}_Celsius.csv",
    map_out="outputs/amsre/mean_temp_2005_{}_Celsius.png"
):

    freq_label = freq.replace("GHz", "")
    csv_kelvin = csv_kelvin.format(freq_label)
    csv_out = csv_out.format(freq_label)
    map_out = map_out.format(freq_label)

    lon_min, lat_min, lon_max, lat_max = -12.984, 35.290, 38.018, 64.090

    if os.path.exists(csv_out):
        print(f"⏭️ Celsius csv already exists: {csv_out}")
        df = pd.read_csv(csv_out)
    else:
        df = pd.read_csv(csv_kelvin)
        df["temp_C_mean"] = df["temp_K_mean"] - 273.15
        df.drop(columns="temp_K_mean", inplace=True)
        os.makedirs(os.path.dirname(csv_out), exist_ok=True)
        df.to_csv(csv_out, index=False)
        print(f"✅ Celsius csv saved in : {csv_out}")

    # 🌍 Carte avec Cartopy
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    pivot = df.pivot(index="lat", columns="lon", values="temp_C_mean")
    lons = pivot.columns.values
    lats = pivot.index.values
    mesh = ax.pcolormesh(lons, lats, pivot.values, cmap="coolwarm", shading="auto", transform=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)

    plt.colorbar(mesh, label=f"Temperature AMSRE {freq} (°C)", orientation="vertical", shrink=0.7, pad=0.05)
    plt.title(f"Mean temperature AMSRE {freq} - 2005 (°C)")
    plt.tight_layout()

    os.makedirs(os.path.dirname(map_out), exist_ok=True)
    plt.savefig(map_out, dpi=600)
    plt.close()
    print(f"🖼️ Celsius map saved in : {map_out}")
