import os
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm import tqdm    
    

def plot_seasonal_temp_evolution(csv_path, output_dir="outputs/fluxnet/seasonal_evolution"):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path, sep=";")

    # Conversion to long format
    df_long = df.melt(id_vars=["TIMESTAMP_START"], var_name="station", value_name="temperature")

    # Date conversion
    df_long["TIMESTAMP_START"] = pd.to_datetime(df_long["TIMESTAMP_START"], format="%d/%m/%Y")

    # Calculating the day of the year
    df_long["DOY"] = df_long["TIMESTAMP_START"].dt.dayofyear

    # Temperature cleaning
    df_long["temperature"] = pd.to_numeric(df_long["temperature"], errors="coerce")
    df_long = df_long.dropna(subset=["temperature"])
    df_long = df_long[(df_long["temperature"] > 180) & (df_long["temperature"] < 330)]

    # Loop on stations
    for station in df_long["station"].unique():
        df_station = df_long[df_long["station"] == station]

        if len(df_station) < 10:
            continue

        plt.figure(figsize=(10, 5))
        plt.scatter(df_station["DOY"], df_station["temperature"], alpha=0.5, label="Température in-situ")

        # Add AMSR-E temperature if available
        if "brightness_temp_37v" in df.columns:
            df_amsre = df[["TIMESTAMP_START", "brightness_temp_37v"]].copy()
            df_amsre["TIMESTAMP_START"] = pd.to_datetime(df_amsre["TIMESTAMP_START"], format="%d/%m/%Y")
            df_amsre["DOY"] = df_amsre["TIMESTAMP_START"].dt.dayofyear
            df_amsre["brightness_temp_37v"] = pd.to_numeric(df_amsre["brightness_temp_37v"], errors="coerce")
            df_amsre = df_amsre.dropna()

            plt.plot(df_amsre["DOY"], df_amsre["brightness_temp_37v"], color="orange", label="TB AMSR-E", linewidth=1.5)

        plt.title(f"Seasonal trends - {station}")
        plt.xlabel("Day of the year (DOY)")
        plt.ylabel("Température (K)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        filename = f"saison_{station}.png".replace("/", "_")
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()

    print(f"✅ Seasonal graph saved in : {output_dir}")

def plot_seasonal_temp_with_tb_evolution(
    matched_folder1="data/processed/amsre/matched/19GHz/",
    matched_folder2="data/processed/amsre/matched/37GHz/",
    output_dir="outputs/fluxnet/seasonal_temp_tb",
    tb_min_threshold=220  
):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    matched_files1 = sorted([
        os.path.join(matched_folder1, f)
        for f in os.listdir(matched_folder1)
        if f.endswith(".csv") and f.startswith("matched_tb_fluxnet")
    ])
    matched_files2 = sorted([
        os.path.join(matched_folder2, f)
        for f in os.listdir(matched_folder2)
        if f.endswith(".csv") and f.startswith("matched_tb_fluxnet")
    ])

    if not matched_files1 or not matched_files2:
        print("⚠️ Missing correspondence files.")
        return

    df_all_matches1, df_all_matches2 = [], []

    # 19 GHz
    for file in matched_files1:
        date_str = os.path.basename(file).split("_")[-1].split(".")[0]
        try:
            date_obj = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            print(f"❌ Wrong date format : {file}")
            continue

        df = pd.read_csv(file).rename(columns={
            "brightness_temp_19v": "tb19",
            "temperature": "temp"
        })

        if "tb19" not in df.columns or "temp" not in df.columns:
            print(f"❌ Missing columns in {file}")
            continue

        df["date"] = date_obj
        df = df[["date", "tb19", "temp"]].dropna()

        # Apply temperature filter + TB
        df = df[(df["temp"] > 180) & (df["temp"] < 330) & (df["tb19"] >= tb_min_threshold)]

        df_all_matches1.append(df)

    if not df_all_matches1:
        print("❌ No valid data for 19 GHz.")
        return

    # 37 GHz
    for file in matched_files2:
        date_str = os.path.basename(file).split("_")[-1].split(".")[0]
        try:
            date_obj = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            print(f"❌ Wrong date format : {file}")
            continue

        df = pd.read_csv(file).rename(columns={
            "brightness_temp_37v": "tb37",
            "temperature": "temp"
        })

        if "tb37" not in df.columns or "temp" not in df.columns:
            print(f"❌ Missing columns in {file}")
            continue

        df["date"] = date_obj
        df = df[["date", "tb37", "temp"]].dropna()

        # Apply temperature filter + TB
        df = df[(df["temp"] > 180) & (df["temp"] < 330) & (df["tb37"] >= tb_min_threshold)]

        df_all_matches2.append(df)

    if not df_all_matches2:
        print("❌ No valid data for 37 GHz.")
        return

    # Merger + Aggregation
    df_all1 = pd.concat(df_all_matches1)
    df_all2 = pd.concat(df_all_matches2)

    df_grouped1 = df_all1.groupby("date").agg({"temp": "mean", "tb19": "mean"}).reset_index()
    df_grouped2 = df_all2.groupby("date").agg({"temp": "mean", "tb37": "mean"}).reset_index()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_grouped1["date"], df_grouped1["temp"], label="Température in-situ (°K)", color="tomato")
    plt.plot(df_grouped1["date"], df_grouped1["tb19"], label="TB 19GHz AMSR-E (°K)", color="royalblue")
    plt.plot(df_grouped2["date"], df_grouped2["tb37"], label="TB 37GHz AMSR-E (°K)", color="cyan")

    plt.xlabel("Date")
    plt.ylabel("Température (K)")
    plt.title("Seasonal trends: FLUXNET vs AMSR-E TB")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_file = os.path.join(output_dir, "evolution_temp_tb.png")
    plt.savefig(output_file)
    plt.close()

    print(f"✅ Plot saved in : {output_file}")


def plot_all_stations_temp_evolution(csv_path, output_path="outputs/fluxnet/seasonal_evolution/temp_by_station.png"):
    #  Reading the CSV file
    df = pd.read_csv(csv_path, sep=";")
    
    # Conversion to long format
    df_long = df.melt(id_vars=["TIMESTAMP_START"], var_name="station", value_name="temperature")
    
    # Date conversion
    df_long["TIMESTAMP_START"] = pd.to_datetime(df_long["TIMESTAMP_START"], format="%d/%m/%Y")
    
    # Temperature cleaning
    df_long["temperature"] = pd.to_numeric(df_long["temperature"], errors="coerce")
    df_long = df_long.dropna(subset=["temperature"])
    df_long = df_long[(df_long["temperature"] > 180) & (df_long["temperature"] < 330)]
    
    # Plot
    plt.figure(figsize=(14, 7))

    for station in df_long["station"].unique():
        df_station = df_long[df_long["station"] == station]
        if len(df_station) < 10:
            continue
        plt.plot(df_station["TIMESTAMP_START"], df_station["temperature"], label=station, alpha=0.7)

    plt.title("Time trend in temperature by station")
    plt.xlabel("Date")
    plt.ylabel("Température (K)")
    plt.legend(loc="upper right", fontsize="small", ncol=2)
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ Plot saved in : {output_path}")


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
