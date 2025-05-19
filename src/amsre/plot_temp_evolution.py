import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

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

        # 🔄 Add AMSR-E temperature if available
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

def plot_seasonal_temp_with_tb_evolution(matched_folder1="data/processed/amsre/matched/19GHz/",matched_folder2 = "data/processed/amsre/matched/37GHz/",output_dir="outputs/fluxnet/seasonal_temp_tb"):
    
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

    if not matched_files1:
        print("No correspondence files found for 1.")
        return
    
    if not matched_files2:
        print("No correspondence files found for 2.")
        return

    df_all_matches1,df_all_matches2 = [],[]

    for file in matched_files1:
        # Extract the date from the file name
        date_str = os.path.basename(file).split("_")[-1].split(".")[0]
        try:
            date_obj = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            print(f"❌ Wrong date format in the file : {file}")
            continue

        df_match = pd.read_csv(file)

        # Rename the columns to match our logic
        df_match = df_match.rename(columns={
            "brightness_temp_19v": "tb19",
            "temperature": "temp"
        })

        if "tb19" not in df_match or "temp" not in df_match:
            print(f"❌ Necessary columns missing from the file : {file}")
            continue

        df_match["date"] = date_obj
        df_match = df_match[["date", "tb19", "temp"]].dropna()

        # Cleaning: filtering out aberrant temperatures
        df_match = df_match[(df_match["temp"] > 180) & (df_match["temp"] < 330)]
        df_all_matches1.append(df_match)

    if not df_all_matches1:
        print("No valid data to trace for 1.")
        return
    
    for file in matched_files2:
        # Extract the date from the file name
        date_str = os.path.basename(file).split("_")[-1].split(".")[0]
        try:
            date_obj = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            print(f"❌ Wrong date format in the file : {file}")
            continue

        df_match = pd.read_csv(file)

        # Rename the columns to match our logic
        df_match = df_match.rename(columns={
            "brightness_temp_37v": "tb37",
            "temperature": "temp"
        })

        if "tb37" not in df_match or "temp" not in df_match:
            print(f"❌ Necessary columns missing from the file : {file}")
            continue

        df_match["date"] = date_obj
        df_match = df_match[["date", "tb37", "temp"]].dropna()

        # Cleaning: filtering out aberrant temperatures
        df_match = df_match[(df_match["temp"] > 180) & (df_match["temp"] < 330)]
        df_all_matches2.append(df_match)

    if not df_all_matches2:
        print("No valid data to trace for 2.")
        return

    df_all1 = pd.concat(df_all_matches1)
    df_all2 = pd.concat(df_all_matches2)

    # Daily average
    df_grouped1 = df_all1.groupby("date").agg({"temp": "mean", "tb19": "mean"}).reset_index()
    df_grouped2 = df_all2.groupby("date").agg({"temp": "mean", "tb37": "mean"}).reset_index()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_grouped1["date"], df_grouped1["temp"], label="Température in-situ (°K)", color="tomato")
    plt.plot(df_grouped1["date"], df_grouped1["tb19"], label="Température de brillance 37GHz AMSR-E (°K)", color="royalblue")
    plt.plot(df_grouped2["date"], df_grouped1["tb37"], label="Température de brillance 37GHz AMSR-E (°K)", color="cyan")

    plt.xlabel("Date")
    plt.ylabel("Température (K)")
    plt.title("Seasonal trends: FLUXNET temperature vs AMSR-E TB 37GHz")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_file = os.path.join(output_dir, "evolution_temp_tb.png")
    plt.savefig(output_file)
    plt.close()

    print(f"✅ Graph saved in : {output_file}")

def plot_all_stations_temp_evolution(csv_path, output_path="outputs/fluxnet/seasonal_evolution/temp_by_station.png"):
    # Lecture du fichier CSV
    df = pd.read_csv(csv_path, sep=";")
    
    # Transformation en format long
    df_long = df.melt(id_vars=["TIMESTAMP_START"], var_name="station", value_name="temperature")
    
    # Conversion des dates
    df_long["TIMESTAMP_START"] = pd.to_datetime(df_long["TIMESTAMP_START"], format="%d/%m/%Y")
    
    # Nettoyage des températures
    df_long["temperature"] = pd.to_numeric(df_long["temperature"], errors="coerce")
    df_long = df_long.dropna(subset=["temperature"])
    df_long = df_long[(df_long["temperature"] > 180) & (df_long["temperature"] < 330)]
    
    # Tracé
    plt.figure(figsize=(14, 7))

    for station in df_long["station"].unique():
        df_station = df_long[df_long["station"] == station]
        if len(df_station) < 10:
            continue
        plt.plot(df_station["TIMESTAMP_START"], df_station["temperature"], label=station, alpha=0.7)

    plt.title("Évolution temporelle de la température par station")
    plt.xlabel("Date")
    plt.ylabel("Température (K)")
    plt.legend(loc="upper right", fontsize="small", ncol=2)
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ Graphe sauvegardé dans : {output_path}")