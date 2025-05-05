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

def plot_seasonal_temp_with_tb_evolution(matched_folder="data/processed/amsre/matched",output_dir="outputs/fluxnet/seasonal_temp_tb"):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    matched_files = sorted([
        os.path.join(matched_folder, f)
        for f in os.listdir(matched_folder)
        if f.endswith(".csv") and f.startswith("matched_tb_fluxnet")
    ])

    if not matched_files:
        print("No correspondence files found.")
        return

    df_all_matches = []

    for file in matched_files:
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
        df_all_matches.append(df_match)

    if not df_all_matches:
        print("No valid data to trace.")
        return

    df_all = pd.concat(df_all_matches)

    # Daily average
    df_grouped = df_all.groupby("date").agg({"temp": "mean", "tb37": "mean"}).reset_index()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_grouped["date"], df_grouped["temp"], label="Température in-situ (°K)", color="tomato")
    plt.plot(df_grouped["date"], df_grouped["tb37"], label="Température de brillance 37GHz AMSR-E (°K)", color="royalblue")

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