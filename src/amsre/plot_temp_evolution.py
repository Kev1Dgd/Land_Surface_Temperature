import os
import pandas as pd
import matplotlib.pyplot as plt 
from datetime import datetime
    

def plot_seasonal_temp_with_tb_evolution(
    matched_folder1="data/processed/amsre/matched/19GHz/",
    matched_folder2="data/processed/amsre/matched/37GHz/",
    output_dir="outputs/fluxnet/seasonal_temp_tb",
    tb_min_threshold=220  
):

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

