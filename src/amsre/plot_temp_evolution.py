import os
import pandas as pd
import matplotlib.pyplot as plt 
from datetime import datetime
    

def plot_seasonal_temp_with_tb_evolution(matched_folder1, matched_folder2, output_dir, tb_min_threshold, freq_label):

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

    # Vertical polarization
    for file in matched_files1:
        date_str = os.path.basename(file).split("_")[-1].split(".")[0]
        try:
            date_obj = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            print(f"❌ Wrong date format : {file}")
            continue

        df = pd.read_csv(file).rename(columns={
            f"brightness_temp_{freq_label[:2]}v": f"tb{freq_label[:2]}v",
            "temperature": "temp"
        })

        if f"tb{freq_label[:2]}v" not in df.columns or "temp" not in df.columns:
            print(f"❌ Missing columns in {file}")
            continue

        df["date"] = date_obj
        df = df[["date", f"tb{freq_label[:2]}v", "temp"]].dropna()

        # Apply LST & TB filter 
        df = df[(df["temp"] > 180) & (df["temp"] < 330) & (df[f"tb{freq_label[:2]}v"] >= tb_min_threshold)]

        df_all_matches1.append(df)

    if not df_all_matches1:
        print(f"❌ No valid data for {freq_label} - vertical.")
        return

    # Horizontal polarization
    for file in matched_files2:
        date_str = os.path.basename(file).split("_")[-1].split(".")[0]
        try:
            date_obj = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            print(f"❌ Wrong date format : {file}")
            continue

        df = pd.read_csv(file).rename(columns={
            f"brightness_temp_{freq_label[:2]}h": f"tb{freq_label[:2]}h",
            "temperature": "temp"
        })

        if f"tb{freq_label[:2]}h" not in df.columns or "temp" not in df.columns:
            print(f"❌ Missing columns in {file}")
            continue

        df["date"] = date_obj
        df = df[["date", f"tb{freq_label[:2]}h", "temp"]].dropna()

        # Apply temperature filter + TB
        df = df[(df["temp"] > 180) & (df["temp"] < 330) & (df[f"tb{freq_label[:2]}h"] >= tb_min_threshold)]

        df_all_matches2.append(df)

    if not df_all_matches2:
        print(f"❌ No valid data for {freq_label} - horizontal.")
        return

    # Merger + Aggregation
    df_all1 = pd.concat(df_all_matches1)
    df_all2 = pd.concat(df_all_matches2)

    df_grouped1 = df_all1.groupby("date").agg({"temp": "mean", f"tb{freq_label[:2]}v" : "mean"}).reset_index()
    df_grouped2 = df_all2.groupby("date").agg({"temp": "mean", f"tb{freq_label[:2]}h" : "mean"}).reset_index()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_grouped1["date"], df_grouped1["temp"], label="In-situ temperature (°K)", color="tomato")
    plt.plot(df_grouped1["date"], df_grouped1[f"tb{freq_label[:2]}v"], label=f"TB {freq_label} vertical AMSR-E (°K)", color="royalblue")
    plt.plot(df_grouped2["date"], df_grouped2[f"tb{freq_label[:2]}h"], label=f"TB {freq_label} horizontal AMSR-E (°K)", color="cyan")

    plt.xlabel("Date")
    plt.ylabel("Temperature (K)")
    plt.title("Seasonal trends: FLUXNET vs AMSR-E TB")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_file = os.path.join(output_dir, f"evolution_temp_tb_{freq_label}.png")
    plt.savefig(output_file)
    plt.close()

    print(f"✅ Plot saved in : {output_file}")


