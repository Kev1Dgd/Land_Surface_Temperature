import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

def load_european_fluxnet_data(fluxnet_csv_path, station_coords_path):

    stations_df = pd.read_csv(station_coords_path)
    df_fluxnet = pd.read_csv(fluxnet_csv_path, sep=";")
    
    # Only European stations
    stations_df = stations_df[stations_df["station"].str.startswith("FLX_")].copy()
    european_stations = stations_df["station"].tolist()

    df_fluxnet = df_fluxnet[["TIMESTAMP_START"] + european_stations]
    df_fluxnet["date"] = pd.to_datetime(df_fluxnet["TIMESTAMP_START"], dayfirst=True)
    df_fluxnet.drop(columns=["TIMESTAMP_START"], inplace=True)

    print("✅ FLUXNET data loaded.")
    print(f"📌 European stations detected : {european_stations}")
    print(f"📅 Date range : {df_fluxnet['date'].min().date()} → {df_fluxnet['date'].max().date()}")

    return df_fluxnet, european_stations


def match_fluxnet_with_amsre(df_fluxnet, european_stations, station_coords_path, amsre_data_path):

    print("🔍 FLUXNET ↔ AMSR-E pairing in progress...")

    stations_df = pd.read_csv(station_coords_path)
    stations_df = stations_df[stations_df["station"].isin(european_stations)].copy()

    df_amsre = pd.read_csv(amsre_data_path, parse_dates=["date"])
    df_amsre = df_amsre.dropna(subset=["brightness_temp_37v"])  

    df_amsre["lat_rounded"] = df_amsre["lat"].round(2)
    df_amsre["lon_rounded"] = df_amsre["lon"].round(2)
    tree = cKDTree(df_amsre[["lat_rounded", "lon_rounded"]].drop_duplicates().values)

    matched_data = []

    for station in european_stations:
        station_row = stations_df[stations_df["station"] == station].iloc[0]
        lat, lon = station_row["lat"], station_row["lon"]

        # Find nearest pixel
        dist, idx = tree.query([lat, lon], k=1)
        nearest_coords = tree.data[idx]
        lat_match, lon_match = nearest_coords

        # Extract AMSR-E time series at found pixel
        subset = df_amsre[
            (df_amsre["lat_rounded"] == lat_match) &
            (df_amsre["lon_rounded"] == lon_match)
        ].copy()
        subset["station"] = station

        # Join with FLUXNET temperatures
        df_flux = df_fluxnet[["date", station]].rename(columns={station: "temp_fluxnet"})
        merged = pd.merge(subset, df_flux, on="date", how="inner")

        matched_data.append(merged)

        print(f"✅ Matching performed for {station} at ({lat_match}, {lon_match}) - {len(merged)} points")

    # Concaténer tous les résultats
    df_matched = pd.concat(matched_data, ignore_index=True)

    print(f"\n📦 Total matched data : {len(df_matched)} lines.")
    return df_matched


def plot_station_comparison(df, station, tb_col, output_dir):
    fluxnet_col = "temp_fluxnet"  

    df_sta = df[df["station"] == station].copy()

    # Conversion of columns to numeric, error handling
    df_sta[fluxnet_col] = pd.to_numeric(df_sta[fluxnet_col], errors='coerce')
    df_sta[tb_col] = pd.to_numeric(df_sta[tb_col], errors='coerce')

    # Filter values within a realistic range (200 K to 350 K)
    valid_mask = (df_sta[fluxnet_col] >= 200) & (df_sta[fluxnet_col] <= 350) & \
                 (df_sta[tb_col] >= 200) & (df_sta[tb_col] <= 350)
    df_sta = df_sta[valid_mask]

    if df_sta.empty:
        print(f"⚠️ No valid data for station {station} and column {tb_col}.")
        return

    rmse = np.sqrt(mean_squared_error(df_sta[fluxnet_col], df_sta[tb_col]))

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Séries temporelles
    axs[0].plot(df_sta["date"], df_sta[fluxnet_col], label="FLUXNET")
    axs[0].plot(df_sta["date"], df_sta[tb_col], label=f"TB AMSR-E ({tb_col})")
    axs[0].set_title(f"Time comparison - {station}")
    axs[0].set_ylabel("Temperature (K)")
    axs[0].legend()
    axs[0].grid(True)

    # Scatter plot
    axs[1].scatter(df_sta[fluxnet_col], df_sta[tb_col], alpha=0.6)
    axs[1].plot([200, 350], [200, 350], 'r--')
    axs[1].set_xlabel("Temp FLUXNET (K)")
    axs[1].set_ylabel(f"TB AMSR-E ({tb_col}) (K)")
    axs[1].set_title(f"RMSE: {rmse:.2f} K")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    station_folder = station.split("_FLUXNET")[0]
    output_path = os.path.join(output_dir, station_folder)
    os.makedirs(output_path, exist_ok=True)
    filename = f"{station}_{tb_col}.png".replace(" ", "_")
    plt.savefig(os.path.join(output_path, filename))
    plt.close()
    print(f"✅ Saved graphics : {filename}")


def batch_plot_all_stations(df_matched, stations, tb_columns, output_dir="outputs/fluxnet_vs_amsre"):

    for station in stations:
        for tb_col in tb_columns:
            print(f"\n📊 Visualization for station {station} - {tb_col}")
            plot_station_comparison(df=df_matched, station=station, tb_col=tb_col, output_dir=output_dir)
