import os
import pandas as pd
from datetime import timedelta
from geopy.distance import geodesic

def match_tb_with_fluxnet(df_fluxnet, df_tb, df_coords, output_path):
    results = []

    for station_name in df_fluxnet.columns[1:]:  # Skip the date column
        if station_name not in df_coords["station"].values:
            print(f"❌ Missing coordinates details for {station_name}")
            continue

        # Coordinates
        station_info = df_coords[df_coords["station"] == station_name].iloc[0]
        lat, lon = station_info["lat"], station_info["lon"]

        # Average TB near the station
        tb_near = df_tb[
            (df_tb["latitude"].between(lat - 1, lat + 1)) &
            (df_tb["longitude"].between(lon - 1, lon + 1))
        ]

        tb_mean = tb_near["brightness_temp_37v"].mean()

        # Cleaning and temperature validation
        temp_raw = df_fluxnet[station_name].values[0]
        try:
            # Remove superfluous points
            cleaned = str(temp_raw).replace('.', '', str(temp_raw).count('.') - 1)
            temp = float(cleaned)
        except:
            print(f"⚠️ Illegible temperature ignored for {station_name} : {temp_raw}")
            continue

        # Realistic range check in Kelvin
        if not (180 <= temp <= 330):
            print(f"⚠️ Temperature outside realistic range ignored for {station_name} : {temp} K")
            continue

        results.append({
            "station": station_name,
            "latitude": lat,
            "longitude": lon,
            "brightness_temp_37v": tb_mean,
            "temperature": temp
        })

    df_result = pd.DataFrame(results)
    df_result.to_csv(output_path, index=False)
    print(f"✅ Results exported to : {output_path}")


def generate_daily_matches(start_date, end_date, fluxnet_path, coords_path, tb_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load FLUXNET once
    df_fluxnet_all = pd.read_csv(fluxnet_path, sep=';')
    df_fluxnet_all["TIMESTAMP_START"] = pd.to_datetime(df_fluxnet_all["TIMESTAMP_START"], format="%d/%m/%Y")

    # Load contact details only once
    df_coords = pd.read_csv(coords_path)

    # Day-by-day loop
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        file_suffix = current_date.strftime("%Y%m%d")

        print(f"\n📅 Treatment of {date_str}...")

        # Extract data for the current day
        df_fluxnet_day = df_fluxnet_all[df_fluxnet_all["TIMESTAMP_START"].dt.date == current_date.date()]
        if df_fluxnet_day.empty:
            print(f"❌ No FLUXNET data for {date_str}")
            current_date += timedelta(days=1)
            continue

        # Load the corresponding TB file
        
        tb_file = os.path.join(tb_folder, f"amsre_combined_37GHz_{date_str}_descending.csv")
        
        if not os.path.exists(tb_file):
            print(f"⚠️ Missing TB file for {date_str} : {tb_file}")
            current_date += timedelta(days=1)
            continue

        # Load and match data
        df_tb = pd.read_csv(tb_file)
        output_csv = os.path.join(output_folder, f"matched_tb_fluxnet_{file_suffix}.csv")
        match_tb_with_fluxnet(df_fluxnet_day, df_tb, df_coords, output_csv)


        current_date += timedelta(days=1)