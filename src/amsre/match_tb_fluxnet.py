import pandas as pd
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