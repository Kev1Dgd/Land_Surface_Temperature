import os
import pandas as pd
from datetime import datetime, timedelta
from src.amsre.match_tb_fluxnet import match_tb_with_fluxnet

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