import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from src.modis.download import authenticate, search_modis_lst, download_results
from src.modis.process import load_modis_data, convert_kelvin_to_celsius, clean_lst_data, save_lst_to_csv

from src.amsre.download import download_amsre_ae_l2a
from src.amsre.plot import plot_bt_map
from src.amsre.process import combine_amsre_files

from src.amsre.plot_brightness_vs_temperature_and_regression import plot_brightness_vs_temperature_and_regression, fit_daily_regressions 
from src.amsre.generate_daily_matches import generate_daily_matches
from src.amsre.plot_regressions import plot_stationwise_and_global_regressions_2005
from src.amsre.plot_temp_evolution import plot_seasonal_temp_evolution, plot_seasonal_temp_with_tb_evolution


def main():
    n_point = 5000
    Sampling = False
    new_graph = False 
    '''
    # Step 1: Authentication with NASA
    auth = authenticate()
    
    # Step 2: Search for MODIS files for 2005 and a specific geographical area
    results = search_modis_lst(
        start_date="2005-01-01",
        end_date="2005-12-31",
        bbox=(-10.0, 35.0, 5.0, 45.0)  # Geographic zone centred on Spain
    )

    if not results:
        print("Aucun fichier trouvé.")
        return

    # Step 3: Downloading MODIS files
    downloaded = download_results(results, output_dir="data/raw/modis")
    if not downloaded:
        print("Aucun fichier téléchargé.")
        return

    # Step 4: Processing MODIS files
    for file_path in downloaded:
        print(f"File processing : {file_path}")
        
        filename = os.path.splitext(os.path.basename(file_path))[0]
        output_file = os.path.join("data/processed/modis", f"{filename}.csv")
        
        # 🛑 Skip si le CSV existe déjà
        if os.path.exists(output_file):
            print(f"⏭️ CSV already in existence, skip : {output_file}")
            continue

        # Load MODIS data (LST)
        lst_data = load_modis_data(file_path)
        if lst_data is None:
            print(f"Unable to load LST data from {file_path}.")
            continue
        
        lst_celsius = convert_kelvin_to_celsius(lst_data)
        lst_cleaned = clean_lst_data(lst_celsius)
        save_lst_to_csv(lst_cleaned, output_file)

    print("Treatment completed.")
    '''

    print("\n===== AMSR-E stage: TB 37GHz processing =====")
    dates = ["2005-01-01", "2005-01-02","2005-01-03","2005-01-04","2005-01-05",]
    for date in dates :
        # Filter only .hdf files
        files = [f for f in download_amsre_ae_l2a(date=date) if f.endswith('.hdf')]

        # Combine the files into two separate CSVs and retrieve the output paths
        output_ascending, output_descending = combine_amsre_files(files, date=date)

        # Load data and generate maps
        if output_ascending and output_descending:
            # Loading renamed files
            df_ascending = pd.read_csv(output_ascending)
            df_descending = pd.read_csv(output_descending)

            # 5000-point sampling
            if Sampling :
                df_ascending_sampled = df_ascending.sample(n=n_point) if len(df_ascending) > 5000 else df_ascending
                df_descending_sampled = df_descending.sample(n=n_point) if len(df_descending) > 5000 else df_descending

                asc_plot_path = f"outputs/amsre/{date}/tb_37ghz_map_{date}_ascending.png"
                des_plot_path = f"outputs/amsre/{date}/tb_37ghz_map_{date}_descending.png"
                comb_plot_path = f"outputs/amsre/{date}/tb_37ghz_map_{date}.png"

                if new_graph or not os.path.exists(asc_plot_path):
                    print("\n📈 Sample visualisation of Ascending")
                    plot_bt_map(df_ascending_sampled, date, pass_type="ascending")
                else : 
                    print("\n✅ Ascending map already generated")

                if new_graph or not os.path.exists(des_plot_path):
                    print("\n📉 Sample visualisation of Descending")
                    plot_bt_map(df_descending_sampled, date, pass_type="descending")
                else : 
                    print("\n✅ Descending map already generated")

                if new_graph or not os.path.exists(comb_plot_path):
                    print("\n📊 Sample visualisation of Combined datas")
                    plot_bt_map(pd.concat([df_ascending_sampled, df_descending_sampled]), date, pass_type="combined")
                else : 
                    print("\n✅  Combined map already generated\n")

            else :  

                asc_plot_path = f"outputs/amsre/{date}/tb_37ghz_map_{date}_ascending.png"
                des_plot_path = f"outputs/amsre/{date}/tb_37ghz_map_{date}_descending.png"
                comb_plot_path = f"outputs/amsre/{date}/tb_37ghz_map_{date}.png"

                if new_graph or not os.path.exists(asc_plot_path):
                    print("\n📈 VSample visualisation of Ascending")
                    plot_bt_map(df_ascending, date, pass_type="ascending")
                else : 
                    print("\n✅ Ascending map already generated")

                if new_graph or not os.path.exists(des_plot_path):
                    print("\n📉 Sample visualisation of Descending")
                    plot_bt_map(df_descending, date, pass_type="descending")
                else : 
                    print("\n✅ Descending map already generated")

                if new_graph or not os.path.exists(comb_plot_path):
                    print("\n📊 Sample visualisation of Combined datas")
                    plot_bt_map(pd.concat([df_ascending, df_descending]), date, pass_type="combined")
                else : 
                    print("\n✅ Combined map already generated\n")

        print(f"Treatment completed for date : {date}\n")


    print("\n===== FLUXNET =====")
    # Parameters
    start_date = datetime(2005, 1, 1)
    end_date = datetime(2005, 1, 5)

    fluxnet_path = "data/raw/fluxnet/FluxNET_AMSRE.csv"
    coords_path = "data/processed/fluxnet/fluxnet_station_coordinates.csv"
    tb_folder = "data/processed/amsre/"
    matched_output_folder = "data/processed/amsre/matched"

    # Load FLUXNET once
    df_fluxnet_all = pd.read_csv(fluxnet_path, sep=';')
    df_fluxnet_all["TIMESTAMP_START"] = pd.to_datetime(df_fluxnet_all["TIMESTAMP_START"], format="%d/%m/%Y")

    # Load coordinates
    df_coords = pd.read_csv(coords_path)

    # Generate day-by-day cross-referenced files for each date
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")        # For file name
        date_folder = current_date.strftime("%Y-%m-%d")   # For folder name

        # Combined file output path
        output_csv = os.path.join(matched_output_folder, f"matched_tb_fluxnet_{date_str}.csv")
        
        if os.path.exists(output_csv):
            print(f"📂 File already exists for {date_str}, move on to the next one.")
            current_date += timedelta(days=1)
            continue

        print(f"\n===== Generation of matches for {date_str} =====")
        generate_daily_matches(
            start_date=current_date,
            end_date=current_date,  # A single date for this iteration
            fluxnet_path=fluxnet_path,
            coords_path=coords_path,
            tb_folder=os.path.join(tb_folder, date_folder),  # Existing file for the date
            output_folder=matched_output_folder
        )

        # If the file has been created correctly, plot and save the regression
        if os.path.exists(output_csv):
            reg_plot_path = f"outputs/{date_folder}/regression_tb_vs_temp_{date_folder}.png"
            if new_graph or not os.path.exists(reg_plot_path):
                print(f"\n===== AMSR-E / FLUXNET regression record ({date_str}) =====")
                plot_brightness_vs_temperature_and_regression(output_csv, date_folder)
            else : 
                print("\n✅ Regression for this day already generated")


        # Go to next date
        current_date += timedelta(days=1)

    # Day-by-day linear regression after processing all dates
    print("\n===== Daily regression TB vs Temperature (multi-day) =====")
    output_regression_csv = "data/processed/amsre/daily_regressions.csv"
    fit_daily_regressions(matched_output_folder, output_regression_csv)

    print("\n===== Regression for each station & overall station regression =====")
    regression_dir = "outputs/fluxnet/stationwise_regressions"
    # Checks whether a single plot already exists, otherwise call the
    example_station = "FLX_FR-LBr_FLUXNET2015_FULLSET_1996-2008_1-4"  
    example_path = os.path.join(regression_dir, f"regression_2005_{example_station}.png")
    if new_graph or not os.path.exists(example_path):
        plot_stationwise_and_global_regressions_2005("data/raw/fluxnet/FluxNET_AMSRE.csv")
    else:
        print("⏭️ Graphics of stations already present, skip.")


    print("\n===== Seasonal temperature trend (DOY) by station =====")
    seasonal_dir = "outputs/fluxnet/seasonal_evolution"
    example_seasonal_path = os.path.join(seasonal_dir, f"saison_{example_station}.png")

    if new_graph or not os.path.exists(example_seasonal_path):
        plot_seasonal_temp_evolution("data/raw/fluxnet/FluxNET_AMSRE.csv")
    else:
        print("⏭️ Seasonal graphics already generated, skip.")
    
    print("\n===== Évolution saisonnière température + TB AMSR-E (comparaison) =====")
    seasonal_tb_dir = "outputs/fluxnet/seasonal_temp_vs_tb"
    example_tb_path = os.path.join(seasonal_tb_dir, f"temp_vs_tb_seasonal_{example_station}.png")

    if new_graph or not os.path.exists(example_tb_path):
        plot_seasonal_temp_with_tb_evolution()
    else:
        print("⏭️ TB vs Temp graphics already generated, skip.")


if __name__ == "__main__":
    main()