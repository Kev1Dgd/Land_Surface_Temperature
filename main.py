import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from src.modis.download import authenticate, search_modis_lst, download_results
from src.modis.process import process_hdf_to_csv
from src.modis.analyze import analyze_all_files 
from src.modis.analysis_utils import analyze_monthly_data, analyze_spatial_data
from src.modis.utils import check_and_create_file

from src.amsre.download import download_amsre_ae_l2a
from src.amsre.plot import plot_bt_map
from src.amsre.process import combine_amsre_files_37ghz, combine_amsre_files_19ghz

from src.amsre.matches import generate_daily_matches
from src.amsre.plot_regressions import plot_stationwise_and_global_regressions_2005, plot_global_tb_vs_temp, plot_brightness_vs_temperature_and_regression, fit_daily_regressions, plot_station_regressions, plot_regression_metrics_evolution
from src.amsre.plot_temp_evolution import plot_seasonal_temp_evolution, plot_seasonal_temp_with_tb_evolution, plot_all_stations_temp_evolution



def main():
    n_point = 5000
    Sampling = False
    new_graph = False 
    
    '''# Step 1: Authentication with NASA
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

        process_hdf_to_csv(file_path, output_file)

    # 🆕 Step 5: Analyse
    print("📊 Starting files analysis...")
    
    check_and_create_file("data/analysis/modis/lst_summary_2005.json", analyze_all_files, input_dir="data/processed/modis")
    check_and_create_file("data/analysis/modis/lst_monthly_summary_2005.json", analyze_monthly_data, input_dir="data/processed/modis")
    check_and_create_file("data/analysis/modis/lst_spatial_summary_2005.json",analyze_spatial_data,input_dir="data/processed/modis")
    
    print("📈 Analysis completed.")'''

    
    
    print("\n===== AMSR-E stage: TB 37GHz processing =====")
    start_date = datetime(2005, 1, 1)
    end_date = datetime(2005, 5, 31)
    dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end_date - start_date).days + 1)]
    authenticate()

    for date in dates :
        # Filter only .hdf files
        files = [f for f in download_amsre_ae_l2a(date=date) if f.endswith('.hdf')]

        # Combine the files into two separate CSVs and retrieve the output paths
        output_ascending, output_descending = combine_amsre_files_37ghz(files, date=date)

        # Load data and generate maps
        '''if output_ascending and output_descending:
            # Loading renamed files
            df_ascending = pd.read_csv(output_ascending)
            df_descending = pd.read_csv(output_descending)

            # 5000-point sampling
            if Sampling :
                df_ascending_sampled = df_ascending.sample(n=n_point) if len(df_ascending) > 5000 else df_ascending
                df_descending_sampled = df_descending.sample(n=n_point) if len(df_descending) > 5000 else df_descending

                asc_plot_path = f"outputs/amsre/dates/{date}/tb_37ghz_map_{date}_ascending.png"
                des_plot_path = f"outputs/amsre/dates/{date}/tb_37ghz_map_{date}_descending.png"
                comb_plot_path = f"outputs/amsre/dates/{date}/tb_37ghz_map_{date}.png"

                if new_graph or not os.path.exists(asc_plot_path):
                    print("\n📈 Sample visualisation of Ascending")
                    plot_bt_map(df_ascending_sampled, date, pass_type="ascending",freq_label="37ghz")
                else : 
                    print("\n✅ Ascending map already generated")

                if new_graph or not os.path.exists(des_plot_path):
                    print("\n📉 Sample visualisation of Descending")
                    plot_bt_map(df_descending_sampled, date, pass_type="descending",freq_label="37ghz")
                else : 
                    print("\n✅ Descending map already generated")

                if new_graph or not os.path.exists(comb_plot_path):
                    print("\n📊 Sample visualisation of Combined datas")
                    plot_bt_map(pd.concat([df_ascending_sampled, df_descending_sampled]), date, pass_type="combined",freq_label="37ghz")
                else : 
                    print("\n✅  Combined map already generated\n")

            else :  

                asc_plot_path = f"outputs/amsre/dates/{date}/tb_37ghz_map_{date}_ascending.png"
                des_plot_path = f"outputs/amsre/dates/{date}/tb_37ghz_map_{date}_descending.png"
                comb_plot_path = f"outputs/amsre/dates/{date}/tb_37ghz_map_{date}.png"

                if new_graph or not os.path.exists(asc_plot_path):
                    print("\n📈 VSample visualisation of Ascending")
                    plot_bt_map(df_ascending, date, pass_type="ascending",freq_label="37ghz")
                else : 
                    print("\n✅ Ascending map already generated")

                if new_graph or not os.path.exists(des_plot_path):
                    print("\n📉 Sample visualisation of Descending")
                    plot_bt_map(df_descending, date, pass_type="descending",freq_label="37ghz")
                else : 
                    print("\n✅ Descending map already generated")

                if new_graph or not os.path.exists(comb_plot_path):
                    print("\n📊 Sample visualisation of Combined datas")
                    plot_bt_map(pd.concat([df_ascending, df_descending]), date, pass_type="combined",freq_label="37ghz")
                else : 
                    print("\n✅ Combined map already generated\n")

        print(f"Treatment completed for date : {date}\n")'''
    
    print("\n===== END of AMSR-E TB 37GHz processing =====")

    '''print("\n===== AMSR-E stage: TB 19GHz processing =====")

    for date in dates:
        # Filter only .hdf files
        files = [f for f in download_amsre_ae_l2a(date=date) if f.endswith('.hdf')]

        # Combine the files into two separate CSVs and retrieve the output paths
        output_ascending, output_descending = combine_amsre_files_19ghz(files, date=date)

        # Load data and generate maps
        if output_ascending and output_descending:
            df_ascending = pd.read_csv(output_ascending)
            df_descending = pd.read_csv(output_descending)

            if Sampling:
                df_ascending_sampled = df_ascending.sample(n=n_point) if len(df_ascending) > 5000 else df_ascending
                df_descending_sampled = df_descending.sample(n=n_point) if len(df_descending) > 5000 else df_descending

                asc_plot_path = f"outputs/amsre/{date}/tb_19ghz_map_{date}_ascending.png"
                des_plot_path = f"outputs/amsre/{date}/tb_19ghz_map_{date}_descending.png"
                comb_plot_path = f"outputs/amsre/{date}/tb_19ghz_map_{date}.png"

                if new_graph or not os.path.exists(asc_plot_path):
                    print("\n📈 Sample visualisation of Ascending (19GHz)")
                    plot_bt_map(df_ascending_sampled, date, pass_type="ascending", freq_label="19ghz")
                else:
                    print("\n✅ Ascending 19GHz map already generated")

                if new_graph or not os.path.exists(des_plot_path):
                    print("\n📉 Sample visualisation of Descending (19GHz)")
                    plot_bt_map(df_descending_sampled, date, pass_type="descending", freq_label="19ghz")
                else:
                    print("\n✅ Descending 19GHz map already generated")

                if new_graph or not os.path.exists(comb_plot_path):
                    print("\n📊 Sample visualisation of Combined datas (19GHz)")
                    plot_bt_map(pd.concat([df_ascending_sampled, df_descending_sampled]), date, pass_type="combined", freq_label="19ghz")
                else:
                    print("\n✅ Combined 19GHz map already generated\n")

            else:
                asc_plot_path = f"outputs/amsre/{date}/tb_19ghz_map_{date}_ascending.png"
                des_plot_path = f"outputs/amsre/{date}/tb_19ghz_map_{date}_descending.png"
                comb_plot_path = f"outputs/amsre/{date}/tb_19ghz_map_{date}.png"

                if new_graph or not os.path.exists(asc_plot_path):
                    print("\n📈 Visualisation of Ascending (19GHz)")
                    plot_bt_map(df_ascending, date, pass_type="ascending", freq_label="19ghz")
                else:
                    print("\n✅ Ascending 19GHz map already generated")

                if new_graph or not os.path.exists(des_plot_path):
                    print("\n📉 Visualisation of Descending (19GHz)")
                    plot_bt_map(df_descending, date, pass_type="descending", freq_label="19ghz")
                else:
                    print("\n✅ Descending 19GHz map already generated")

                if new_graph or not os.path.exists(comb_plot_path):
                    print("\n📊 Visualisation of Combined datas (19GHz)")
                    plot_bt_map(pd.concat([df_ascending, df_descending]), date, pass_type="combined", freq_label="19ghz")
                else:
                    print("\n✅ Combined 19GHz map already generated\n")

        print(f"Treatment completed for 19GHz - date : {date}\n")'''


    
    print("\n===== FLUXNET =====")
    # Parameters
    fluxnet_path = "data/raw/fluxnet/FluxNET_AMSRE.csv"
    coords_path = "data/processed/fluxnet/fluxnet_station_coordinates.csv"
    tb_folder = "data/processed/amsre/dates"
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
    output_regression_csv = "data/analysis/amsre/daily_regressions.csv"
    fit_daily_regressions(matched_output_folder, output_regression_csv)
    plot_regression_metrics_evolution(output_regression_csv)

    plot_global_tb_vs_temp("data/processed/amsre/matched")


    print("\n===== Régressions TB vs Température for each station =====")
    all_matched_df = pd.concat([pd.read_csv(os.path.join(matched_output_folder, f)) for f in os.listdir(matched_output_folder) if f.endswith(".csv")],ignore_index=True)
    plot_station_regressions(all_matched_df)


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


    print("\n===== Évolution temporelle de toutes les températures =====")

    example_temp_path = os.path.join(seasonal_tb_dir, f"temp_by_station.png")
    if new_graph or not os.path.exists(example_tb_path):
        plot_all_stations_temp_evolution("data/raw/fluxnet/FluxNET_AMSRE.csv")
    else:
        print("⏭️ All temp graphic already generated, skip.")
    

if __name__ == "__main__":
    main()