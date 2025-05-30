### Library and function imports ###

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import xarray as xr
from sklearn.model_selection import train_test_split

from src.modis.download import authenticate, search_modis_lst, download_results
from src.modis.process import process_nc_to_csv_light, process_all_modis_csv
from src.modis.plot import plot_temp_mean_Celsius, plot_temp_mean_Kelvin

from src.amsre.download import download_amsre_ae_l2a
from src.amsre.plot import plot_bt_map, plot_temp_estimated_map
from src.amsre.process import combine_amsre_files, concat_amsre_files
from src.amsre.fix_headers import fix_amsre_headers

from src.amsre.matches import generate_daily_matches
from src.amsre.plot_regressions import plot_stationwise_and_global_regressions_2005, plot_global_tb_vs_temp, plot_brightness_vs_temperature_and_regression, fit_daily_regressions, plot_station_regressions, plot_regression_metrics_evolution
from src.amsre.plot_temp_evolution import plot_seasonal_temp_evolution, plot_seasonal_temp_with_tb_evolution, plot_all_stations_temp_evolution, plot_temp_mean_amsre_Celsius, plot_temp_mean_amsre_Kelvin

from src.land_cover.process import convert_land_cover_nc_to_csv
from src.land_cover.plot import plot_land_cover_map

from src.visualization.maps import plot_difference_map_explicit

from src.machine_learning.model.regression import train_regression
from src.machine_learning.model.knn import train_knn
from src.machine_learning.model.svr import train_svr
from src.machine_learning.model.random_forest import train_random_forest
from src.machine_learning.model.gradient_boosting import train_gradient_boosting
from src.machine_learning.utils import load_and_merge_data, clean_data, evaluate_model, plot_results
from src.machine_learning.create_dataset import merge_daily_datasets


def main():

    ### SETTINGS ### 


    new_graph = True       # If the maps are to be generated again if they already exist
    

    ### MODIS PART ###
    

    print("\n===== MODIS stage: Land Surface Temperature =====")

    # Processing MODIS files
    nc_path = "data/raw/modis/MOD11A1.061_1km_aid0001.nc"  
    print(f"👓 Read file MOD11A1.061_1km_aid0001.nc\n")
    ds = xr.open_dataset(nc_path)
    start_date = datetime(2005, 1, 1)

    for day in range(365): 
        current_date = start_date + timedelta(days=day)
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"\n📅 Processing for date : {date_str}")
        process_nc_to_csv_light(ds, f"data/processed/modis/modis_lst_{date_str}.csv", day_index=day, variable_name="LST_Day_1km")
    ds.close()
    process_all_modis_csv(input_folder="data/processed/modis", output_folder="outputs/modis/dates")
    

    ### AMSRE PART ###
    

    start_date = datetime(2005, 1, 1)
    end_date = datetime(2005, 12, 31)
    dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end_date - start_date).days + 1)]
    #authenticate()
    
    print("\n===== AMSR-E stage: TB 37GHz processing and plotting map =====")
    
    for date in dates :
        date_str = date.replace("-", "")
        # Filter only .hdf files
        #files = [f for f in download_amsre_ae_l2a(date=date) if f.endswith('.hdf')]
        files = [os.path.join('data/raw/amsre',f) for f in os.listdir("data/raw/amsre") if f.endswith(".hdf") and date_str in f]

        # Combine the files into two separate CSVs and retrieve the output paths
        output_ascending_37, output_descending_37 = combine_amsre_files(files, date=date, frequency=37)
        output_ascending_19, output_descending_19 = combine_amsre_files(files, date=date, frequency=19)
    '''   
        # Maps Generation - 37GHz
        if output_ascending_37 and output_descending_37:
        
            print(f"\n===== AMSR-E Map Generation : TB_37GHz. Date : {date} =====")

            # Loading renamed files
            df_ascending_37 = pd.read_csv(output_ascending_37)
            df_descending_37 = pd.read_csv(output_descending_37)

            asc_plot_path_tb_37 = f"outputs/amsre/dates/{date}/tb_37ghz_map_{date}_ascending.png"
            des_plot_path_tb_37 = f"outputs/amsre/dates/{date}/tb_37ghz_map_{date}_descending.png"
            comb_plot_path_tb_37 = f"outputs/amsre/dates/{date}/tb_37ghz_map_{date}.png"

            if new_graph or not os.path.exists(asc_plot_path_tb_37):
                print("\n📈 visualisation of Ascending")
                plot_bt_map(df_ascending_37, date, pass_type="ascending",freq_label="37ghz")
            else : 
                print("\n✅ [37GHz] - Ascending TB map already generated")

            if new_graph or not os.path.exists(des_plot_path_tb_37):
                print("\n📉 Visualisation of Descending")
                plot_bt_map(df_descending_37, date, pass_type="descending",freq_label="37ghz")
            else : 
                print("\n✅ [37GHz] - Descending map already generated")

            if new_graph or not os.path.exists(comb_plot_path_tb_37):
                print("\n📊 Visualisation of Combined datas")
                plot_bt_map(pd.concat([df_ascending_37, df_descending_37]), date, pass_type="combined",freq_label="37ghz")
            else : 
                print("\n✅ [37GHz] - Combined map already generated\n")
            
        print(f"\n📊✅ 37GHz Maps completed for date : {date}\n")


        # Maps Generation - 19GHz
        if output_ascending_19 and output_descending_19:
        
            print(f"\n===== AMSR-E Map Generation : TB_19GHz. Date : {date} =====")

            # Loading renamed files
            df_ascending_19 = pd.read_csv(output_ascending_19)
            df_descending_19 = pd.read_csv(output_descending_19)

            asc_plot_path_tb_19 = f"outputs/amsre/dates/{date}/tb_19ghz_map_{date}_ascending.png"
            des_plot_path_tb_19 = f"outputs/amsre/dates/{date}/tb_19ghz_map_{date}_descending.png"
            comb_plot_path_tb_19 = f"outputs/amsre/dates/{date}/tb_19ghz_map_{date}.png"

            if new_graph or not os.path.exists(asc_plot_path_tb_19):
                print("\n📈 visualisation of Ascending")
                plot_bt_map(df_ascending_19, date, pass_type="ascending",freq_label="19ghz")
            else : 
                print("\n✅ [19GHz] - Ascending TB map already generated")

            if new_graph or not os.path.exists(des_plot_path_tb_19):
                print("\n📉 Visualisation of Descending")
                plot_bt_map(df_descending_19, date, pass_type="descending",freq_label="19ghz")
            else : 
                print("\n✅ [19GHz] - Descending map already generated")

            if new_graph or not os.path.exists(comb_plot_path_tb_19):
                print("\n📊 Visualisation of Combined datas")
                plot_bt_map(pd.concat([df_ascending_19, df_descending_19]), date, pass_type="combined",freq_label="19ghz")
            else : 
                print("\n✅ [19GHz] - Combined map already generated\n")
            
        print(f"\n📊✅ 19 GHz Maps completed for date : {date}\n")
    

    print("\n===== END of AMSR-E TB and Temperature by Regression =====")
    '''
    
    ### FLUXNET & PLOTS PART ###
    '''
    print("\n=====📥 Analysis with FLUXNET 📥=====")
    fluxnet_path = "data/raw/fluxnet/FluxNET_AMSRE.csv"
    coords_path = "data/processed/fluxnet/fluxnet_station_coordinates.csv"
    tb_folder = "data/processed/amsre"
    matched_output_folder = "data/processed/amsre/matched"

    # Load FLUXNET & coordinates
    df_fluxnet_all = pd.read_csv(fluxnet_path, sep=';')
    df_fluxnet_all["TIMESTAMP_START"] = pd.to_datetime(df_fluxnet_all["TIMESTAMP_START"], format="%d/%m/%Y")
    
    df_coords = pd.read_csv(coords_path)

    # Generate day-by-day cross-referenced files for each date
    current_date = start_date
    
    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")        # For file name
        date_folder = current_date.strftime("%Y-%m-%d")   # For folder name
        
        matched_output_folder_37 = matched_output_folder + "/37GHz"
        matched_output_folder_19 = matched_output_folder + "/19GHz"

        # Combined file output path for both frequencies
        output_csv_37 = os.path.join(matched_output_folder_37, f"matched_tb_fluxnet_{date_str}.csv")
        output_csv_19 = os.path.join(matched_output_folder_19, f"matched_tb_fluxnet_{date_str}.csv")
        
        if os.path.exists(output_csv_37) and os.path.exists(output_csv_19):
            print(f"📂 File already exists for {date_str} for both frequencies, move on to the next one.")
            current_date += timedelta(days=1)
            continue

        print(f"\n===== Generation of matches for {date_str} : 37GHz =====")
        generate_daily_matches(
            start_date=current_date,
            end_date=current_date,  # A single date for this iteration
            freq_label = "37GHz",
            fluxnet_path=fluxnet_path,
            coords_path=coords_path,
            tb_folder=os.path.join(tb_folder, date_folder),  # Existing file for the date
            output_folder=matched_output_folder_37
        )

        print(f"\n===== Generation of matches for {date_str} : 19GHz =====")
        generate_daily_matches(
            start_date=current_date,
            end_date=current_date,  # A single date for this iteration
            freq_label = "19GHz",
            fluxnet_path=fluxnet_path,
            coords_path=coords_path,
            tb_folder=os.path.join(tb_folder, date_folder),  # Existing file for the date
            output_folder=matched_output_folder_19
        )
        
                # Vérifie que le fichier existe et contient des données utiles
        if os.path.exists(output_csv_37):
            df_check = pd.read_csv(output_csv_37)
            if df_check.empty:
                print(f"⚠️ Fichier généré vide, graphique non généré pour {date_str} : {output_csv_37}")
            else:
                plot_brightness_vs_temperature_and_regression(output_csv_37, date_folder, "37GHz")
        else:
            print(f"⚠️ Fichier manquant, graphique non généré pour {date_str} : {output_csv_37}")


        if os.path.exists(output_csv_19):
            reg_plot_path_19 = f"outputs/{date_folder}/regression_tb_vs_temp_{date_folder}_19GHz.png"
            if new_graph or not os.path.exists(reg_plot_path_19):
                print(f"\n===== AMSR-E / FLUXNET regression record ({date_str}) for 19GHz =====")
                plot_brightness_vs_temperature_and_regression(output_csv_19, date_folder, "19Ghz")
            else : 
                print("\n✅ Regression for this day already generated")

        # Go to next date
        current_date += timedelta(days=1)
    
    
    # Day-by-day linear regression after processing all dates
    print("\n===== Daily regression TB vs Temperature (multi-day) for the 37GHz frequency =====")
    output_regression_csv_37 = "data/analysis/amsre/daily_regressions_37GHz.csv"
    fit_daily_regressions(matched_output_folder_37, output_regression_csv_37, "37GHz")
    plot_regression_metrics_evolution(output_regression_csv_37,"37GHz")
    aglob37, bglob37 = plot_global_tb_vs_temp("data/processed/amsre/matched/37GHz", "37GHz")


    print("\n===== Daily regression TB vs Temperature (multi-day) for the 19GHz frequency =====")
    output_regression_csv_19 = "data/analysis/amsre/daily_regressions_19GHz.csv"
    fit_daily_regressions(matched_output_folder_19, output_regression_csv_19, "19GHz")
    plot_regression_metrics_evolution(output_regression_csv_19,"19GHz")
    aglob19, bglob19 = plot_global_tb_vs_temp("data/processed/amsre/matched/19GHz", "19GHz")

    
    print("\n===== Régressions TB vs Température for each station =====")
    all_matched_df_37 = pd.concat([pd.read_csv(os.path.join(matched_output_folder_37, f)) for f in os.listdir(matched_output_folder_37) if f.endswith(".csv")],ignore_index=True)
    all_matched_df_19 = pd.concat([pd.read_csv(os.path.join(matched_output_folder_19, f)) for f in os.listdir(matched_output_folder_19) if f.endswith(".csv")],ignore_index=True)
    plot_station_regressions(all_matched_df_37,all_matched_df_19)


    print("\n===== Regression for each station & overall station regression =====")
    regression_dir = "outputs/fluxnet/stationwise_regressions"
    # Checks whether a single plot already exists, otherwise call the
    example_station = "FLX_FR-LBr_FLUXNET2015_FULLSET_1996-2008_1-4"  
    example_path = os.path.join(regression_dir, f"regression_2005_{example_station}.png")
    if new_graph or not os.path.exists(example_path):
        plot_stationwise_and_global_regressions_2005("data/raw/fluxnet/FluxNET_AMSRE.csv","37GHz")
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
    example_station = "FLX_FR-LBr_FLUXNET2015_FULLSET_1996-2008_1-4"
    example_tb_path = os.path.join(seasonal_tb_dir, f"temp_vs_tb_seasonal_{example_station}.png")
    if new_graph or not os.path.exists(example_tb_path):
        plot_seasonal_temp_with_tb_evolution()
    else:
        print("⏭️ TB vs Temp graphics already generated, skip.")

    
    print("\n===== Évolution temporelle de toutes les températures =====")
    seasonal_tb_dir = "outputs/fluxnet/seasonal_temp_vs_tb"
    example_temp_path = os.path.join(seasonal_tb_dir, f"temp_by_station.png")
    if new_graph or not os.path.exists(example_temp_path):
        plot_all_stations_temp_evolution("data/raw/fluxnet/FluxNET_AMSRE.csv")
    else:
        print("⏭️ All temp graphic already generated, skip.")'''
    
    ### Temperature generated from linear regression ###
    '''
    files = [f for f in os.listdir("data/raw/amsre") if f.endswith(".hdf")]         # A supprimer

    # Plotting supposed temperature maps - AMSRE
    for date in dates :          
        output_ascending_37, output_descending_37 = combine_amsre_files(files, date=date, frequency=37)
        output_ascending_19, output_descending_19 = combine_amsre_files(files, date=date, frequency=19)

        df_ascending_37 = pd.read_csv(output_ascending_37)
        df_descending_37 = pd.read_csv(output_descending_37)

        df_ascending_19 = pd.read_csv(output_ascending_19)
        df_descending_19 = pd.read_csv(output_descending_19)
        
        ### Temperatures generated from linear regression - 37 GHz
                
        asc_plot_path_regtemp_37 = f"outputs/amsre/dates/{date}/temp_by_reg_37ghz_map_{date}_ascending.png"
        des_plot_path_regtemp_37 = f"outputs/amsre/dates/{date}/temp_by_reg_37ghz_map_{date}_descending.png"
        comb_plot_path_regtemp_37 = f"outputs/amsre/dates/{date}/temp_by_reg_37ghz_map_{date}_combined.png"

        print(f"\n📊 37GHz Supposed Temperature Maps for date : {date}")
        
        if new_graph or not os.path.exists(asc_plot_path_regtemp_37):
            print("\n📈 Visualisation of Ascending Supposed Temperature")
            plot_temp_estimated_map(df_ascending_37, date, pass_type="ascending", freq_label="37ghz", a=aglob37, b=bglob37)
        else : 
            print("\n✅ [37GHz] - Ascending supposed temperatures map already generated")

        if new_graph or not os.path.exists(des_plot_path_regtemp_37):    
            print("\n📉 Visualisation of Descending Supposed Temperature")
            plot_temp_estimated_map(df_descending_37, date, pass_type="descending", freq_label="37ghz", a=aglob37, b=bglob37)
        else : 
            print("\n✅ [37GHz] - Descending supposed temperatures map already generated")

        if new_graph or not os.path.exists(comb_plot_path_regtemp_37):   
            print("\n📊 Visualisation of Combined Supposed Temperature datas") 
            plot_temp_estimated_map(pd.concat([df_ascending_37, df_descending_37]), date, pass_type="combined", freq_label="37ghz", a=aglob37, b=bglob37)
        else : 
            print("\n✅ [37GHz] - Combined supposed temperatures map already generated")       

        ### Temperatures generated from linear regression - 19 GHz

        asc_plot_path_regtemp_19 = f"outputs/amsre/dates/{date}/temp_by_reg_19ghz_map_{date}_ascending.png"
        des_plot_path_regtemp_19 = f"outputs/amsre/dates/{date}/temp_by_reg_19ghz_map_{date}_descending.png"
        comb_plot_path_regtemp_19 = f"outputs/amsre/dates/{date}/temp_by_reg_19ghz_map_{date}_combined.png"

        print(f"\n📊 19GHz Supposed Temperature Maps for date : {date}")

        if new_graph or not os.path.exists(asc_plot_path_regtemp_19):
            print("\n📈 Sampled Visualisation of Ascending Supposed Temperature")
            plot_temp_estimated_map(df_ascending_19, date, pass_type="ascending", freq_label="19ghz", a=aglob19, b=bglob19)
        else : 
            print("\n✅ [19GHz] - Ascending supposed temperatures map already generated")

        if new_graph or not os.path.exists(des_plot_path_regtemp_19):    
            print("\n📉 [19GHz] - Sampled visualisation of Descending Supposed Temperature")
            plot_temp_estimated_map(df_descending_19, date, pass_type="descending", freq_label="19ghz", a=aglob19, b=bglob19)
        else : 
            print("\n✅ [19GHz] - Descending supposed temperatures map already generated")

        if new_graph or not os.path.exists(comb_plot_path_regtemp_19):   
            print("\n📊 Sampled visualisation of Combined Supposed Temperature datas") 
            plot_temp_estimated_map(pd.concat([df_ascending_19, df_descending_19]), date, pass_type="combined", freq_label="19ghz", a=aglob19, b=bglob19)
        else : 
            print("\n✅ Combined supposed temperatures map already generated")'''
    
    ### Average temperature maps ###
    '''
    output_file_average_temp_kelvin_amsre_19 = "outputs/amsre/mean_temp_2005_19GHz_Kelvin.png"
    output_file_average_temp_celsius_amsre_19 = "outputs/amsre/mean_temp_2005_19GHz_Celsius.png"

    output_file_average_temp_kelvin_amsre_37 = "outputs/amsre/mean_temp_2005_37GHz_Kelvin.png"
    output_file_average_temp_celsius_amsre_37 = "outputs/amsre/mean_temp_2005_37GHz_Celsius.png"

    input_dir = "data/processed/modis" 
    csv_dir_Kelvin = "data/processed/modis/mean_temp_2005_Kelvin.csv"
    csv_dir_Celsius = "data/processed/modis/mean_temp_2005_Celsius.csv"
    output_file_Kelvin = "outputs/modis/mean_temp_2005_Kelvin.png"
    output_file_Celsius = "outputs/modis/mean_temp_2005_Celsius.png"

    # AMSRE #
    
    if new_graph or not os.path.exists(output_file_average_temp_kelvin_amsre_19): 
        print("\n📊 19GHz - Generation of the AMSRE average annual calculated temperature map in Kelvin ")
        plot_temp_mean_amsre_Kelvin(freq_label="19")
        print(f"\n✅ AMSRE average annual calculated temperature map in Kelvin generated at {output_file_average_temp_celsius_amsre_19}")
    else : 
        print("\n✅ AMSRE average annual calculated temperature map in Kelvin already generated")
    
    if new_graph or not os.path.exists(output_file_average_temp_kelvin_amsre_19): 
        print("\n📊 19GHz - Generation of the AMSRE average annual calculated temperature map in Celsius")
        plot_temp_mean_amsre_Celsius(freq="19")
        print(f"\n✅ AMSRE average annual calculated temperature map generated at {output_file_average_temp_kelvin_amsre_19}")
    else : 
        print("\n✅ AMSRE average annual calculated temperature map generated")
    
    if new_graph or not os.path.exists(output_file_average_temp_kelvin_amsre_37): 
        print("\n📊 37GHz - Generation of the AMSRE average annual calculated temperature map in Kelvin ")
        plot_temp_mean_amsre_Kelvin(freq_label="37")
        print(f"\n✅ AMSRE average annual calculated temperature map in Kelvin generated at {output_file_average_temp_celsius_amsre_37}")
    else : 
        print("\n✅ AMSRE average annual calculated temperature map in Kelvin already generated")
    
    if new_graph or not os.path.exists(output_file_average_temp_kelvin_amsre_37): 
        print("\n📊 37GHz - Generation of the AMSRE average annual calculated temperature map in Celsius")
        plot_temp_mean_amsre_Celsius(freq="37")
        print(f"\n✅ AMSRE average annual calculated temperature map generated at {output_file_average_temp_kelvin_amsre_37}")
    else : 
        print("\n✅ AMSRE average annual calculated temperature map generated")
    
    # MODIS #
    
    if new_graph or not os.path.exists(output_file_Kelvin): 
        print("\n📊 Generation of the average annual temperature map in Kelvin ")
        plot_temp_mean_Kelvin(input_dir,csv_dir_Kelvin,output_file_Kelvin)
        print(f"\n✅ Map of average annual temperatures in Kelvin generated at {output_file_Kelvin}")
    else : 
        print("\n✅ Map of average annual temperatures in Kelvin already generated")
    
    if new_graph or not os.path.exists(output_file_Celsius): 
        print("\n📊 Generation of the average annual temperature map in Celsius")
        plot_temp_mean_Celsius(input_dir,csv_dir_Celsius,output_file_Celsius)
        print(f"\n✅ Map of average annual temperatures in Celsius generated at {output_file_Celsius}")
    else : 
        print("\n✅ Map of average annual temperatures in Celsius already generated")'''

    # MODIS vs 37GHz - Kelvin
    '''
    if new_graph or not os.path.exists("outputs/comparisons/diff_MODIS_37GHz_Kelvin.png"): 
        print("\n📊 Generation of the difference average annual temperature map between MODIS et AMSRE 37GHz in Kelvin")
        
        plot_difference_map_explicit(
            modis_csv="data/processed/modis/mean_temp_2005_Kelvin.csv",
            amsre_csv="data/processed/amsre/mean_temp_2005_37_Kelvin.csv",
            modis_col="LST_Kelvin_mean",
            amsre_col="temp_K_mean",
            output_path="outputs/comparisons/diff_MODIS_37GHz_Kelvin.png",
            title="Différence MODIS - AMSRE (37GHz) [Kelvin]",
            color_label="Différence de température (K)")
        
        print(f"🖼️ Map of the difference average annual temperatures between MODIS et AMSRE 37GHz in Kelvin generated at : outputs/comparisons/diff_MODIS_37GHz_Kelvin.png")
    else : 
        print("\n✅ Map of the difference average annual temperatures between MODIS et AMSRE 37GHz in Kelvin already generated")


    # MODIS vs 37GHz - Celsius
    if new_graph or not os.path.exists("outputs/comparisons/diff_MODIS_37GHz_Celsius.png"): 
        print("\n📊 Generation of the difference average annual temperature map between MODIS et AMSRE 37GHz in Celsius")
        
        plot_difference_map_explicit(
            modis_csv="data/processed/modis/mean_temp_2005_Celsius.csv",
            amsre_csv="data/processed/amsre/mean_temp_2005_37_Celsius.csv",
            modis_col="LST_Celsius_mean",
            amsre_col="temp_C_mean",
            output_path="outputs/comparisons/diff_MODIS_37GHz_Celsius.png",
            title="Différence MODIS - AMSRE (37GHz) [Celsius]",
            color_label="Différence de température (°C)")
        
        print(f"🖼️ Map of the difference average annual temperatures between MODIS et AMSRE 37GHz in Celsius generated at : outputs/comparisons/diff_MODIS_37GHz_Celsius.png")
    else : 
        print("\n✅ Map of the difference average annual temperatures between MODIS et AMSRE 37GHz in Celsius already generated")


    # MODIS vs 19GHz - Kelvin
    if new_graph or not os.path.exists("outputs/comparisons/diff_MODIS_19GHz_Kelvin.png"): 
        print("\n📊 Generation of the difference average annual temperature map between MODIS et AMSRE 19GHz in Kelvin")
        
        plot_difference_map_explicit(
            modis_csv="data/processed/modis/mean_temp_2005_Kelvin.csv",
            amsre_csv="data/processed/amsre/mean_temp_2005_19_Kelvin.csv",
            modis_col="LST_Kelvin_mean",
            amsre_col="temp_K_mean",
            output_path="outputs/comparisons/diff_MODIS_19GHz_Kelvin.png",
            title="Différence MODIS - AMSRE (19GHz) [Kelvin]",
            color_label="Différence de température (K)")
        
        print(f"🖼️ Map of the difference average annual temperatures between MODIS et AMSRE 19GHz in Kelvin generated at : outputs/comparisons/diff_MODIS_19GHz_Kelvin.png")
    else : 
        print("\n✅ Map of the difference average annual temperatures between MODIS et AMSRE 19GHz in Kelvin already generated")


    # MODIS vs 19GHz - Celsius
    if new_graph or not os.path.exists("outputs/comparisons/diff_MODIS_19GHz_Celsius.png"): 
        print("\n📊 Generation of the difference average annual temperature map between MODIS et AMSRE 19GHz in Celsius")
        
        plot_difference_map_explicit(
            modis_csv="data/processed/modis/mean_temp_2005_Celsius.csv",
            amsre_csv="data/processed/amsre/mean_temp_2005_19_Celsius.csv",
            modis_col="LST_Celsius_mean",
            amsre_col="temp_C_mean",
            output_path="outputs/comparisons/diff_MODIS_19GHz_Celsius.png",
            title="Différence MODIS - AMSRE (19GHz) [Celsius]",
            color_label="Différence de température (°C)")
        
        print(f"🖼️ Map of the difference average annual temperatures between MODIS et AMSRE 19GHz in Celsius generated at : outputs/comparisons/diff_MODIS_19GHz_Celsius.png")
    else : 
        print("\n✅ Map of the difference average annual temperatures between MODIS et AMSRE 19GHz in Celsius already generated")'''
    

    ### LAND COVER PART ###
    '''
    nc_path = "data/raw/land_cover/968_Land_Cover_Class_0.25degree.nc4"
    land_cover_map_output = "outputs/land_cover/land_cover_map.png"
    land_cover_csv_output = "data/processed/land_cover/land_cover_classes.csv"

    print("\n===== Convert NetCDF Land Cover to CSV =====")
    convert_land_cover_nc_to_csv(nc_path=nc_path, output_csv_path=land_cover_csv_output)

    print("\n===== Plot Land Cover Map =====")
    if new_graph or not os.path.exists(land_cover_map_output):
            print("\n📊 Visualisation of Land Cover Map")
            plot_land_cover_map(nc_path=nc_path,output_img_path=land_cover_map_output)
    else : 
            print("\n✅ Land Cover map already generated")'''
    

    ### MACHINE LEARNING ###
    '''
    MERGED_FOLDER = "data/processed/machine_learning"
    CLEANED_FILE = "data/processed/machine_learning/cleaned_data.csv"
    OUTPUT_DIR = "outputs/machine_learning"

    
    print("\n===== Merge all AMSRE CSVs into one =====")
    concat_amsre_files(input_dir="data/processed/amsre/",output_file="data/processed/amsre/merged_amsre_data.csv")

    print("\n===== Merge AMSRE, MODIS and Land Cover data for ML =====")
    merge_daily_datasets()

    print("\n✅ Data merge for ML completed.")

    print("🚀 Data processing and cleansing...")
    load_and_merge_data(MERGED_FOLDER, output_file=CLEANED_FILE)

    print("📥 Loading the cleaned file...")
    df = pd.read_csv(CLEANED_FILE)

    # Features (X) and target (y)
    X = df[["brightness_temp_19v", "brightness_temp_37v"]]
    y = df["LST_Celsius"]  # ou "LST_Kelvin" si tu préfères

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model dictionary
    models = {
        "LinearRegression": train_regression,
        "KNN": train_knn,
        "SVR": train_svr,
        "RandomForest": train_random_forest,
        "GradientBoosting": train_gradient_boosting
    }

    results = []

    # Training and assessment
    for name, train_func in models.items():
        print(f"\n⚙️  Model drive : {name}")
        model = train_func(X_train, y_train)
        y_pred, rmse, r2 = evaluate_model(model, X_test, y_test)
        plot_path = os.path.join(OUTPUT_DIR, f"{name}_prediction.png")
        plot_results(y_test, y_pred, plot_path)
        results.append((name, rmse, r2))
        print(f"📈 {name} — RMSE : {rmse:.2f}, R² : {r2:.2f}")

    # Final summary
    print("\n📊 Performance summary :")
    print(f"{'Modèle':<20} {'RMSE':<10} {'R²':<10}")
    for name, rmse, r2 in results:
        print(f"{name:<20} {rmse:<10.2f} {r2:<10.2f}")
    '''


if __name__ == "__main__":
    main()
    
    


