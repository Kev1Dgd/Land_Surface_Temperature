### Library and function imports ###

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import xarray as xr
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split

from src.modis.download import authenticate, search_modis_lst, download_results
from src.modis.process import process_nc_to_csv_light, process_all_modis_csv
from src.modis.plot import plot_temp_mean_Celsius, plot_temp_mean_Kelvin

from src.amsre.download import download_amsre_ae_l2a
from src.amsre.plot import plot_bt_map, plot_temp_estimated_map, plot_temp_mean_amsre_Kelvin, plot_temp_mean_amsre_Celsius
from src.amsre.process import combine_amsre_files, merge_amsre_csvs_per_frequency
from src.amsre.fix_headers import fix_amsre_headers
from src.amsre.matches import generate_daily_matches
from src.amsre.plot_regressions import plot_stationwise_and_global_regressions_2005, plot_global_tb_vs_temp, plot_brightness_vs_temperature_and_regression, plot_station_regressions, plot_regression_metrics_evolution
from src.amsre.plot_temp_evolution import plot_seasonal_temp_with_tb_evolution, plot_all_stations_temp_evolution
from src.amsre.regression import fit_daily_regressions

from src.land_cover.process import convert_land_cover_nc_to_csv
from src.land_cover.plot import plot_land_cover_map

from src.comparisons.maps import plot_difference_map_explicit
from src.comparisons.fluxnet_vs_satellite import load_european_fluxnet_data, match_fluxnet_with_amsre, batch_plot_all_stations
from src.comparisons.utils import clean_temp_fluxnet

from src.machine_learning.model.regression import train_regression
from src.machine_learning.model.knn import train_knn
from src.machine_learning.model.random_forest import train_random_forest
from src.machine_learning.model.gradient_boosting import train_gradient_boosting
from src.machine_learning.utils import load_and_merge_data, evaluate_model, normalize_z_score, load_land_cover_lookup
from src.machine_learning.model.catboost import train_catboost
from src.machine_learning.model.lightgbm import train_lightgbm
from src.machine_learning.model.xgboost import train_xgboost
from src.machine_learning.create_dataset import merge_daily_datasets, concat_amsre_files
from src.machine_learning.plot import plot_results, plot_prediction_map, plot_mean_map, plot_error_map, plot_mean_error_map, plot_error_distributions, plot_error_histogram_vs_modis, plot_error_by_landcover, plot_daily_error_trend
from src.machine_learning.correlation import generate_heatmap_correlation


def main():

    ### SETTINGS ### 


    new_graph = True                    # If the maps are to be generated again if they already exist
    normalized_input = True            # If you want to normalize input data
    start_date = datetime(2005, 1, 1)
    end_date = datetime(2005, 12, 31)
    dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end_date - start_date).days + 1)]
    

    ### MODIS PART ###
    

    '''
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
    '''


    ### AMSRE PART ### A EXECUTER POUR LES GRAPHES
    

    '''
    # === FOLDER DEFINITIONS ===
    OUTPUT_DIR = "outputs/amsre"
    OUTPUT_DIR_V = OUTPUT_DIR + "/vertical_polarization"
    OUTPUT_DIR_H = OUTPUT_DIR +"/horizontal_polarization"

    authenticate()
    
    print("\n===== AMSR-E stage: TB 37GHz processing and plotting map =====")
    
    for date in dates :
        print(f"\n📅 Processing date : {date}")
        if date == '2005-11-17':
            print("\n❌ No data avalaible for this date : 2005-11-17")
            continue 
        
        date_str = date.replace("-", "")
        # Filter only .hdf files
        #files = [f for f in download_amsre_ae_l2a(date=date) if f.endswith('.hdf')]
        files = [os.path.join('data/raw/amsre',f) for f in os.listdir("data/raw/amsre") if f.endswith(".hdf") and date_str in f]

        # Combine the files into two separate CSVs and retrieve the output paths
        output_ascending_37v, output_descending_37v, output_ascending_37h, output_descending_37h = combine_amsre_files(files, date=date, frequency=37)
        output_ascending_19v, output_descending_19v, output_ascending_19h, output_descending_19h = combine_amsre_files(files, date=date, frequency=19)
        
        merge_amsre_csvs_per_frequency(date)
        
        # === FOLDER DEFINITIONS ===
        OUTPUT_DIR_V_DAY = OUTPUT_DIR_V + "/dates"
        OUTPUT_DIR_H_DAY = OUTPUT_DIR_H + "/dates"
        
        # == Maps Generation - Vertical polarization - 37GHz == 
        if output_ascending_37v and output_descending_37v:
        
            print(f"\n===== AMSR-E Map Generation : TB_37GHz. Date : {date} =====")

            df_ascending_37v = pd.read_csv(output_ascending_37v)
            df_descending_37v = pd.read_csv(output_descending_37v)

            asc_plot_path_tb_37v = OUTPUT_DIR_V_DAY + f"{date}/tb_37ghz_vertical_map_{date}_ascending.png"
            des_plot_path_tb_37v = OUTPUT_DIR_V_DAY + f"/{date}/tb_37ghz_vertical_map_{date}_descending.png"
            comb_plot_path_tb_37v = OUTPUT_DIR_V_DAY + f"/{date}/tb_37ghz_vertical_map_{date}.png"

            if new_graph or not os.path.exists(asc_plot_path_tb_37v):
                print("\n📈 visualisation of Ascending")
                plot_bt_map(df=df_ascending_37v, date=date, pass_type="ascending", polar="vertical", freq_label="37ghz", output_dir=OUTPUT_DIR_V_DAY)
            else : 
                print("\n✅ [37GHz] - Ascending TB map already generated - Vertical polarization")

            if new_graph or not os.path.exists(des_plot_path_tb_37v):
                print("\n📉 Visualisation of Descending")
                plot_bt_map(df=df_descending_37v, date=date, pass_type="descending", polar="vertical", freq_label="37ghz", output_dir=OUTPUT_DIR_V_DAY)
            else : 
                print("\n✅ [37GHz] - Descending map already generated - Vertical polarization")

            if new_graph or not os.path.exists(comb_plot_path_tb_37v):
                print("\n📊 Visualisation of Combined datas")
                plot_bt_map(df=pd.concat([df_ascending_37v, df_descending_37v]), date=date, pass_type="combined", polar="vertical", freq_label="37ghz", output_dir=OUTPUT_DIR_V_DAY)
            else : 
                print("\n✅ [37GHz] - Combined map already generated - Vertical polarization\n")
            
        print(f"\n📊✅ 37GHz Maps completed for date : {date} - Vertical polarization\n")

        # == Maps Generation - Horizontal polarization - 37GHz ==
        if output_ascending_37h and output_descending_37h:
        
            print(f"\n===== AMSR-E Map Generation : TB_37GHz. Date : {date} =====")

            # Loading renamed files
            df_ascending_37h = pd.read_csv(output_ascending_37h)
            df_descending_37h = pd.read_csv(output_descending_37h)

            asc_plot_path_tb_37h = OUTPUT_DIR_H_DAY + f"/{date}/tb_37ghz_map_{date}_ascending.png"
            des_plot_path_tb_37h = OUTPUT_DIR_H_DAY + f"/{date}/tb_37ghz_map_{date}_descending.png"
            comb_plot_path_tb_37h = OUTPUT_DIR_H_DAY + f"/{date}/tb_37ghz_map_{date}.png"

            if new_graph or not os.path.exists(asc_plot_path_tb_37h):
                print("\n📈 visualisation of Ascending")
                plot_bt_map(df=df_ascending_37h, date=date, pass_type="ascending", polar="horizontal", freq_label="37ghz", output_dir=OUTPUT_DIR_H_DAY)
            else : 
                print("\n✅ [37GHz] - Ascending TB map already generated")

            if new_graph or not os.path.exists(des_plot_path_tb_37h):
                print("\n📉 Visualisation of Descending")
                plot_bt_map(df_descending_37h, date, pass_type="descending", polar="horizontal", freq_label="37ghz", output_dir=OUTPUT_DIR_H_DAY)
            else : 
                print("\n✅ [37GHz] - Descending map already generated")

            if new_graph or not os.path.exists(comb_plot_path_tb_37h):
                print("\n📊 Visualisation of Combined datas")
                plot_bt_map(pd.concat([df_ascending_37h, df_descending_37h]), date, pass_type="combined", polar="horizontal", freq_label="37ghz", output_dir=OUTPUT_DIR_H_DAY)
            else : 
                print("\n✅ [37GHz] - Combined map already generated\n")
            
        print(f"\n📊✅ 37GHz Maps completed for date : {date} - Horizontal polarization\n")


        # == Maps Generation - Vertical polarization - 19GHz == 
        if output_ascending_19v and output_descending_19v:
        
            print(f"\n===== AMSR-E Map Generation : TB_19GHz. Date : {date} =====")

            df_ascending_19v = pd.read_csv(output_ascending_19v)
            df_descending_19v = pd.read_csv(output_descending_19v)

            asc_plot_path_tb_19v = OUTPUT_DIR_V + f"/dates/{date}/tb_19ghz_vertical_map_{date}_ascending.png"
            des_plot_path_tb_19v = OUTPUT_DIR_V + f"/dates/{date}/tb_19ghz_vertical_map_{date}_descending.png"
            comb_plot_path_tb_19v = OUTPUT_DIR_V + f"/dates/{date}/tb_19ghz_vertical_map_{date}.png"

            if new_graph or not os.path.exists(asc_plot_path_tb_19v):
                print("\n📈 visualisation of Ascending")
                plot_bt_map(df=df_ascending_19v, date=date, pass_type="ascending", polar="vertical", freq_label="19ghz", output_dir=OUTPUT_DIR_V_DAY)
            else : 
                print("\n✅ [19GHz] - Ascending TB map already generated - Vertical polarization")

            if new_graph or not os.path.exists(des_plot_path_tb_19v):
                print("\n📉 Visualisation of Descending")
                plot_bt_map(df=df_descending_19v, date=date, pass_type="descending", polar="vertical", freq_label="19ghz", output_dir=OUTPUT_DIR_V_DAY)
            else : 
                print("\n✅ [19GHz] - Descending map already generated - Vertical polarization")

            if new_graph or not os.path.exists(comb_plot_path_tb_19v):
                print("\n📊 Visualisation of Combined datas")
                plot_bt_map(df=pd.concat([df_ascending_19v, df_descending_19v]), date=date, pass_type="combined", polar="vertical", freq_label="19ghz", output_dir=OUTPUT_DIR_V_DAY)
            else : 
                print("\n✅ [19GHz] - Combined map already generated - Vertical polarization\n")
            
        print(f"\n📊✅ 19GHz Maps completed for date : {date} - Vertical polarization\n")

        # Maps Generation - Horizontal polarization - 19GHz
        if output_ascending_19h and output_descending_19h:
        
            print(f"\n===== AMSR-E Map Generation : TB_19GHz. Date : {date} =====")

            df_ascending_19h = pd.read_csv(output_ascending_19h)
            df_descending_19h = pd.read_csv(output_descending_19h)

            asc_plot_path_tb_19h = OUTPUT_DIR_H + f"/dates/{date}/tb_19ghz_map_{date}_ascending.png"
            des_plot_path_tb_19h = OUTPUT_DIR_H + f"/dates/{date}/tb_19ghz_map_{date}_descending.png"
            comb_plot_path_tb_19h = OUTPUT_DIR_H + f"/dates/{date}/tb_19ghz_map_{date}.png"

            if new_graph or not os.path.exists(asc_plot_path_tb_19h):
                print("\n📈 visualisation of Ascending")
                plot_bt_map(df=df_ascending_19h, date=date, pass_type="ascending", polar="horizontal", freq_label="19ghz", output_dir=OUTPUT_DIR_H_DAY)
            else : 
                print("\n✅ [19GHz] - Ascending TB map already generated")

            if new_graph or not os.path.exists(des_plot_path_tb_19h):
                print("\n📉 Visualisation of Descending")
                plot_bt_map(df_descending_19h, date, pass_type="descending", polar="horizontal", freq_label="19ghz", output_dir=OUTPUT_DIR_H_DAY)
            else : 
                print("\n✅ [19GHz] - Descending map already generated")

            if new_graph or not os.path.exists(comb_plot_path_tb_19h):
                print("\n📊 Visualisation of Combined datas")
                plot_bt_map(pd.concat([df_ascending_19h, df_descending_19h]), date, pass_type="combined", polar="horizontal", freq_label="19ghz", output_dir=OUTPUT_DIR_H_DAY)
            else : 
                print("\n✅ [19GHz] - Combined map already generated\n")
            
        print(f"\n📊✅ 19GHz Maps completed for date : {date} - Horizontal polarization\n")
    

    print("\n===== END of AMSR-E TB and Temperature by Regression =====")
    '''
    

    ### FLUXNET & PLOTS PART ###
    

    '''
    print("\n=====📥 Analysis with FLUXNET 📥=====")

    fluxnet_path = "data/raw/fluxnet/FluxNET_AMSRE.csv"
    coords_path = "data/processed/fluxnet/fluxnet_station_coordinates.csv"
    tb_base_folder = "data/processed/amsre"
    matched_output_base = "data/processed/amsre/matched"

    # Charger les données FLUXNET une seule fois
    df_fluxnet_all = pd.read_csv(fluxnet_path, sep=';')
    df_fluxnet_all["TIMESTAMP_START"] = pd.to_datetime(df_fluxnet_all["TIMESTAMP_START"], format="%d/%m/%Y")

    current_date = start_date

    frequencies = ["37GHz", "19GHz"]
    polarizations = ["vertical", "horizontal"]  # vertical et horizontal

    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        date_folder = current_date.strftime("%Y-%m-%d")

        for freq in frequencies:
            for pol in polarizations:

                tb_folder = os.path.join(tb_base_folder, pol + "_polarization",date_folder)
                output_folder = os.path.join(matched_output_base, freq.lower(), pol)
                os.makedirs(output_folder, exist_ok=True)
                output_csv = os.path.join(output_folder, f"matched_tb_fluxnet_{date_str}.csv")

                if os.path.exists(output_csv):
                    print(f"📂 Match file already exists for {date_str} - {freq} {pol.upper()}, skipping.")
                    continue

                print(f"\n===== Matching FLUXNET & AMSR-E for {date_str} - {freq} {pol.upper()} =====")
                generate_daily_matches(
                    start_date=current_date,
                    end_date=current_date,
                    freq_label=freq,
                    polarization=pol,
                    fluxnet_path=fluxnet_path,
                    coords_path=coords_path,
                    tb_folder=tb_folder,
                    output_folder=output_folder
                )

                if os.path.exists(output_csv):
                    df_check = pd.read_csv(output_csv)
                    if df_check.empty:
                        print(f"⚠️ Fichier vide, aucun graphique généré pour {date_str} - {freq} {pol.upper()}")
                    else:
                        reg_plot_path = f"outputs/amsre/dates/{date_folder}/regression_tb_vs_temp_{date_folder}_{freq}_{pol}.png"
                        if new_graph or not os.path.exists(reg_plot_path):
                            print(f"\n📈 Génération du graphique de régression pour {date_str} - {freq} {pol.upper()}")
                            #plot_brightness_vs_temperature_and_regression(output_csv, date_folder, freq, pol)
                        else:
                            print(f"✅ Graphique déjà généré pour {date_str} - {freq} {pol.upper()}")
                else:
                    print(f"⚠️ Fichier manquant : {output_csv}")

        current_date += timedelta(days=1)
    
    
    # === Day-by-day linear regression after processing all dates ===

    # == VERTICAL 37GHz == 
    print("\n===== ↕️ Daily regression TB vs Temperature (multi-day) for the 37GHz frequency - vertical =====")

    matched_output_folder_37v = "data/processed/amsre/matched/37ghz/vertical"
    output_regression_csv_37v = "data/analysis/daily_AMSRE_regressions_37GHz_vertical.csv"
    output_regression_metrics_37v = "outputs/amsre/vertical_polarization/regression_metrics_evolution_37GHz_vertical.png"
    output_global_tb_temp_37v = "outputs/amsre/vertical_polarization/global_tb_vs_temp_37GHz_vertical.png"

    fit_daily_regressions(folder_path=matched_output_folder_37v, output_csv_path=output_regression_csv_37v, freq_label="37GHz", polar="vertical")

    if new_graph or not os.path.exists(output_regression_metrics_37v):
        print(f"\n📊 Evolution of the regression metrics for 37GHz")
        plot_regression_metrics_evolution(regression_csv_path=output_regression_csv_37v, output_path=output_regression_metrics_37v, freq_label="37GHz", polar="vertical")
    else : 
        print(f"\n✅ Evolution of the regression metrics already generated for 37GHz : {output_regression_metrics_37v}")
    
    if new_graph or not os.path.exists(output_global_tb_temp_37v):
        print(f"\n📊 Evolution of the global tb(temp) plot for 37GHz")
        aglob37v, bglob37v = plot_global_tb_vs_temp(matched_folder="data/processed/amsre/matched/37GHz/vertical", output_path=output_global_tb_temp_37v, new_graph=True, freq_label="37GHz", polar="vertical")
    else : 
        print(f"\n✅ Evolution of the global tb(temp) plot already generated : {output_global_tb_temp_37v}")
        aglob37v, bglob37v = plot_global_tb_vs_temp(matched_folder="data/processed/amsre/matched/37GHz/vertical", output_path=output_global_tb_temp_37v, new_graph=False, freq_label="37GHz", polar="vertical")
    

    # == HORIZONTAL 37GHz ==
    print("\n===== ↔️ Daily regression TB vs Temperature (multi-day) for the 37GHz frequency - horizontal =====")

    matched_output_folder_37h = "data/processed/amsre/matched/37ghz/horizontal"
    output_regression_csv_37h = "data/analysis/daily_AMSRE_regressions_37GHz_horizontal.csv"
    output_regression_metrics_37h = "outputs/amsre/horizontal_polarization/regression_metrics_evolution_37GHz_horizontal.png"
    output_global_tb_temp_37h = "outputs/amsre/horizontal_polarization/global_tb_vs_temp_37GHz_horizontal.png"

    fit_daily_regressions(folder_path=matched_output_folder_37h, output_csv_path=output_regression_csv_37h, freq_label="37GHz", polar="horizontal")

    if new_graph or not os.path.exists(output_regression_metrics_37h):
        print(f"\n📊 Evolution of the regression metrics for 37GHz")
        plot_regression_metrics_evolution(regression_csv_path=output_regression_csv_37h, output_path=output_regression_metrics_37h, freq_label = "37GHz", polar="horizontal")
    else : 
        print(f"\n✅ Evolution of the regression metrics already generated for 37GHz : {output_regression_metrics_37h}")
    
    if new_graph or not os.path.exists(output_global_tb_temp_37h):
        print(f"\n📊 Evolution of the global tb(temp) plot for 37GHz")
        aglob37h, bglob37h = plot_global_tb_vs_temp(matched_folder="data/processed/amsre/matched/37GHz/horizontal", output_path=output_global_tb_temp_37h, new_graph=True, freq_label="37GHz", polar="horizontal")
    else : 
        print(f"\n✅ Evolution of the global tb(temp) plot already generated : {output_global_tb_temp_37h}")
        aglob37h, bglob37h = plot_global_tb_vs_temp(matched_folder = "data/processed/amsre/matched/37GHz/horizontal", output_path=output_global_tb_temp_37h, new_graph=False, freq_label="37GHz", polar="horizontal")

    
    # == VERTICAL 19GHz == 
    print("\n===== ↕️ Daily regression TB vs Temperature (multi-day) for the 19GHz frequency - vertical =====")

    matched_output_folder_19v = "data/processed/amsre/matched/19ghz/vertical"
    output_regression_csv_19v = "data/analysis/daily_AMSRE_regressions_19GHz_vertical.csv"
    output_regression_metrics_19v = "outputs/amsre/vertical_polarization/regression_metrics_evolution_19GHz_vertical.png"
    output_global_tb_temp_19v = "outputs/amsre/vertical_polarization/global_tb_vs_temp_19GHz_vertical.png"

    fit_daily_regressions(folder_path=matched_output_folder_19v, output_csv_path=output_regression_csv_19v, freq_label="19GHz", polar="vertical")

    if new_graph or not os.path.exists(output_regression_metrics_19v):
        print(f"\n📊 Evolution of the regression metrics for 19GHz")
        plot_regression_metrics_evolution(regression_csv_path=output_regression_csv_19v, output_path=output_regression_metrics_19v, freq_label="19GHz", polar="vertical")
    else : 
        print(f"\n✅ Evolution of the regression metrics already generated for 19GHz : {output_regression_metrics_19v}")
    
    if new_graph or not os.path.exists(output_global_tb_temp_19v):
        print(f"\n📊 Evolution of the global tb(temp) plot for 19GHz")
        aglob19v, bglob19v = plot_global_tb_vs_temp(matched_folder="data/processed/amsre/matched/19GHz/vertical", output_path=output_global_tb_temp_19v, new_graph=True, freq_label="19GHz", polar="vertical")
    else : 
        print(f"\n✅ Evolution of the global tb(temp) plot already generated : {output_global_tb_temp_19v}")
        aglob19v, bglob19v = plot_global_tb_vs_temp(matched_folder="data/processed/amsre/matched/19GHz/vertical", output_path=output_global_tb_temp_19v, new_graph=False, freq_label="19GHz", polar="vertical")
    

    # == HORIZONTAL 19GHz ==
    print("\n===== ↔️ Daily regression TB vs Temperature (multi-day) for the 19GHz frequency - horizontal =====")

    matched_output_folder_19h = "data/processed/amsre/matched/19ghz/horizontal"
    output_regression_csv_19h = "data/analysis/daily_AMSRE_regressions_19GHz_horizontal.csv"
    output_regression_metrics_19h = "outputs/amsre/horizontal_polarization/regression_metrics_evolution_19GHz_horizontal.png"
    output_global_tb_temp_19h = "outputs/amsre/horizontal_polarization/global_tb_vs_temp_19GHz_horizontal.png"

    fit_daily_regressions(folder_path=matched_output_folder_19h, output_csv_path=output_regression_csv_19h, freq_label="19GHz", polar="horizontal")

    if new_graph or not os.path.exists(output_regression_metrics_19h):
        print(f"\n📊 Evolution of the regression metrics for 19GHz")
        plot_regression_metrics_evolution(regression_csv_path=output_regression_csv_19h, output_path=output_regression_metrics_19h, freq_label = "19GHz", polar="horizontal")
    else : 
        print(f"\n✅ Evolution of the regression metrics already generated for 19GHz : {output_regression_metrics_19h}")
    
    if new_graph or not os.path.exists(output_global_tb_temp_19h):
        print(f"\n📊 Evolution of the global tb(temp) plot for 19GHz")
        aglob19h, bglob19h = plot_global_tb_vs_temp(matched_folder="data/processed/amsre/matched/19GHz/horizontal", output_path=output_global_tb_temp_19h, new_graph=True, freq_label="19GHz", polar="horizontal")
    else : 
        print(f"\n✅ Evolution of the global tb(temp) plot already generated : {output_global_tb_temp_19h}")
        aglob19h, bglob19h = plot_global_tb_vs_temp(matched_folder = "data/processed/amsre/matched/19GHz/horizontal", output_path=output_global_tb_temp_19h, new_graph=False, freq_label="19GHz", polar="horizontal")

        
    print("\n===== Régressions TB vs Température for each station =====")
    all_matched_df_37v = pd.concat([pd.read_csv(os.path.join(matched_output_folder_37v, f)) for f in os.listdir(matched_output_folder_37v) if f.endswith(".csv")],ignore_index=True)
    all_matched_df_37h = pd.concat([pd.read_csv(os.path.join(matched_output_folder_37h, f)) for f in os.listdir(matched_output_folder_37h) if f.endswith(".csv")],ignore_index=True)
    all_matched_df_19v = pd.concat([pd.read_csv(os.path.join(matched_output_folder_19v, f)) for f in os.listdir(matched_output_folder_19v) if f.endswith(".csv")],ignore_index=True)
    all_matched_df_19h = pd.concat([pd.read_csv(os.path.join(matched_output_folder_19h, f)) for f in os.listdir(matched_output_folder_19h) if f.endswith(".csv")],ignore_index=True)
    plot_station_regressions(df_matched_37v=all_matched_df_37v, df_matched_37h=all_matched_df_37h, df_matched_19v=all_matched_df_19v, df_matched_19h=all_matched_df_19h, output_dir="outputs/amsre/stations", new_graph=new_graph)

    
    print("\n===== Regression for each station & overall station regression =====")
    output_dir = "outputs/fluxnet"
    # Checks whether a single plot already exists, otherwise call the
    example_station = "FLX_FR-LBr_FLUXNET2015_FULLSET_1996-2008_1-4"  
    example_path = os.path.join(output_dir, f"regression_2005_{example_station}.png")
    if new_graph or not os.path.exists(example_path):
        plot_stationwise_and_global_regressions_2005(csv_path = "data/raw/fluxnet/FluxNET_AMSRE.csv", freq_label = "37GHz", output_dir = output_dir)
    else:
        print("⏭️ Graphics of stations already present, skip.")
    
    
    print("\n===== Évolution saisonnière température + TB AMSR-E (comparaison) =====")
    seasonal_tb_dir = "outputs/fluxnet"
    temp_with_tb_path_37ghz = os.path.join(seasonal_tb_dir, "evolution_temp_tb_37ghz.png")
    temp_with_tb_path_19ghz = os.path.join(seasonal_tb_dir, "evolution_temp_tb_19ghz.png")

    if new_graph or not os.path.exists(temp_with_tb_path_37ghz):
        plot_seasonal_temp_with_tb_evolution(matched_folder1="data/processed/amsre/matched/37GHz/vertical", matched_folder2="data/processed/amsre/matched/37GHz/horizontal", output_dir="outputs/fluxnet", tb_min_threshold=230, freq_label="37ghz")
    else:
        print("⏭️ TB vs Temp graphics already generated for 37GHz, skip.")

    if new_graph or not os.path.exists(temp_with_tb_path_19ghz):    
        plot_seasonal_temp_with_tb_evolution(matched_folder1="data/processed/amsre/matched/19GHz/vertical", matched_folder2="data/processed/amsre/matched/19GHz/horizontal", output_dir="outputs/fluxnet", tb_min_threshold=230, freq_label="19ghz")
    else:
        print("⏭️ TB vs Temp graphics already generated for 19GHz, skip.")
    
        
    print("\n===== Évolution temporelle de toutes les températures =====")
    output_path = os.path.join(seasonal_tb_dir, f"temp_by_station.png")
    if new_graph or not os.path.exists(output_path):
        plot_all_stations_temp_evolution(csv_path = "data/raw/fluxnet/FluxNET_AMSRE.csv", output_path=output_path)
    else:
        print("⏭️ All temp graphic already generated, skip.")
    '''


    ### Temperature generated from linear regression ### A EXECUTER POUR LES GRAPHES ET LES CSV

    '''
    # Plotting supposed temperature maps - AMSRE
    for date in dates :          
        if date == '2005-11-17' :
            print("\n❌ No data avalaible for this date : 2005-11-17")
            continue

        # Vertical 
        output_ascending_37v = f"data/processed/amsre/vertical_polarization/{date}/amsre_37GHz_vertical_{date}_ascending.csv"
        output_descending_37v = f"data/processed/amsre/vertical_polarization/{date}/amsre_37GHz_vertical_{date}_descending.csv"
        output_ascending_19v = f"data/processed/amsre/vertical_polarization/{date}/amsre_19GHz_vertical_{date}_ascending.csv"
        output_descending_19v = f"data/processed/amsre/vertical_polarization/{date}/amsre_19GHz_vertical_{date}_descending.csv"

        df_ascending_37v = pd.read_csv(output_ascending_37v)
        df_descending_37v = pd.read_csv(output_descending_37v)
        df_ascending_19v = pd.read_csv(output_ascending_19v)
        df_descending_19v = pd.read_csv(output_descending_19v)
        
        # Horizontal 
        output_ascending_37h = f"data/processed/amsre/horizontal_polarization/{date}/amsre_37GHz_horizontal_{date}_ascending.csv"
        output_descending_37h = f"data/processed/amsre/horizontal_polarization/{date}/amsre_37GHz_horizontal_{date}_descending.csv"
        output_ascending_19h = f"data/processed/amsre/horizontal_polarization/{date}/amsre_19GHz_horizontal_{date}_ascending.csv"
        output_descending_19h = f"data/processed/amsre/horizontal_polarization/{date}/amsre_19GHz_horizontal_{date}_descending.csv"

        df_ascending_37h = pd.read_csv(output_ascending_37h)
        df_descending_37h = pd.read_csv(output_descending_37h)
        df_ascending_19h = pd.read_csv(output_ascending_19h)
        df_descending_19h = pd.read_csv(output_descending_19h)


        ### Temperatures generated from linear regression - 37 GHz - vertical
                
        asc_plot_path_regtemp_37v = f"outputs/amsre/vertical_polarization/dates/{date}/estimated_temperature/37ghz/temp_by_reg_37ghz_vertical_map_{date}_ascending.png"
        des_plot_path_regtemp_37v = f"outputs/amsre/vertical_polarization/dates/{date}/estimated_temperature/37ghz/temp_by_reg_37ghz_vertical_map_{date}_descending.png"
        comb_plot_path_regtemp_37v = f"outputs/amsre/vertical_polarization/dates/{date}/estimated_temperature/37ghz/temp_by_reg_37ghz_vertical_map_{date}_combined.png"

        print(f"\n📊 37GHz vertical Supposed Temperature Maps for date : {date}")
        
        if new_graph or not os.path.exists(asc_plot_path_regtemp_37v):
            print("\n📈 Visualisation of Ascending Supposed Temperature")
            plot_temp_estimated_map(df_ascending_37v, date, pass_type="ascending", freq_label="37ghz", a=aglob37v, b=bglob37v, polar="vertical")
        else : 
            print("\n✅ [37GHz] vertical - Ascending supposed temperatures map already generated")

        if new_graph or not os.path.exists(des_plot_path_regtemp_37v):    
            print("\n📉 Visualisation of Descending Supposed Temperature")
            plot_temp_estimated_map(df_descending_37v, date, pass_type="descending", freq_label="37ghz", a=aglob37v, b=bglob37v, polar="vertical")
        else : 
            print("\n✅ [37GHz] vertical - Descending supposed temperatures map already generated")

        if new_graph or not os.path.exists(comb_plot_path_regtemp_37v):   
            print("\n📊 Visualisation of Combined Supposed Temperature datas") 
            plot_temp_estimated_map(pd.concat([df_ascending_37v, df_descending_37v]), date, pass_type="combined", freq_label="37ghz", a=aglob37v, b=bglob37v, polar="vertical")
        else : 
            print("\n✅ [37GHz] vertical - Combined supposed temperatures map already generated")       

        ### Temperatures generated from linear regression - 19 GHz - vertical
                
        asc_plot_path_regtemp_19v = f"outputs/amsre/vertical_polarization/dates/{date}/estimated_temperature/19ghz/temp_by_reg_19ghz_vertical_map_{date}_ascending.png"
        des_plot_path_regtemp_19v = f"outputs/amsre/vertical_polarization/dates/{date}/estimated_temperature/19ghz/temp_by_reg_19ghz_vertical_map_{date}_descending.png"
        comb_plot_path_regtemp_19v = f"outputs/amsre/vertical_polarization/dates/{date}/estimated_temperature/19ghz/temp_by_reg_19ghz_vertical_map_{date}_combined.png"

        print(f"\n📊 19GHz vertical Supposed Temperature Maps for date : {date}")
        
        if new_graph or not os.path.exists(asc_plot_path_regtemp_19v):
            print("\n📈 Visualisation of Ascending Supposed Temperature")
            plot_temp_estimated_map(df_ascending_19v, date, pass_type="ascending", freq_label="19ghz", a=aglob19v, b=bglob19v, polar="vertical")
        else : 
            print("\n✅ [19GHz] vertical - Ascending supposed temperatures map already generated")

        if new_graph or not os.path.exists(des_plot_path_regtemp_19v):    
            print("\n📉 Visualisation of Descending Supposed Temperature")
            plot_temp_estimated_map(df_descending_19v, date, pass_type="descending", freq_label="19ghz", a=aglob19v, b=bglob19v, polar="vertical")
        else : 
            print("\n✅ [19GHz] vertical - Descending supposed temperatures map already generated")

        if new_graph or not os.path.exists(comb_plot_path_regtemp_19v):   
            print("\n📊 Visualisation of Combined Supposed Temperature datas") 
            plot_temp_estimated_map(pd.concat([df_ascending_19v, df_descending_19v]), date, pass_type="combined", freq_label="19ghz", a=aglob19v, b=bglob19v, polar="vertical")
        else : 
            print("\n✅ [19GHz] vertical - Combined supposed temperatures map already generated")
    
        ### Temperatures generated from linear regression - 37 GHz - horizontal
                
        asc_plot_path_regtemp_37h = f"outputs/amsre/horizontal_polarization/dates/{date}/estimated_temperature/37ghz/temp_by_reg_37ghz_horizontal_map_{date}_ascending.png"
        des_plot_path_regtemp_37h = f"outputs/amsre/horizontal_polarization/dates/{date}/estimated_temperature/37ghz/temp_by_reg_37ghz_horizontal_map_{date}_descending.png"
        comb_plot_path_regtemp_37h = f"outputs/amsre/horizontal_polarization/dates/{date}/estimated_temperature/37ghz/temp_by_reg_37ghz_horizontal_map_{date}_combined.png"

        print(f"\n📊 37GHz horizontal Supposed Temperature Maps for date : {date}")
        
        if new_graph or not os.path.exists(asc_plot_path_regtemp_37h):
            print("\n📈 Visualisation of Ascending Supposed Temperature")
            plot_temp_estimated_map(df_ascending_37h, date, pass_type="ascending", freq_label="37ghz", a=aglob37h, b=bglob37h, polar="horizontal")
        else : 
            print("\n✅ [37GHz] horizontal - Ascending supposed temperatures map already generated")

        if new_graph or not os.path.exists(des_plot_path_regtemp_37h):    
            print("\n📉 Visualisation of Descending Supposed Temperature")
            plot_temp_estimated_map(df_descending_37h, date, pass_type="descending", freq_label="37ghz", a=aglob37h, b=bglob37h, polar="horizontal")
        else : 
            print("\n✅ [37GHz] horizontal - Descending supposed temperatures map already generated")

        if new_graph or not os.path.exists(comb_plot_path_regtemp_37h):   
            print("\n📊 Visualisation of Combined Supposed Temperature datas") 
            plot_temp_estimated_map(pd.concat([df_ascending_37h, df_descending_37h]), date, pass_type="combined", freq_label="37ghz", a=aglob37h, b=bglob37h, polar="horizontal")
        else : 
            print("\n✅ [37GHz] horizontal - Combined supposed temperatures map already generated")       

        ### Temperatures generated from linear regression - 19 GHz - horizontal
                
        asc_plot_path_regtemp_19h = f"outputs/amsre/horizontal_polarization/dates/{date}/estimated_temperature/19ghz/temp_by_reg_19ghz_horizontal_map_{date}_ascending.png"
        des_plot_path_regtemp_19h = f"outputs/amsre/horizontal_polarization/dates/{date}/estimated_temperature/19ghz/temp_by_reg_19ghz_horizontal_map_{date}_descending.png"
        comb_plot_path_regtemp_19h = f"outputs/amsre/horizontal_polarization/dates/{date}/estimated_temperature/19ghz/temp_by_reg_19ghz_horizontal_map_{date}_combined.png"

        print(f"\n📊 19GHz horizontal Supposed Temperature Maps for date : {date}")
        
        if new_graph or not os.path.exists(asc_plot_path_regtemp_19h):
            print("\n📈 Visualisation of Ascending Supposed Temperature")
            plot_temp_estimated_map(df_ascending_19h, date, pass_type="ascending", freq_label="19ghz", a=aglob19h, b=bglob19h, polar="horizontal")
        else : 
            print("\n✅ [19GHz] horizontal - Ascending supposed temperatures map already generated")

        if new_graph or not os.path.exists(des_plot_path_regtemp_19h):    
            print("\n📉 Visualisation of Descending Supposed Temperature")
            plot_temp_estimated_map(df_descending_19h, date, pass_type="descending", freq_label="19ghz", a=aglob19h, b=bglob19h, polar="horizontal")
        else : 
            print("\n✅ [19GHz] horizontal - Descending supposed temperatures map already generated")

        if new_graph or not os.path.exists(comb_plot_path_regtemp_19h):   
            print("\n📊 Visualisation of Combined Supposed Temperature datas") 
            plot_temp_estimated_map(pd.concat([df_ascending_19h, df_descending_19h]), date, pass_type="combined", freq_label="19ghz", a=aglob19h, b=bglob19h, polar="horizontal")
        else : 
            print("\n✅ [19GHz] horizontal - Combined supposed temperatures map already generated")
    '''


    ### Average temperature maps ### 


    '''
    input_dir = "data/processed/modis" 
    csv_dir_Kelvin = "data/processed/modis/mean_temperature/mean_temp_2005_Kelvin.csv"
    csv_dir_Celsius = "data/processed/modis/mean_temperature/mean_temp_2005_Celsius.csv"
    output_file_Kelvin = "outputs/modis/mean_temp_2005_Kelvin.png"
    output_file_Celsius = "outputs/modis/mean_temp_2005_Celsius.png"

    csv_dir_19v_kelvin = "data/processed/amsre/vertical_polarization/mean_temp_2005_19GHz_Kelvin.csv"
    csv_dir_19h_kelvin = "data/processed/amsre/horizontal_polarization/mean_temp_2005_19GHz_Kelvin.csv"
    csv_dir_37v_kelvin = "data/processed/amsre/vertical_polarization/mean_temp_2005_37GHz_Kelvin.csv"
    csv_dir_37h_kelvin = "data/processed/amsre/horizontal_polarization/mean_temp_2005_37GHz_Kelvin.csv"

    csv_dir_19v_celsius = "data/processed/amsre/vertical_polarization/mean_temp_2005_19GHz_Celsius.csv"
    csv_dir_19h_celsius = "data/processed/amsre/horizontal_polarization/mean_temp_2005_19GHz_Celsius.csv"
    csv_dir_37v_celsius = "data/processed/amsre/vertical_polarization/mean_temp_2005_37GHz_Celsius.csv"
    csv_dir_37h_celsius = "data/processed/amsre/horizontal_polarization/mean_temp_2005_37GHz_Celsius.csv"

    map_dir_19v_kelvin = "outputs/amsre/mean_values_maps/mean_temp_2005_19GHz_vertical_Kelvin.png"
    map_dir_19h_kelvin = "outputs/amsre/mean_values_maps/mean_temp_2005_19GHz_horizontal_Kelvin.png"
    map_dir_37v_kelvin = "outputs/amsre/mean_values_maps/mean_temp_2005_37GHz_vertical_Kelvin.png"
    map_dir_37h_kelvin = "outputs/amsre/mean_values_maps/mean_temp_2005_37GHz_horizontal_Kelvin.png"

    map_dir_19v_celsius = "outputs/amsre/mean_values_maps/mean_temp_2005_19GHz_vertical_Celsius.png"
    map_dir_19h_celsius = "outputs/amsre/mean_values_maps/mean_temp_2005_19GHz_horizontal_Celsius.png"
    map_dir_37v_celsius = "outputs/amsre/mean_values_maps/mean_temp_2005_37GHz_vertical_Celsius.png"
    map_dir_37h_celsius = "outputs/amsre/mean_values_maps/mean_temp_2005_37GHz_horizontal_Celsius.png"

    input_dir_vertical = "data/processed/amsre/vertical_polarization"
    input_dir_horizontal = "data/processed/amsre/horizontal_polarization"

    # AMSRE #
    
    # [19GHz] - Vertical 
    if new_graph or not os.path.exists(map_dir_19v_kelvin): 
        print("\n📊 19GHz vertical - Generation of the AMSRE average annual calculated temperature map in Kelvin ")
        plot_temp_mean_amsre_Kelvin(freq_label="19ghz", polar="vertical", input_dir=input_dir_vertical, csv_out=csv_dir_19v_kelvin, map_out=map_dir_19v_kelvin)
        print(f"\n✅ AMSRE (vertical polarization) average annual calculated temperature map in Kelvin generated at {map_dir_19v_kelvin}")
    else : 
        print("\n✅ AMSRE (vertical polarization) average annual calculated temperature map in Kelvin already generated")

    if new_graph or not os.path.exists(map_dir_19v_celsius): 
        print("\n📊 19GHz vertical - Generation of the AMSRE average annual calculated temperature map in celsius ")
        plot_temp_mean_amsre_Celsius(freq_label="19ghz", polar="vertical", csv_kelvin=csv_dir_19v_kelvin, csv_out=csv_dir_19v_celsius, map_out=map_dir_19v_celsius)
        print(f"\n✅ AMSRE (vertical polarization) average annual calculated temperature map in celsius generated at {map_dir_19v_celsius}")
    else : 
        print("\n✅ AMSRE (vertical polarization) average annual calculated temperature map in celsius already generated")
    

    # [19GHz] - Horizontal
    if new_graph or not os.path.exists(map_dir_19h_kelvin): 
        print("\n📊 19GHz horizontal - Generation of the AMSRE average annual calculated temperature map in Kelvin ")
        plot_temp_mean_amsre_Kelvin(freq_label="19ghz", polar="horizontal", input_dir=input_dir_horizontal, csv_out=csv_dir_19h_kelvin, map_out=map_dir_19h_kelvin)
        print(f"\n✅ AMSRE (horizontal polarization) average annual calculated temperature map in Kelvin generated at {map_dir_19h_kelvin}")
    else : 
        print("\n✅ AMSRE (horizontal polarization) average annual calculated temperature map in Kelvin already generated")

    if new_graph or not os.path.exists(map_dir_19h_celsius): 
        print("\n📊 19GHz horizontal - Generation of the AMSRE average annual calculated temperature map in celsius ")
        plot_temp_mean_amsre_Celsius(freq_label="19ghz", polar="horizontal", csv_kelvin=csv_dir_19h_kelvin, csv_out=csv_dir_19h_celsius, map_out=map_dir_19h_celsius)
        print(f"\n✅ AMSRE (horizontal polarization) average annual calculated temperature map in celsius generated at {map_dir_19h_celsius}")
    else : 
        print("\n✅ AMSRE (horizontal polarization) average annual calculated temperature map in celsius already generated")
    

    # [37GHz] - Vertical 
    if new_graph or not os.path.exists(map_dir_37v_kelvin): 
        print("\n📊 37GHz vertical - Generation of the AMSRE average annual calculated temperature map in Kelvin ")
        plot_temp_mean_amsre_Kelvin(freq_label="37ghz", polar="vertical", input_dir=input_dir_vertical, csv_out=csv_dir_37v_kelvin, map_out=map_dir_37v_kelvin)
        print(f"\n✅ AMSRE (vertical polarization) average annual calculated temperature map in Kelvin generated at {map_dir_37v_kelvin}")
    else : 
        print("\n✅ AMSRE (vertical polarization) average annual calculated temperature map in Kelvin already generated")

    if new_graph or not os.path.exists(map_dir_37v_celsius): 
        print("\n📊 37GHz vertical - Generation of the AMSRE average annual calculated temperature map in celsius ")
        plot_temp_mean_amsre_Celsius(freq_label="37ghz", polar="vertical", csv_kelvin=csv_dir_37v_kelvin, csv_out=csv_dir_37v_celsius, map_out=map_dir_37v_celsius)
        print(f"\n✅ AMSRE (vertical polarization) average annual calculated temperature map in celsius generated at {map_dir_37v_celsius}")
    else : 
        print("\n✅ AMSRE (vertical polarization) average annual calculated temperature map in celsius already generated")
    

    # [37GHz] - Horizontal
    if new_graph or not os.path.exists(map_dir_37h_kelvin): 
        print("\n📊 37GHz horizontal - Generation of the AMSRE average annual calculated temperature map in Kelvin ")
        plot_temp_mean_amsre_Kelvin(freq_label="37ghz", polar="horizontal", input_dir=input_dir_horizontal, csv_out=csv_dir_37h_kelvin, map_out=map_dir_37h_kelvin)
        print(f"\n✅ AMSRE (horizontal polarization) average annual calculated temperature map in Kelvin generated at {map_dir_37h_kelvin}")
    else : 
        print("\n✅ AMSRE (horizontal polarization) average annual calculated temperature map in Kelvin already generated")

    if new_graph or not os.path.exists(map_dir_37h_celsius): 
        print("\n📊 37GHz horizontal - Generation of the AMSRE average annual calculated temperature map in celsius ")
        plot_temp_mean_amsre_Celsius(freq_label="37ghz", polar="horizontal", csv_kelvin=csv_dir_37h_kelvin, csv_out=csv_dir_37h_celsius, map_out=map_dir_37h_celsius)
        print(f"\n✅ AMSRE (horizontal polarization) average annual calculated temperature map in celsius generated at {map_dir_37h_celsius}")
    else : 
        print("\n✅ AMSRE (horizontal polarization) average annual calculated temperature map in celsius already generated")
    
    
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
        print("\n✅ Map of average annual temperatures in Celsius already generated")
    '''


    ### Comparisons MODIS - AMSRE average maps ###


    '''
    # === MODIS vs AMSRE - Kelvin ===

    modis_csv_path_kelvin = "data/processed/modis/mean_temperature/mean_temp_2005_Kelvin.csv"

    amsre_csv_path_19v_kelvin = "data/processed/amsre/vertical_polarization/mean_temp_2005_19GHz_Kelvin.csv"
    amsre_csv_path_19h_kelvin = "data/processed/amsre/horizontal_polarization/mean_temp_2005_19GHz_Kelvin.csv"
    amsre_csv_path_37v_kelvin = "data/processed/amsre/vertical_polarization/mean_temp_2005_37GHz_Kelvin.csv"
    amsre_csv_path_37h_kelvin = "data/processed/amsre/horizontal_polarization/mean_temp_2005_37GHz_Kelvin.csv"

    output_map_modis_19v_kelvin = "outputs/comparisons_modis_amsre/diff_MODIS_19GHz_vertical_Kelvin.png"
    output_map_modis_19h_kelvin = "outputs/comparisons_modis_amsre/diff_MODIS_19GHz_horizontal_Kelvin.png"
    output_map_modis_37v_kelvin = "outputs/comparisons_modis_amsre/diff_MODIS_37GHz_vertical_Kelvin.png"
    output_map_modis_37h_kelvin = "outputs/comparisons_modis_amsre/diff_MODIS_37GHz_horizontal_Kelvin.png"

    # = MODIS vs 19GHz (vertical polarization) =
    if new_graph or not os.path.exists(output_map_modis_19v_kelvin): 
        print("\n📊 Generation of the difference average annual temperature map between MODIS et AMSRE 19GHz (vertical polarization) in Kelvin")
        
        plot_difference_map_explicit(
            modis_csv=modis_csv_path_kelvin,
            amsre_csv=amsre_csv_path_19v_kelvin,
            modis_col="LST_Kelvin_mean",
            amsre_col="temp_K_mean",
            output_path=output_map_modis_19v_kelvin,
            title="Différence MODIS - AMSRE (19GHz) - vertical polarization [Kelvin]",
            color_label="Temperature differences (K)")
        
        print(f"🖼️ Map of the difference average annual temperatures between MODIS et AMSRE 19GHz (vertical polarization) in Kelvin generated at : {output_map_modis_19v_kelvin}")
    else : 
        print("\n✅ Map of the difference average annual temperatures between MODIS et AMSRE 19GHz (vertical polarization) in Kelvin already generated")


    # = MODIS vs 19GHz (horizontal polarization) =
    if new_graph or not os.path.exists(output_map_modis_19h_kelvin): 
        print("\n📊 Generation of the difference average annual temperature map between MODIS et AMSRE 19GHz (horizontal polarization) in Kelvin")
        
        plot_difference_map_explicit(
            modis_csv=modis_csv_path_kelvin,
            amsre_csv=amsre_csv_path_19h_kelvin,
            modis_col="LST_Kelvin_mean",
            amsre_col="temp_K_mean",
            output_path=output_map_modis_19h_kelvin,
            title="Différence MODIS - AMSRE (19GHz) - horizontal polarization [Kelvin]",
            color_label="Temperature differences (K)")
        
        print(f"🖼️ Map of the difference average annual temperatures between MODIS et AMSRE 19GHz (horizontal polarization) in Kelvin generated at : {output_map_modis_19h_kelvin}")
    else : 
        print("\n✅ Map of the difference average annual temperatures between MODIS et AMSRE 19GHz (horizontal polarization) in Kelvin already generated")


    # = MODIS vs 37GHz (vertical polarization) =
    if new_graph or not os.path.exists(output_map_modis_37v_kelvin): 
        print("\n📊 Generation of the difference average annual temperature map between MODIS et AMSRE 37GHz (vertical polarization) in Kelvin")
        
        plot_difference_map_explicit(
            modis_csv=modis_csv_path_kelvin,
            amsre_csv=amsre_csv_path_37v_kelvin,
            modis_col="LST_Kelvin_mean",
            amsre_col="temp_K_mean",
            output_path=output_map_modis_37v_kelvin,
            title="Différence MODIS - AMSRE (37GHz) - vertical polarization [Kelvin]",
            color_label="Temperature differences (K)")
        
        print(f"🖼️ Map of the difference average annual temperatures between MODIS et AMSRE 37GHz (vertical polarization) in Kelvin generated at : {output_map_modis_37v_kelvin}")
    else : 
        print("\n✅ Map of the difference average annual temperatures between MODIS et AMSRE 37GHz (vertical polarization) in Kelvin already generated")


    # = MODIS vs 37GHz (horizontal polarization) =
    if new_graph or not os.path.exists(output_map_modis_37h_kelvin): 
        print("\n📊 Generation of the difference average annual temperature map between MODIS et AMSRE 37GHz (horizontal polarization) in Kelvin")
        
        plot_difference_map_explicit(
            modis_csv=modis_csv_path_kelvin,
            amsre_csv=amsre_csv_path_37h_kelvin,
            modis_col="LST_Kelvin_mean",
            amsre_col="temp_K_mean",
            output_path=output_map_modis_37h_kelvin,
            title="Différence MODIS - AMSRE (37GHz) - horizontal polarization [Kelvin]",
            color_label="Temperature differences (K)")
        
        print(f"🖼️ Map of the difference average annual temperatures between MODIS et AMSRE 37GHz (horizontal polarization) in Kelvin generated at : {output_map_modis_37h_kelvin}")
    else : 
        print("\n✅ Map of the difference average annual temperatures between MODIS et AMSRE 37GHz (horizontal polarization) in Kelvin already generated")


    # === MODIS vs AMSRE - Celsius ===

    modis_csv_path_celsius = "data/processed/modis/mean_temperature/mean_temp_2005_Celsius.csv"
    amsre_csv_path_19v_celsius = "data/processed/amsre/vertical_polarization/mean_temp_2005_19GHz_Celsius.csv"
    amsre_csv_path_19h_celsius = "data/processed/amsre/horizontal_polarization/mean_temp_2005_19GHz_Celsius.csv"
    amsre_csv_path_37v_celsius = "data/processed/amsre/vertical_polarization/mean_temp_2005_37GHz_Celsius.csv"
    amsre_csv_path_37h_celsius = "data/processed/amsre/horizontal_polarization/mean_temp_2005_37GHz_Celsius.csv"

    output_map_modis_19v_celsius = "outputs/comparisons_modis_amsre/diff_MODIS_19GHz_vertical_Celsius.png"
    output_map_modis_19h_celsius = "outputs/comparisons_modis_amsre/diff_MODIS_19GHz_horizontal_Celsius.png"
    output_map_modis_37v_celsius = "outputs/comparisons_modis_amsre/diff_MODIS_37GHz_vertical_Celsius.png"
    output_map_modis_37h_celsius = "outputs/comparisons_modis_amsre/diff_MODIS_37GHz_horizontal_Celsius.png"

    # = MODIS vs 19GHz (vertical polarization) =
    if new_graph or not os.path.exists(output_map_modis_19v_celsius): 
        print("\n📊 Generation of the difference average annual temperature map between MODIS et AMSRE 19GHz (vertical polarization) in Celsius")
        
        plot_difference_map_explicit(
            modis_csv=modis_csv_path_celsius,
            amsre_csv=amsre_csv_path_19v_celsius,
            modis_col="LST_Celsius_mean",
            amsre_col="temp_C_mean",
            output_path=output_map_modis_19v_celsius,
            title="Différence MODIS - AMSRE (19GHz) - vertical polarization [Celsius]",
            color_label="Temperature differences (K)")
        
        print(f"🖼️ Map of the difference average annual temperatures between MODIS et AMSRE 19GHz (vertical polarization) in Celsius generated at : {output_map_modis_19v_celsius}")
    else : 
        print("\n✅ Map of the difference average annual temperatures between MODIS et AMSRE 19GHz (vertical polarization) in Celsius already generated")


    # = MODIS vs 19GHz (horizontal polarization) =
    if new_graph or not os.path.exists(output_map_modis_19h_celsius): 
        print("\n📊 Generation of the difference average annual temperature map between MODIS et AMSRE 19GHz (horizontal polarization) in Celsius")
        
        plot_difference_map_explicit(
            modis_csv=modis_csv_path_celsius,
            amsre_csv=amsre_csv_path_19h_celsius,
            modis_col="LST_Celsius_mean",
            amsre_col="temp_C_mean",
            output_path=output_map_modis_19h_celsius,
            title="Différence MODIS - AMSRE (19GHz) - horizontal polarization [Celsius]",
            color_label="Temperature differences (K)")
        
        print(f"🖼️ Map of the difference average annual temperatures between MODIS et AMSRE 19GHz (horizontal polarization) in Celsius generated at : {output_map_modis_19h_celsius}")
    else : 
        print("\n✅ Map of the difference average annual temperatures between MODIS et AMSRE 19GHz (horizontal polarization) in Celsius already generated")


    # = MODIS vs 37GHz (vertical polarization) =
    if new_graph or not os.path.exists(output_map_modis_37v_celsius): 
        print("\n📊 Generation of the difference average annual temperature map between MODIS et AMSRE 37GHz (vertical polarization) in Celsius")
        
        plot_difference_map_explicit(
            modis_csv=modis_csv_path_celsius,
            amsre_csv=amsre_csv_path_37v_celsius,
            modis_col="LST_Celsius_mean",
            amsre_col="temp_C_mean",
            output_path=output_map_modis_37v_celsius,
            title="Différence MODIS - AMSRE (37GHz) - vertical polarization [Celsius]",
            color_label="Temperature differences (K)")
        
        print(f"🖼️ Map of the difference average annual temperatures between MODIS et AMSRE 37GHz (vertical polarization) in Celsius generated at : {output_map_modis_37v_celsius}")
    else : 
        print("\n✅ Map of the difference average annual temperatures between MODIS et AMSRE 37GHz (vertical polarization) in Celsius already generated")


    # = MODIS vs 37GHz (horizontal polarization) =
    if new_graph or not os.path.exists(output_map_modis_37h_celsius): 
        print("\n📊 Generation of the difference average annual temperature map between MODIS et AMSRE 37GHz (horizontal polarization) in Celsius")
        
        plot_difference_map_explicit(
            modis_csv=modis_csv_path_celsius,
            amsre_csv=amsre_csv_path_37h_celsius,
            modis_col="LST_Celsius_mean",
            amsre_col="temp_C_mean",
            output_path=output_map_modis_37h_celsius,
            title="Différence MODIS - AMSRE (37GHz) - horizontal polarization [Celsius]",
            color_label="Temperature differences (K)")
        
        print(f"🖼️ Map of the difference average annual temperatures between MODIS et AMSRE 37GHz (horizontal polarization) in Celsius generated at : {output_map_modis_37h_celsius}")
    else : 
        print("\n✅ Map of the difference average annual temperatures between MODIS et AMSRE 37GHz (horizontal polarization) in Celsius already generated")
    '''


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
            print("\n✅ Land Cover map already generated")
    '''
    

    ### MACHINE LEARNING ###
    
    
    
    # === FOLDER DEFINITIONS ===
    MERGED_FOLDER = "data/processed/machine_learning/dates"
    CLEANED_FILE = "data/processed/machine_learning/cleaned_data.csv"
    INPUT_DIR = "data/processed/amsre"
    INPUT_DIR_VERTICAL = INPUT_DIR + "/vertical_polarization"
    INPUT_DIR_HORIZONTAL = INPUT_DIR + "/horizontal_polarization"
    OUTPUT_DIR = "outputs/machine_learning"
    OUTPUT_DIR_UNNORMALIZED = OUTPUT_DIR + "/unnormalized_inputs"
    OUTPUT_DIR_NORMALIZED = OUTPUT_DIR +"/normalized_inputs"
    PREDICTIONS_DIR = "data/processed/machine_learning/predictions"
    MEAN_TRUE_PATH = "data/processed/modis/mean_temperature/mean_temp_2005_Kelvin.csv"

    # === PIPELINE ===

    # 1. Data preparation
    concat_amsre_files(input_dir_vertical=INPUT_DIR_VERTICAL, input_dir_horizontal=INPUT_DIR_HORIZONTAL, output_file="data/processed/machine_learning/merged_amsre_data.csv")
    merge_daily_datasets()
    print("\n✅ Data merge for ML completed.")
    load_and_merge_data(MERGED_FOLDER, output_file=CLEANED_FILE)
    df = pd.read_csv(CLEANED_FILE)

    # 1.5. Removing pixels with water
    initial_count = len(df)
    df = df[df["land_cover_class"] != 16]
    removed = initial_count - len(df)
    print(f"🧭 Suppressed water pixels : {removed} over {initial_count} ({removed/initial_count:.1%})")
        
    
    generate_heatmap_correlation(df, OUTPUT_DIR if normalized_input else OUTPUT_DIR_UNNORMALIZED)
    
    # 2. Train/Test split
    df['month'] = df['date'].str[:7]
    test_months = ['2005-04', '2005-08', '2005-12']
    train_months = ['2005-01', '2005-02', '2005-03', '2005-05', '2005-06', '2005-07', '2005-09', '2005-10', '2005-11']
    df_train = df[df['month'].isin(train_months)]
    df_test = df[df['month'].isin(test_months)]

    feature_cols = ["brightness_temp_19v", "brightness_temp_37v", "brightness_temp_19h", "brightness_temp_37h", "land_cover_class"]
    target_col = "LST_Kelvin"
    X_train = df_train[feature_cols]
    y_train = df_train[target_col]
    X_test = df_test[feature_cols]
    y_test = df_test[target_col]

    X_test_index = X_test.index

    if normalized_input:
        data_type = "normalized"
        X_train, X_test = normalize_z_score(df_train, df_test, feature_cols)
        output_dir = OUTPUT_DIR_NORMALIZED
        os.makedirs(OUTPUT_DIR_NORMALIZED, exist_ok=True)
    else:
        data_type = "unnormalized"
        output_dir = OUTPUT_DIR_UNNORMALIZED

    # 3. Models
    models = {
        "LinearRegression": train_regression,
        "KNN": train_knn,
        "RandomForest": train_random_forest,
        "GradientBoosting": train_gradient_boosting, 
        "CatBoost": train_catboost, 
        "LightGBM": train_lightgbm, 
        "XGBoost": train_xgboost
    }

    results = []

    # Load ground truth for mean values
    df_mean_true = pd.read_csv(MEAN_TRUE_PATH)
    df_mean_true["true"] = df_mean_true["LST_Kelvin_mean"]

    # 4. Training + Prediction
    for name, train_func in models.items():
        print(f"\n⚙️  Model training: {name}")

        model = train_func(X_train, y_train)
        y_pred, rmse, r2 = evaluate_model(model, X_test, y_test)

        plot_path = os.path.join(output_dir, "prediction_graphs", f"{name}_{data_type}_prediction.png")
        plot_results(y_test, y_pred, plot_path, data_type=data_type)

        plot_error_distributions(y_test, y_pred, output_dir, name, data_type)

        results.append((name, rmse, r2))
        print(f"📈 {name} — RMSE: {rmse:.2f}, R²: {r2:.2f}")

        df_test_plot = df_test.loc[X_test_index, ["lat", "lon", "date"]].copy()
        df_test_plot["prediction"] = y_pred
        df_test_plot["true"] = y_test.values

        pred_folder = os.path.join(PREDICTIONS_DIR, name)
        os.makedirs(pred_folder, exist_ok=True)
        df_test_plot[["lat", "lon", "date", "prediction", "true"]].to_csv(
            os.path.join(pred_folder, f"{name}_{data_type}_predictions.csv"), index=False
        )

        # === Comparative analysis vs MODIS ===
        df_comparison = df_test_plot.merge(df_mean_true.rename(columns={"true": "true_modis"})[["lat", "lon", "true_modis"]], on=["lat", "lon"], how="left")
        df_comparison["diff_pred_vs_modis"] = df_comparison["prediction"] - df_comparison["true_modis"]
        df_comparison["land_cover_class"] = df_test.loc[X_test_index, "land_cover_class"].values

        land_cover_mapping = load_land_cover_lookup(language="fr")

        output_dir_error_hist = os.path.join(output_dir, "error_histogram_vs_modis")
        output_dir_error_landcover = os.path.join(output_dir, "error_by_landcover")    
        output_dir_daily_error = os.path.join(output_dir, "daily_error_trend")

        plot_error_histogram_vs_modis(df_comparison, name, output_dir_error_hist, data_type)
        plot_error_by_landcover(df_comparison, name, output_dir_error_landcover, data_type, land_cover_mapping)
        plot_daily_error_trend(df_comparison, name, output_dir_daily_error, data_type)

        for date_str in df_test_plot["date"].unique():
            print(f"\n📅 Processing date: {date_str}")
            df_day = df_test_plot[df_test_plot["date"] == date_str].copy()

            plot_prediction_map(
                df=df_day,
                y_pred=df_day["prediction"].values,
                model_name=name,
                date=date_str,
                output_dir=os.path.join(output_dir, "prediction_maps"),
                cmap="coolwarm", 
                data_type=data_type
            )

            plot_error_map(
                df_day=df_day,
                model_name=name,
                date_str=date_str,
                output_base_dir=output_dir, 
                data_type=data_type
            )

        plot_mean_map(
            df_test_plot=df_test_plot,
            model_name=name,
            data_type=data_type,
            output_dir=output_dir+"/mean_values_maps"
        )

        plot_mean_error_map(
            df_test_plot=df_test_plot,
            df_mean_true=df_mean_true,
            model_name=name,
            data_type=data_type,
            output_dir=output_dir+"/mean_values_maps"
        )

    print("\n📊 Performance summary:")
    print(f"{'Model':<20} {'RMSE':<10} {'R²':<10}")
    for name, rmse, r2 in results:
        print(f"{name:<20} {rmse:<10.2f} {r2:<10.2f}")
    


    ### Comparison fluxnet vs satellite ###


    '''
    # === FOLDER DEFINITIONS ===

    station_coords_path = "data/processed/fluxnet/fluxnet_station_coordinates.csv"
    fluxnet_csv_path = "data/raw/fluxnet/FluxNET_AMSRE.csv"
    amsre_csv_path = "data/processed/machine_learning/merged_amsre_data.csv"
    output_dir = "outputs/fluxnet_vs_amsre"

    
    # === data processing === 

    df_fluxnet, european_stations = load_european_fluxnet_data(fluxnet_csv_path=fluxnet_csv_path, station_coords_path=station_coords_path)

    # Pair FLUXNET with AMSR-E
    df_matched = match_fluxnet_with_amsre(df_fluxnet=df_fluxnet, european_stations=european_stations, station_coords_path=station_coords_path, amsre_data_path=amsre_csv_path)

    # Cleaning the temp_fluxnet column before the plots
    df_matched['temp_fluxnet'] = df_matched['temp_fluxnet'].apply(clean_temp_fluxnet)

    tb_columns = ["brightness_temp_19v", "brightness_temp_19h", "brightness_temp_37v", "brightness_temp_37h"]

    
    # === plots ===

    batch_plot_all_stations(df_matched, european_stations, tb_columns, output_dir=output_dir)
    '''




if __name__ == "__main__":
    main()

    


