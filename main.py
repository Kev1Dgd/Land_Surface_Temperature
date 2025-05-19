### Library and function imports ###

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import xarray as xr

from src.modis.download import authenticate, search_modis_lst, download_results
from src.modis.process import process_nc_to_csv_light
from src.modis.plot import process_all_modis_csv
from src.modis.analyze import analyze_all_files 
from src.modis.analysis_utils import analyze_monthly_data, analyze_spatial_data
from src.modis.utils import check_and_create_file

from src.amsre.download import download_amsre_ae_l2a
from src.amsre.plot import plot_bt_map, plot_temp_estimated_map
from src.amsre.process import combine_amsre_files_37ghz, combine_amsre_files_19ghz, concat_amsre_files
from src.amsre.fix_headers import fix_amsre_headers

from src.amsre.matches import generate_daily_matches
from src.amsre.plot_regressions import plot_stationwise_and_global_regressions_2005, plot_global_tb_vs_temp, plot_brightness_vs_temperature_and_regression, fit_daily_regressions, plot_station_regressions, plot_regression_metrics_evolution
from src.amsre.plot_temp_evolution import plot_seasonal_temp_evolution, plot_seasonal_temp_with_tb_evolution, plot_all_stations_temp_evolution

from src.merge.create_dataset import merge_daily_datasets

from src.land_cover.process import convert_land_cover_nc_to_csv
from src.land_cover.plot import plot_land_cover_map


def main():

    ### SETTINGS ### 

    n_point = 5000          # Number of sampling points (saves time) 
    Sampling = False        # If you want to sample your maps
    new_graph = False       # If the maps are to be generated again if they already exist
    '''
    ### MODIS PART ###

    print("\n===== MODIS stage: Land Surface Temperature =====")

    # Processing MODIS files
    nc_path = "data/raw/modis/MOD11A1.061_1km_aid0001.nc"  
    for day in range(0, 364): 
        process_nc_to_csv_light(nc_path, f"data/processed/modis/modis_lst_{day}.csv", day_index=day)
    process_all_modis_csv()

    # Analysis
    print("📊 Starting files analysis...") 
    check_and_create_file("data/analysis/modis/lst_summary_2005.json", analyze_all_files, input_dir="data/processed/modis")
    check_and_create_file("data/analysis/modis/lst_monthly_summary_2005.json", analyze_monthly_data, input_dir="data/processed/modis")
    check_and_create_file("data/analysis/modis/lst_spatial_summary_2005.json",analyze_spatial_data,input_dir="data/processed/modis")   
    print("📈 Analysis completed.")
    '''

    ### AMSRE PART ###

    '''start_date = datetime(2005, 1, 1)
    end_date = datetime(2005, 12, 31)
    dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end_date - start_date).days + 1)]
    authenticate()
    
    print("\n===== AMSR-E stage: TB 37GHz processing and plotting map =====")
    
    for date in dates :
        # Filter only .hdf files
        #files = [f for f in download_amsre_ae_l2a(date=date) if f.endswith('.hdf')]
        files = [f for f in os.listdir("data/raw/amsre") if f.endswith(".hdf")]

        date_str = date.replace('-', '')

        # Combine the files into two separate CSVs and retrieve the output paths
        output_ascending_37, output_descending_37 = combine_amsre_files_37ghz(files, date=date)
        output_ascending_19, output_descending_19 = combine_amsre_files_19ghz(files, date=date)
        
        # Maps Generation - 37GHz
        if output_ascending_37 and output_descending_37:
            print(f"\n===== AMSR-E Map Generation : TB_37GHz. Date : {date} =====")
            # Loading renamed files
            df_ascending_37 = pd.read_csv(output_ascending_37)
            df_descending_37 = pd.read_csv(output_descending_37)

            # 5000-point sampling
            if Sampling :
                df_ascending_sampled_37 = df_ascending_37.sample(n=n_point) if len(df_ascending_37) > 5000 else df_ascending_37
                df_descending_sampled_37 = df_descending_37.sample(n=n_point) if len(df_descending_37) > 5000 else df_descending_37

                asc_plot_path_tb_37 = f"outputs/amsre/dates/{date}/tb_37ghz_map_{date}_ascending.png"
                des_plot_path_tb_37 = f"outputs/amsre/dates/{date}/tb_37ghz_map_{date}_descending.png"
                comb_plot_path_tb_37 = f"outputs/amsre/dates/{date}/tb_37ghz_map_{date}.png"

                if new_graph or not os.path.exists(asc_plot_path_tb_37):
                    print("\n📈 Sampled visualisation of Ascending")
                    plot_bt_map(df_ascending_sampled_37, date, pass_type="ascending",freq_label="37ghz")
                else : 
                    print("\n✅ [37GHz] - Ascending TB map already generated")

                if new_graph or not os.path.exists(des_plot_path_tb_37):
                    print("\n📉 Sampled visualisation of Descending")
                    plot_bt_map(df_descending_sampled_37, date, pass_type="descending",freq_label="37ghz")
                else : 
                    print("\n✅ [37GHz] - Descending map already generated")

                if new_graph or not os.path.exists(comb_plot_path_tb_37):
                    print("\n📊 Sampled visualisation of Combined datas")
                    plot_bt_map(pd.concat([df_ascending_sampled_37, df_descending_sampled_37]), date, pass_type="combined",freq_label="37ghz")
                else : 
                    print("\n✅ [37GHz] - Combined map already generated\n")
            
            else :  

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

            # 5000-point sampling
            if Sampling :
                df_ascending_sampled_19 = df_ascending_19.sample(n=n_point) if len(df_ascending_19) > 5000 else df_ascending_19
                df_descending_sampled_19 = df_descending_19.sample(n=n_point) if len(df_descending_19) > 5000 else df_descending_19

                asc_plot_path_tb_19 = f"outputs/amsre/dates/{date}/tb_19ghz_map_{date}_ascending.png"
                des_plot_path_tb_19 = f"outputs/amsre/dates/{date}/tb_19ghz_map_{date}_descending.png"
                comb_plot_path_tb_19 = f"outputs/amsre/dates/{date}/tb_19ghz_map_{date}.png"

                if new_graph or not os.path.exists(asc_plot_path_tb_19):
                    print("\n📈 Sampled visualisation of Ascending")
                    plot_bt_map(df_ascending_sampled_19, date, pass_type="ascending",freq_label="19ghz")
                else : 
                    print("\n✅ [19GHz] - Ascending TB map already generated")

                if new_graph or not os.path.exists(des_plot_path_tb_19):
                    print("\n📉 Sampled visualisation of Descending")
                    plot_bt_map(df_descending_sampled_19, date, pass_type="descending",freq_label="19ghz")
                else : 
                    print("\n✅ [19GHz] - Descending map already generated")

                if new_graph or not os.path.exists(comb_plot_path_tb_19):
                    print("\n📊 Sampled visualisation of Combined datas")
                    plot_bt_map(pd.concat([df_ascending_sampled_19, df_descending_sampled_19]), date, pass_type="combined",freq_label="19ghz")
                else : 
                    print("\n✅ [19GHz] - Combined map already generated\n")
            
            else :  

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
    

    print("\n===== END of AMSR-E TB and Temperature by Regression =====")'''

    '''
    ### FLUXNET & PLOTS PART ###
    
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
    plot_global_tb_vs_temp("data/processed/amsre/matched/37GHz", "37GHz")


    print("\n===== Daily regression TB vs Temperature (multi-day) for the 19GHz frequency =====")
    output_regression_csv_19 = "data/analysis/amsre/daily_regressions_19GHz.csv"
    fit_daily_regressions(matched_output_folder_19, output_regression_csv_19, "19GHz")
    plot_regression_metrics_evolution(output_regression_csv_19,"19GHz")
    plot_global_tb_vs_temp("data/processed/amsre/matched/19GHz", "19GHz")


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

    '''# Obtaining regression metrics
    reg_df_37 = pd.read_csv("data/analysis/amsre/daily_regressions_37GHz.csv")
    reg_df_19 = pd.read_csv("data/analysis/amsre/daily_regressions_19GHz.csv")

    # Plotting supposed temperature maps - AMSRE
    for date in dates :    
        date_str = date.replace('-', '')
        date_int = int(date_str)

        row_37 = reg_df_37[reg_df_37['date'] == date_int]  
        row_19 = reg_df_19[reg_df_19['date'] == date_int] 

        if not row_37.empty:   
            a37, b37 = row_37["a"].iloc[0], row_37["b"].iloc[0]                    
        else:
            print(f"37GHz - Aucune régression disponible pour {date_str}, skip.")
            continue
            
        if not row_19.empty:   
            a19, b19 = row_19["a"].iloc[0], row_19["b"].iloc[0]  
        else:
            print(f"19GHz - Aucune régression disponible pour {date_str}, skip.")
            continue
        
        output_ascending_37, output_descending_37 = combine_amsre_files_37ghz(files, date=date)
        output_ascending_19, output_descending_19 = combine_amsre_files_19ghz(files, date=date)

        df_ascending_37 = pd.read_csv(output_ascending_37)
        df_descending_37 = pd.read_csv(output_descending_37)

        df_ascending_19 = pd.read_csv(output_ascending_19)
        df_descending_19 = pd.read_csv(output_descending_19)
        
        ### Temperatures generated from linear regression - 37 GHz
                
        asc_plot_path_regtemp_37 = f"outputs/amsre/dates/{date}/temp_by_reg_37ghz_map_{date}_ascending.png"
        des_plot_path_regtemp_37 = f"outputs/amsre/dates/{date}/temp_by_reg_37ghz_map_{date}_descending.png"
        comb_plot_path_regtemp_37 = f"outputs/amsre/dates/{date}/temp_by_reg_37ghz_map_{date}_combined.png"

        print("\n📊 37GHz Supposed Temperature Maps for date : {date}")
        
        if new_graph or not os.path.exists(asc_plot_path_regtemp_37):
            print("\n📈 Visualisation of Ascending Supposed Temperature")
            plot_temp_estimated_map(df_ascending_37, date, pass_type="ascending", freq_label="37ghz", a=a37, b=b37)
        else : 
            print("\n✅ [37GHz] - Ascending supposed temperatures map already generated")

        if new_graph or not os.path.exists(des_plot_path_regtemp_37):    
            print("\n📉 Visualisation of Descending Supposed Temperature")
            plot_temp_estimated_map(df_descending_37, date, pass_type="descending", freq_label="37ghz", a=a37, b=b37)
        else : 
            print("\n✅ [37GHz] - Descending supposed temperatures map already generated")

        if new_graph or not os.path.exists(comb_plot_path_regtemp_37):   
            print("\n📊 Visualisation of Combined Supposed Temperature datas") 
            plot_temp_estimated_map(pd.concat([df_ascending_37, df_descending_37]), date, pass_type="combined", freq_label="37ghz", a=a37, b=b37)
        else : 
            print("\n✅ [37GHz] - Combined supposed temperatures map already generated")       

        ### Temperatures generated from linear regression - 19 GHz

        asc_plot_path_regtemp_19 = f"outputs/amsre/dates/{date}/temp_by_reg_19ghz_map_{date}_ascending.png"
        des_plot_path_regtemp_19 = f"outputs/amsre/dates/{date}/temp_by_reg_19ghz_map_{date}_descending.png"
        comb_plot_path_regtemp_19 = f"outputs/amsre/dates/{date}/temp_by_reg_19ghz_map_{date}_combined.png"

        print("\n📊 19GHz Supposed Temperature Maps for date : {date}")

        if new_graph or not os.path.exists(asc_plot_path_regtemp_19):
            print("\n📈 Sampled Visualisation of Ascending Supposed Temperature")
            plot_temp_estimated_map(df_ascending_19, date, pass_type="ascending", freq_label="19ghz", a=a19, b=b19)
        else : 
            print("\n✅ [19GHz] - Ascending supposed temperatures map already generated")

        if new_graph or not os.path.exists(des_plot_path_regtemp_19):    
            print("\n📉 [19GHz] - Sampled visualisation of Descending Supposed Temperature")
            plot_temp_estimated_map(df_descending_19, date, pass_type="descending", freq_label="19ghz", a=a19, b=b19)
        else : 
            print("\n✅ [19GHz] - Descending supposed temperatures map already generated")

        if new_graph or not os.path.exists(comb_plot_path_regtemp_19):   
            print("\n📊 Sampled visualisation of Combined Supposed Temperature datas") 
            plot_temp_estimated_map(pd.concat([df_ascending_19, df_descending_19]), date, pass_type="combined", freq_label="19ghz", a=a19, b=b19)
        else : 
            print("\n✅ Combined supposed temperatures map already generated")'''

    ### LAND COVER PART ###
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
    

    ### MACHINE LEARNING ###

    print("\n===== Merge all AMSRE CSVs into one =====")
    concat_amsre_files(input_dir="data/processed/amsre/",output_file="data/processed/amsre/merged_amsre_data.csv")

    print("\n===== Merge AMSRE, MODIS and Land Cover data for ML =====")
    merge_daily_datasets()

    print("\n✅ Data merge for ML completed.")


if __name__ == "__main__":
    main()