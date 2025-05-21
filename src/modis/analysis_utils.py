import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta

def extract_date_from_filename(filename):
    # Retrieve the date part of the file name
    date_str = filename.split('.')[1][1:]  
    year = int(date_str[:4])  
    day_of_year = int(date_str[4:])  # Day Of the Year 
    
    # Convert this to a real date
    start_date = datetime(year, 1, 1)  
    date = start_date + timedelta(days=day_of_year - 1)  # Add days to January 1st
    
    return date

def load_and_process_csv(filename):
    # Load CSV without headers if necessary
    df = pd.read_csv(filename)
    
    # Extract date from file name
    date = extract_date_from_filename(filename)
    
    # Add a new 'Date' column with the extracted date
    df['Date'] = date
    
    return df


def compute_basic_stats(df):
    stats = {
        "mean": np.nanmean(df.values),
        "min": np.nanmin(df.values),
        "max": np.nanmax(df.values),
        "std": np.nanstd(df.values),
    }
    return stats

def analyze_csv_file(file_path):
    try:
        df = pd.read_csv(file_path)
        return compute_basic_stats(df)
    except Exception as e:
        print(f"⚠️ Reading error for {file_path} : {e}")
        return None
    

def compute_monthly_mean(df):
    df['Date'] = pd.to_datetime(df['Date'])  
    df.set_index('Date', inplace=True)
    monthly_mean = df.resample('M').mean()  # Mensual mean
    return monthly_mean


def analyze_monthly_data(input_dir, output_file):
    monthly_summary = {}
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            print(f"📊 {filename} mensual analysis...")
            
            try:
                # Load file and add 'Date' column
                df = load_and_process_csv(os.path.join(input_dir, filename))
                
                df['Month'] = df['Date'].dt.month  # Extract month from 'Date' column
                
                # Example of a simple analysis: average temperature by month
                monthly_avg = df.groupby('Month')['LST_Day_1km'].mean()
                
                # Store monthly results
                monthly_summary[filename] = monthly_avg.to_dict()
            
            except Exception as e:
                print(f"⚠️ Reading error for {filename} : {e}")
    
    # Save results in a JSON file
    with open(output_file, 'w') as f:
        json.dump(monthly_summary, f, indent=2)



def compute_spatial_mean(df):
    spatial_mean = df.mean().mean()  
    return spatial_mean

def analyze_spatial_data(input_dir="data/processed/modis", output_file="data/analysis/lst_spatial_summary_2005.json"):
    spatial_summary = {}

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_dir, file_name)
            print(f"🌍 {file_name} spatial analysis...")

            try:
                df = pd.read_csv(file_path)
                spatial_mean = compute_spatial_mean(df)
                spatial_summary[file_name] = spatial_mean
            except Exception as e:
                print(f"⚠️ Reading error for {file_path} : {e}")

    with open(output_file, "w") as f:
        json.dump(spatial_summary, f, indent=2)
        print(f"✅ Spatial summary saved in {output_file}")
