import os
import json
from src.modis.analysis_utils import analyze_csv_file, analyze_monthly_data, analyze_spatial_data

def analyze_all_files(input_dir="data/processed/modis", output_file="data/analysis/lst_summary_2005.json"):
    summary = {}
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_dir, file_name)
            print(f"📊 {file_name} analyse...")
            stats = analyze_csv_file(file_path)
            if stats:
                summary[file_name] = stats

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
        print(f"✅ Summary saved in {output_file}")

def analyze_all(input_dir="data/processed/modis"):
    # Setp 1 : Global analysis
    print("📊 Launching global analysis...")
    analyze_all_files(input_dir=input_dir, output_file="data/analysis/lst_summary_2005.json")
    print("✅ Global analysis completed.")

    # Step 2 : Monthly analysis
    print("🗓️ Launching Monthly analysis...")
    analyze_monthly_data(input_dir=input_dir, output_file="data/analysis/lst_monthly_summary_2005.json")
    print("✅ Monthly analysis completed.")

    # Step 3 : Spatial analysis
    print("🌍 Launching spatial analysis...")
    analyze_spatial_data(input_dir=input_dir, output_file="data/analysis/lst_spatial_summary_2005.json")
    print("✅ Spatial analysis completed.")


