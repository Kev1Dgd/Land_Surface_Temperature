import os
import json
from src.modis.analysis_utils import analyze_csv_file, analyze_monthly_data, analyze_spatial_data

def analyze_all_files(input_dir="data/processed/modis", output_file="data/analysis/lst_summary_2005.json"):
    summary = {}
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_dir, file_name)
            print(f"📊 Analyse de {file_name}...")
            stats = analyze_csv_file(file_path)
            if stats:
                summary[file_name] = stats

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
        print(f"✅ Résumé sauvegardé dans {output_file}")

def analyze_all(input_dir="data/processed/modis"):
    # Étape 1: Analyse globale
    print("📊 Lancement de l'analyse globale des fichiers...")
    analyze_all_files(input_dir=input_dir, output_file="data/analysis/lst_summary_2005.json")
    print("✅ Analyse globale terminée.")

    # Étape 2: Analyse mensuelle
    print("🗓️ Lancement de l'analyse mensuelle...")
    analyze_monthly_data(input_dir=input_dir, output_file="data/analysis/lst_monthly_summary_2005.json")
    print("✅ Analyse mensuelle terminée.")

    # Étape 3: Analyse spatiale
    print("🌍 Lancement de l'analyse spatiale...")
    analyze_spatial_data(input_dir=input_dir, output_file="data/analysis/lst_spatial_summary_2005.json")
    print("✅ Analyse spatiale terminée.")

if __name__ == "__main__":
    analyze_all()

