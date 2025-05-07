import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta

def extract_date_from_filename(filename):
    """
    Extrait la date du nom de fichier au format 'AYYYYDDD'.
    Par exemple 'A2005001' donne le 1er janvier 2005.
    """
    # Récupérer la partie de la date dans le nom du fichier (par exemple 'A2005001')
    date_str = filename.split('.')[1][1:]  # Extrait '2005001' de 'A2005001'
    year = int(date_str[:4])  # Année (ex: 2005)
    day_of_year = int(date_str[4:])  # Jour de l'année (ex: 1 pour le 1er janvier)
    
    # Convertir cela en une date réelle
    start_date = datetime(year, 1, 1)  # 1er janvier de l'année
    date = start_date + timedelta(days=day_of_year - 1)  # Ajouter les jours au 1er janvier
    
    return date

def load_and_process_csv(filename):
    """
    Charge un fichier CSV, extrait la date du nom et ajoute-la au DataFrame.
    """
    # Charger le CSV sans les en-têtes si nécessaire
    df = pd.read_csv(filename)
    
    # Extraire la date à partir du nom du fichier
    date = extract_date_from_filename(filename)
    
    # Ajouter une nouvelle colonne 'Date' avec la date extraite
    df['Date'] = date
    
    return df


def compute_basic_stats(df):
    """Calcule des stats descriptives sur les données LST."""
    stats = {
        "mean": np.nanmean(df.values),
        "min": np.nanmin(df.values),
        "max": np.nanmax(df.values),
        "std": np.nanstd(df.values),
    }
    return stats

def analyze_csv_file(file_path):
    """Charge un CSV et calcule les statistiques de base."""
    try:
        df = pd.read_csv(file_path)
        return compute_basic_stats(df)
    except Exception as e:
        print(f"⚠️ Erreur de lecture {file_path} : {e}")
        return None
    

def compute_monthly_mean(df):
    """Calcule la moyenne mensuelle des valeurs LST."""
    df['Date'] = pd.to_datetime(df['Date'])  # Assurez-vous qu'une colonne 'Date' existe dans vos CSV
    df.set_index('Date', inplace=True)
    monthly_mean = df.resample('M').mean()  # Moyenne mensuelle
    return monthly_mean


def analyze_monthly_data(input_dir, output_file):
    monthly_summary = {}
    
    # Liste des fichiers CSV dans le répertoire
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            print(f"📊 Analyse mensuelle de {filename}...")
            
            try:
                # Charger le fichier et ajouter la colonne 'Date'
                df = load_and_process_csv(os.path.join(input_dir, filename))
                
                # Assurez-vous que les colonnes nécessaires existent
                # Par exemple, vous pouvez filtrer ou regrouper par mois
                df['Month'] = df['Date'].dt.month  # Extraire le mois à partir de la colonne 'Date'
                
                # Exemple d'analyse simple : moyenne de la température par mois
                monthly_avg = df.groupby('Month')['LST_Day_1km'].mean()
                
                # Stocker les résultats mensuels
                monthly_summary[filename] = monthly_avg.to_dict()
            
            except Exception as e:
                print(f"⚠️ Erreur de lecture {filename} : {e}")
    
    # Sauvegarder les résultats dans un fichier JSON
    with open(output_file, 'w') as f:
        json.dump(monthly_summary, f, indent=2)



def compute_spatial_mean(df):
    """Calcule la moyenne des pixels pour chaque fichier LST."""
    spatial_mean = df.mean().mean()  # Moyenne de toutes les valeurs de la grille
    return spatial_mean

def analyze_spatial_data(input_dir="data/processed/modis", output_file="data/analysis/lst_spatial_summary_2005.json"):
    spatial_summary = {}

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_dir, file_name)
            print(f"🌍 Analyse spatiale de {file_name}...")

            try:
                df = pd.read_csv(file_path)
                spatial_mean = compute_spatial_mean(df)
                spatial_summary[file_name] = spatial_mean
            except Exception as e:
                print(f"⚠️ Erreur de lecture {file_path} : {e}")

    with open(output_file, "w") as f:
        json.dump(spatial_summary, f, indent=2)
        print(f"✅ Résumé spatial sauvegardé dans {output_file}")
