import os
import pandas as pd

def create_fluxnet_site_list(availability_df, year=2005):
    """Génère un fichier CSV des sites Fluxnet disponibles pour une année donnée avec une colonne 'status'."""
    available_sites = availability_df[(availability_df["year"] == year) & (availability_df["available"])].copy()
    
    # Ajouter une colonne 'status' pour indiquer si le site doit être téléchargé ou a déjà été téléchargé
    available_sites["status"] = available_sites["site_id"].apply(lambda site: "downloaded" if os.path.exists(f"data/processed/fluxnet/{site}_{year}.csv") else "to download")
    
    # Sauvegarder les données sous forme de CSV
    try:
        file_path = f"outputs/fluxnet/sites_{year}.csv"
        # Ajouter une vérification pour s'assurer que les données sont correctes
        if available_sites.empty:
            print(f"❌ Aucune donnée disponible pour l'année {year}.")
        else:
            available_sites.to_csv(file_path, index=False)
            print(f"✅ Liste des sites {year} sauvegardée dans {file_path}")
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde du fichier {file_path}: {e}")


def update_status(availability_df, year=2005):
    """Met à jour le statut des sites Fluxnet en fonction de la présence des fichiers téléchargés."""
    
    # Vérifier si les fichiers existent pour chaque site et mettre à jour le statut
    availability_df["status"] = availability_df.apply(
        lambda row: "downloaded" if os.path.exists(f"data/processed/fluxnet/{row['site_id']}_{year}.csv") else "to download",
        axis=1
    )
    
    # Sauvegarder les mises à jour dans un fichier CSV
    try:
        file_path = f"outputs/fluxnet/sites_{year}.csv"
        availability_df.to_csv(file_path, index=False)
        print(f"✅ Liste des sites mise à jour avec le statut des téléchargements sauvegardée dans {file_path}")
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde du fichier {file_path}: {e}")

