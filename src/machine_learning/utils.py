import pandas as pd
from glob import glob
import os
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from tqdm import tqdm

def load_and_merge_data(merged_folder, output_file="data/processed/cleaned_data.csv"):
    if os.path.exists(output_file):
        print(f"⏭️ File already exists, no processing needed: {output_file}")
        return

    all_files = glob(os.path.join(merged_folder, "*.csv"))
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    for f in tqdm(all_files, desc="🔄 Processing files"):
        try:
            df = pd.read_csv(f)
            df = df.dropna()

            # Appliquer les filtres si les colonnes existent
            filters = (df["LST_Kelvin"].between(220, 330))  # filtre LST

            if "brightness_temp_19GHz" in df.columns:
                filters &= df["brightness_temp_19GHz"].between(220, 330)

            if "brightness_temp_37GHz" in df.columns:
                filters &= df["brightness_temp_37GHz"].between(220, 330)

            df = df[filters]

            if df.empty:
                print(f"\n ⚠️ Empty file after filtering: {os.path.basename(f)}")
                continue

            # Ajouter la date depuis le nom du fichier
            basename = os.path.basename(f)
            date_str = os.path.splitext(basename)[0]
            df["date"] = date_str

            # Sauvegarde
            df.to_csv(output_file, mode="a", header=not os.path.exists(output_file), index=False)
            print(f"\n ✅ File processed: {basename}")

        except Exception as e:
            print(f"❌ Error while processing {os.path.basename(f)}: {e}")

    print(f"\n📁 Cleaned data saved to: {output_file}")


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  
    r2 = r2_score(y_test, y_pred)
    return y_pred, rmse, r2
