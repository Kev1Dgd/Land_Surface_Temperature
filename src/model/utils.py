import pandas as pd
import glob
import os
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

def load_and_merge_data(merged_folder, output_file="data/processed/cleaned_data.csv", chunksize=100_000):
    all_files = glob.glob(os.path.join(merged_folder, "*.csv"))
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if os.path.exists(output_file):
        os.remove(output_file)

    for f in tqdm(all_files, desc="🔄 Traitement des fichiers"):
        try:
            chunk_iter = pd.read_csv(f, chunksize=chunksize)
            for i, chunk in enumerate(chunk_iter):
                chunk = chunk.dropna()
                chunk = chunk[
                    (chunk["LST_Kelvin"] >= 220) & (chunk["LST_Kelvin"] <= 330) &
                    (chunk["brightness_temp_37v"] >= 220) & (chunk["brightness_temp_37v"] <= 330) &
                    (chunk["brightness_temp_19v"] >= 220) & (chunk["brightness_temp_19v"] <= 330)
                ]
                if not chunk.empty:
                    chunk.to_csv(output_file, mode="a", header=not os.path.exists(output_file), index=False)
            print(f"✅ File processed : {os.path.basename(f)}")
        except Exception as e:
            print(f"❌ Erreur lors du traitement de {os.path.basename(f)} : {e}")

    print(f"\n📁 Cleaned data written in : {output_file}")


def clean_data(df):
    df = df.dropna()
    df = df[(df["LST_Kelvin"] >= 220) & (df["LST_Kelvin"] <= 330)]
    df = df[(df["brightness_temp_37v"] >= 220) & (df["brightness_temp_37v"] <= 330)]
    df = df[(df["brightness_temp_19v"] >= 220) & (df["brightness_temp_19v"] <= 330)]
    print(f"✅ Cleaned data: {len(df)} remaining points.")
    return df

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    return y_pred, rmse, r2


def plot_results(y_test, y_pred, output_path):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.4)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual LST (°C)")
    plt.ylabel("Predicted LST (°C)")
    plt.title("MODIS surface temperature prediction")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Registered regression graph in : {output_path}")