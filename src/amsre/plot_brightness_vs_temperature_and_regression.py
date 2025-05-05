import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import os

def plot_brightness_vs_temperature_and_regression(csv_path, date_str, output_dir="outputs/amsre"):
    # Load data
    df = pd.read_csv(csv_path)

    # Data cleansing: removal of missing and inconsistent values
    df = df.dropna(subset=["brightness_temp_37v", "temperature"])
    df = df[(df["temperature"] > 180) & (df["temperature"] < 330)]
    df = df[(df["brightness_temp_37v"] > 180) & (df["brightness_temp_37v"] < 330)]

    # Variables for regression
    X = df["brightness_temp_37v"].values.reshape(-1, 1)
    y = df["temperature"].values.reshape(-1, 1)

    # Linear regression model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Creating the plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color="blue", label="Stations FLUXNET")
    plt.plot(X, y_pred, color="red", linewidth=2, label="Régression linéaire")

    # Regression coefficients
    a = model.coef_[0][0]
    b = model.intercept_[0]
    r2 = model.score(X, y)
    plt.title(f"Temperature FLUXNET vs TB AMSR-E (37 GHz)\nRegression : T = {a:.2f} × TB + {b:.2f} (R² = {r2:.2f})")

    plt.xlabel("Brightness temperature AMSR-E (K)")
    plt.ylabel("FLUXNET temperature (K)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Create a folder for this date if necessary
    date_output_dir = os.path.join(output_dir, date_str)
    os.makedirs(date_output_dir, exist_ok=True)

    # Save the regression image
    output_file = os.path.join(date_output_dir, f"regression_tb_vs_temp_{date_str}.png")
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"✅ Regression saved in {output_file}")


def fit_daily_regressions(folder_path, output_csv_path):
    results = []

    for root, _, files in os.walk(folder_path):
        for filename in sorted(files):
            if filename.startswith("matched_tb_fluxnet_") and filename.endswith(".csv"):
                date_str = filename.replace("matched_tb_fluxnet_", "").replace(".csv", "")
                file_path = os.path.join(root, filename)
                df = pd.read_csv(file_path)

                # Cleaning
                df = df.dropna(subset=["brightness_temp_37v", "temperature"])
                df = df[(df["temperature"] > 180) & (df["temperature"] < 330)]
                df = df[(df["brightness_temp_37v"] > 180) & (df["brightness_temp_37v"] < 330)]

                if len(df) < 3:  # If less than 3 points, ignore this day
                    print(f"⚠️ Pas assez de données pour {date_str}")
                    continue

                # Linear regression
                X = df["brightness_temp_37v"].values.reshape(-1, 1)
                y = df["temperature"].values.reshape(-1, 1)

                model = LinearRegression()
                model.fit(X, y)

                a = model.coef_[0][0]
                b = model.intercept_[0]
                r2 = model.score(X, y)

                results.append({
                    "date": date_str,
                    "a": a,
                    "b": b,
                    "r2": r2,
                    "n_points": len(df)
                })

    # Save results in a CSV file
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv_path, index=False)
    print(f"✅ Daily regressions saved in : {output_csv_path}")
