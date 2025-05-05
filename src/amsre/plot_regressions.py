import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import os
from datetime import datetime  

def plot_stationwise_and_global_regressions_2005(csv_path, output_dir="outputs/fluxnet/stationswise_regressions"):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path, sep=";")

    # Wide → long format transformation
    df_long = df.melt(id_vars=["TIMESTAMP_START"], var_name="station", value_name="temperature")

    # Date conversion
    df_long["TIMESTAMP_START"] = pd.to_datetime(df_long["TIMESTAMP_START"], format="%d/%m/%Y")

    # Temperature cleaning
    df_long["temperature"] = pd.to_numeric(df_long["temperature"], errors="coerce")
    df_long = df_long.dropna(subset=["temperature"])
    df_long = df_long[(df_long["temperature"] > 180) & (df_long["temperature"] < 330)]

    all_X = []
    all_y = []

    for station in df_long["station"].unique():
        df_station = df_long[df_long["station"] == station]

        # Creating variables
        X = df_station["TIMESTAMP_START"].map(lambda d: d.toordinal()).values.reshape(-1, 1)
        y = df_station["temperature"].values.reshape(-1, 1)

        if len(X) < 2:
            continue

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # Saving points for global regression
        all_X.append(X)
        all_y.append(y)

        # Individual plot
        plt.figure(figsize=(8, 6))
        plt.scatter(df_station["TIMESTAMP_START"], df_station["temperature"], label=f"{station}", alpha=0.5)
        plt.plot(df_station["TIMESTAMP_START"], y_pred, color="red", label="Régression")
        plt.title(f"T FLUXNET - {station} (2005)")
        plt.xlabel("Date")
        plt.ylabel("Température (K)")
        plt.legend()
        plt.grid(True)

        filename = f"regression_2005_{station}.png".replace("/", "_")  # au cas où
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()

    # Overall regression
    if all_X and all_y:
        X_all = np.vstack(all_X)
        y_all = np.vstack(all_y)

        model_all = LinearRegression()
        model_all.fit(X_all, y_all)

        # Generation of ordered dates and inverse conversion for the plot
        x_range_ord = np.linspace(X_all.min(), X_all.max(), 100).reshape(-1, 1)
        y_range = model_all.predict(x_range_ord)

        # Converting to datetime with fromordinal
        x_range_dates = [datetime.fromordinal(int(val[0])) for val in x_range_ord]
        all_dates = [datetime.fromordinal(int(val[0])) for val in X_all]

        # Global plot
        plt.figure(figsize=(10, 6))
        plt.scatter(all_dates, y_all, alpha=0.3, label="Toutes les stations")
        plt.plot(x_range_dates, y_range, color="black", linewidth=2, label="Régression globale")

        a = model_all.coef_[0][0]
        b = model_all.intercept_[0]
        r2 = model_all.score(X_all, y_all)

        plt.title(f"Global regression FLUXNET 2005\nT = {a:.2f} × date + {b:.2f} (R² = {r2:.2f})")
        plt.xlabel("Date")
        plt.ylabel("Température (K)")
        plt.legend()
        plt.grid(True)

        global_path = os.path.join(output_dir, "regression_globale_2005.png")
        plt.tight_layout()
        plt.savefig(global_path, dpi=300)
        plt.close()
        print(f"✅ Global regression saved : {global_path}")



