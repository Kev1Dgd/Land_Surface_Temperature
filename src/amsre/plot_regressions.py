import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import os
from datetime import datetime  


def plot_brightness_vs_temperature_and_regression(csv_path, date_str, freq_label, output_dir="outputs/amsre/dates"):
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
    output_file = os.path.join(date_output_dir, f"regression_tb_vs_temp_{date_str}_{freq_label}.png")
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
                y_pred = model.predict(X)

                a = model.coef_[0][0]
                b = model.intercept_[0]
                r2 = model.score(X, y)

                rmse = rmse = np.sqrt(mean_squared_error(y, y_pred))

                results.append({
                    "date": date_str,
                    "a": a,
                    "b": b,
                    "r2": r2,
                    "rmse": rmse,
                    "n_points": len(df)
                })


    # Save results in a CSV file
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv_path, index=False)
    print(f"✅ Daily regressions saved in : {output_csv_path}")

def get_season_from_month(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Easter"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"


def plot_global_tb_vs_temp(matched_folder, freq_label):
    all_data = []
    output_path=f"outputs/amsre/global_tb_vs_temp_{freq_label}.png"
    for filename in os.listdir(matched_folder):
        if filename.startswith("matched_tb_fluxnet_") and filename.endswith(".csv"):
            # Extraire la date depuis le nom du fichier
            try:
                date_str = filename.replace("matched_tb_fluxnet_", "").replace(".csv", "")
                file_date = datetime.strptime(date_str, "%Y%m%d")
                saison = get_season_from_month(file_date.month)
            except Exception as e:
                print(f"❌ Erreur lors de la lecture de la date dans le fichier {filename} : {e}")
                continue

            df = pd.read_csv(os.path.join(matched_folder, filename))
            df = df.dropna(subset=["brightness_temp_37v", "temperature"])
            df = df[(df["temperature"] > 180) & (df["temperature"] < 330)]
            df = df[(df["brightness_temp_37v"] > 180) & (df["brightness_temp_37v"] < 330)]
            df["saison"] = saison  # Ajout de la saison comme colonne

            all_data.append(df)

    if not all_data:
        print("❗ Aucune donnée trouvée ou utilisable.")
        return

    df_all = pd.concat(all_data, ignore_index=True)

    # Régression globale
    X = df_all["brightness_temp_37v"].values.reshape(-1, 1)
    y = df_all["temperature"].values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Couleurs saisonnières
    season_colors = {
        "Winter": "blue",
        "Easter": "green",
        "Summer": "orange",
        "Autumn": "brown"
    }

    # Tracé
    plt.figure(figsize=(10, 6))
    for saison, group in df_all.groupby("saison"):
        plt.scatter(
            group["brightness_temp_37v"],
            group["temperature"],
            s=10,
            alpha=0.4,
            color=season_colors[saison],
            label=saison
        )

    plt.plot(X, y_pred, color='red', linewidth=2, label="Régression linéaire")
    plt.xlabel("Température de brillance AMSR-E (K)")
    plt.ylabel("Température FLUXNET (K)")
    r2 = model.score(X, y)
    a = model.coef_[0][0]
    b = model.intercept_[0]
    plt.title(f"Régression globale 2005 for the {freq_label}GHz frequency : T = {a:.2f} × TB + {b:.2f} (R² = {r2:.2f})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ Graphe global enregistré : {output_path}")


def plot_stationwise_and_global_regressions_2005(csv_path, freq_label, output_dir="outputs/fluxnet/stationswise_regressions"):
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
        plt.scatter(all_dates, y_all, alpha=0.3, label="All the stations")
        plt.plot(x_range_dates, y_range, color="black", linewidth=2, label="Global Regression")

        a = model_all.coef_[0][0]
        b = model_all.intercept_[0]
        r2 = model_all.score(X_all, y_all)

        plt.title(f"Global regression FLUXNET 2005\nT = {a:.2f} × date + {b:.2f} (R² = {r2:.2f})")
        plt.xlabel("Date")
        plt.ylabel("Température (K)")
        plt.legend()
        plt.grid(True)

        global_path = os.path.join(output_dir, f"regression_globale_2005_{freq_label}.png")
        plt.tight_layout()
        plt.savefig(global_path, dpi=300)
        plt.close()
        print(f"✅ Global regression saved : {global_path}")



def plot_station_regressions(df_matched1, df_matched2, output_dir="outputs/amsre/stations"):
    os.makedirs(output_dir, exist_ok=True)

    # Intersections des stations
    common_stations = set(df_matched1["station"]).intersection(df_matched2["station"])

    for station in common_stations:
        group1 = df_matched1[df_matched1["station"] == station]
        group2 = df_matched2[df_matched2["station"] == station]

        # Vérification données valides
        x1, y1 = group1["brightness_temp_37v"], group1["temperature"]
        x2, y2 = group2["brightness_temp_37v"], group2["temperature"]

        valid1 = np.isfinite(x1) & np.isfinite(y1)
        valid2 = np.isfinite(x2) & np.isfinite(y2)

        if valid1.sum() < 2 and valid2.sum() < 2:
            print(f"⏭️ Pas assez de données valides pour {station}, skip.")
            continue

        plt.figure(figsize=(8, 6))

        if valid1.sum() >= 2:
            coef1 = np.polyfit(x1[valid1], y1[valid1], deg=1)
            poly1 = np.poly1d(coef1)
            plt.scatter(x1, y1, alpha=0.4, label="Obs 1", color="blue")
            plt.plot(x1, poly1(x1), color="blue", label=f"Régression 1: y = {coef1[0]:.2f}x + {coef1[1]:.2f}")

        if valid2.sum() >= 2:
            coef2 = np.polyfit(x2[valid2], y2[valid2], deg=1)
            poly2 = np.poly1d(coef2)
            plt.scatter(x2, y2, alpha=0.4, label="Obs 2", color="red")
            plt.plot(x2, poly2(x2), color="green", label=f"Régression 2: y = {coef2[0]:.2f}x + {coef2[1]:.2f}")

        plt.xlabel("Température de brillance (tb)")
        plt.ylabel("Température mesurée")
        plt.title(f"Régressions température vs tb - Station {station}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        output_path = os.path.join(output_dir, f"regression_tb_vs_temp_{station}.png")
        plt.savefig(output_path)
        plt.close()



def plot_regression_metrics_evolution(regression_csv_path,freq_label):
    output_path=f"outputs/amsre/regression_metrics_evolution_{freq_label}.png"
    df = pd.read_csv(regression_csv_path)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

    plt.figure(figsize=(12, 6))

    # On trace les trois courbes sur le même graphe
    plt.plot(df["date"], df["a"], label="Pente (a)", color="blue")
    plt.plot(df["date"], df["r2"], label="R²", color="green")
    plt.plot(df["date"], df["rmse"], label="RMSE", color="red")

    plt.title(f"Évolution temporelle des métriques de régression {freq_label}")
    plt.xlabel("Date")
    plt.ylabel("Valeurs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Graphe des métriques sauvegardé : {output_path}")

