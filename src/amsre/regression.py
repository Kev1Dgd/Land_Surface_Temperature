import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import os

def fit_daily_regressions(folder_path, output_csv_path, freq_label):
    results = []

    # Charger les résultats existants s'ils existent déjà
    if os.path.exists(output_csv_path):
        df_existing = pd.read_csv(output_csv_path)
        existing_dates = set(df_existing["date"].astype(str))
        print(f"🔁 Existing regression file found. {len(existing_dates)} dates already processed.")
    else:
        df_existing = pd.DataFrame()
        existing_dates = set()

    for root, _, files in os.walk(folder_path):
        for filename in sorted(files):
            if filename.startswith("matched_tb_fluxnet_") and filename.endswith(".csv"):
                date_str = filename.replace("matched_tb_fluxnet_", "").replace(".csv", "")
                
                if date_str in existing_dates:
                    print(f"⏭️ Skipping already processed date: {date_str}")
                    continue

                file_path = os.path.join(root, filename)
                df = pd.read_csv(file_path)

                # Nettoyage
                col_tb = f"brightness_temp_{freq_label[:2]}v"
                df = df.dropna(subset=[col_tb, "temperature"])
                df = df[
                    (df["temperature"].between(180, 330)) & 
                    (df[col_tb].between(180, 330))
                ]

                if len(df) < 2:
                    print(f"⚠️ Not enough data for {date_str}")
                    continue

                # Conversion mémoire
                df["temperature"] = df["temperature"].astype(np.float32)
                df[col_tb] = df[col_tb].astype(np.float32)

                # Régression
                X = df[col_tb].values.reshape(-1, 1)
                y = df["temperature"].values.reshape(-1, 1)

                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)

                a = model.coef_[0][0]
                b = model.intercept_[0]
                r2 = model.score(X, y)
                rmse = np.sqrt(mean_squared_error(y, y_pred))

                results.append({
                    "date": date_str,
                    "a": a,
                    "b": b,
                    "r2": r2,
                    "rmse": rmse,
                    "n_points": len(df)
                })

                # Libération mémoire pour éviter surcharge
                del df, X, y, y_pred, model

    # Fusion avec résultats existants si besoin
    if not df_existing.empty:
        df_results = pd.concat([df_existing, pd.DataFrame(results)], ignore_index=True)
    else:
        df_results = pd.DataFrame(results)

    df_results.to_csv(output_csv_path, index=False)
    print(f"\n✅ Daily regressions saved in : {output_csv_path}")