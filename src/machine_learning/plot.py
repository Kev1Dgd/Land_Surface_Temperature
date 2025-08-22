import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import seaborn as sns
import pandas as pd
import numpy as np



def plot_results(y_test, y_pred, output_path, data_type):

    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    mask = y_pred > 230
    y_test_filtered = y_test[mask]
    y_pred_filtered = y_pred[mask]

    # Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test_filtered, y=y_pred_filtered, alpha=0.4)
    plt.plot([y_test_filtered.min(), y_test_filtered.max()],[y_test_filtered.min(), y_test_filtered.max()], 'r--')
    plt.xlabel("Actual LST (°K)")
    plt.ylabel(f"Predicted LST (°K) with {data_type} data")
    plt.title("MODIS Surface Temperature Prediction")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=600)
    plt.close()
    print(f"✅ Regression plot saved at: {output_path}")


def plot_prediction_map(df, y_pred, model_name, date, output_dir, 
                        cmap, data_type, vmin=None, vmax=None, binning=False, create_date_folder=True):
    
    print(f"🗺️ Generating map for model: {model_name}")

    df_plot = df.copy()
    df_plot["prediction"] = y_pred

    if binning:
        # Round coordinates for spatial binning
        df_plot["lat_bin"] = df_plot["lat"].round(2)
        df_plot["lon_bin"] = df_plot["lon"].round(2)
        print(f"[DEBUG] Number of points before groupby: {len(df_plot)}")
        df_grouped = df_plot.groupby(["lat_bin", "lon_bin"]).agg({"prediction": "mean"}).reset_index()
        df_grouped.rename(columns={"lat_bin": "latitude", "lon_bin": "longitude"}, inplace=True)
        print(f"[DEBUG] Number of points after groupby: {len(df_grouped)}")
    else:
        # No binning: plot all points
        df_grouped = df_plot.rename(columns={"lat": "latitude", "lon": "longitude"})

    if create_date_folder:
        map_output_dir = os.path.join(output_dir, date)
    else:
        map_output_dir = output_dir
    os.makedirs(map_output_dir, exist_ok=True)

    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)

    scatter = ax.scatter(
        df_grouped["longitude"], df_grouped["latitude"],
        c=df_grouped["prediction"],
        cmap=cmap,
        s=10,
        transform=ccrs.PlateCarree(),
        alpha=0.7,
        vmin=vmin,
        vmax=vmax
    )

    # Determine colorbar label
    model_name_lower = model_name.lower()
    if "relative_error" in model_name_lower:
        colorbar_label = "Relative Error (%)"
    elif "absolute_error" in model_name_lower:
        colorbar_label = "Absolute Error (°K)"
    else:
        colorbar_label = "Predicted LST (°K)"

    plt.title(f"LST Prediction – {model_name} – {date}")
    plt.colorbar(scatter, label=colorbar_label, orientation="vertical", shrink=0.7, pad=0.05)
    plt.tight_layout()

    file_path = os.path.join(map_output_dir, f"{model_name}_{data_type}_map_{date}.png")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path, dpi=600)
    plt.close()
    print(f"✅ Map saved at: {file_path}")


def plot_error_map(df_day, model_name, date_str, output_base_dir, data_type):
    # Compute absolute and relative errors
    df_day["abs_error"] = np.abs(df_day["prediction"] - df_day["true"])
    df_day["rel_error"] = 100 * df_day["abs_error"] / (np.abs(df_day["true"]) + 1e-6)

    # Filter out extreme relative error values
    df_day_rel_filtered = df_day[df_day["rel_error"] <= 200]

    # 1. Absolute error map
    abs_dir = os.path.join(output_base_dir, "absolute_error_maps")
    os.makedirs(abs_dir, exist_ok=True)
    plot_prediction_map(
        df=df_day,
        y_pred=df_day["abs_error"].values,
        model_name=f"{model_name}_absolute_error",
        date=date_str,
        output_dir=abs_dir,
        cmap="Reds", 
        data_type=data_type
    )

    # 2. Relative error map
    rel_dir = os.path.join(output_base_dir, "relative_error_maps")
    os.makedirs(rel_dir, exist_ok=True)
    plot_prediction_map(
        df=df_day_rel_filtered,
        y_pred=df_day_rel_filtered["rel_error"].values,
        model_name=f"{model_name}_relative_error",
        date=date_str,
        output_dir=rel_dir,
        cmap="Reds", 
        data_type=data_type
    )


def plot_mean_map(df_test_plot, model_name, data_type, output_dir="outputs/machine_learning/mean_values_maps"):
    # Moyenne des prédictions par point géographique
    df_mean_pred = df_test_plot.groupby(["lat", "lon"])["prediction"].mean().reset_index()

    # Création du dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    mean_map_path = os.path.join(output_dir, f"{model_name}_{data_type}_mean_value.png")

    # === Plot Cartopy ===
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False

    scatter = ax.scatter(
        df_mean_pred["lon"], df_mean_pred["lat"],
        c=df_mean_pred["prediction"],
        cmap="viridis",
        s=10,
        transform=ccrs.PlateCarree(),
        alpha=0.7,
        vmin=np.min(df_mean_pred["prediction"]),
        vmax=np.max(df_mean_pred["prediction"])
    )

    plt.title(f"{model_name} - Mean Predictions (2005)")
    plt.colorbar(scatter, label="Mean Predicted Temperature (°K)", orientation="vertical", shrink=0.7, pad=0.05)
    plt.tight_layout()

    plt.savefig(mean_map_path, dpi=600)
    plt.close()
    print(f"\n✅ Mean temperature {data_type} map saved at: {mean_map_path}")


def plot_mean_error_map(df_test_plot, df_mean_true, model_name, data_type, output_dir="outputs/machine_learning/mean_values_maps"):

    # Moyenne des prédictions
    df_mean_pred = df_test_plot.groupby(["lat", "lon"])["prediction"].mean().reset_index()

    # Moyenne des vraies valeurs – très important
    df_mean_true = df_mean_true.groupby(["lat", "lon"])["LST_Kelvin_mean"].mean().reset_index()
    df_mean_true = df_mean_true.rename(columns={"LST_Kelvin_mean": "true"})

    # Fusion des deux
    df_merged = pd.merge(df_mean_pred, df_mean_true, on=["lat", "lon"], how="inner")

    # Supprimer les doublons potentiels (pas strictement nécessaire mais défensif)
    df_merged = df_merged.drop_duplicates(subset=["lat", "lon"])

    # Calcul des erreurs
    df_merged["abs_error"] = np.abs(df_merged["prediction"] - df_merged["true"])
    df_merged["rel_error"] = 100* df_merged["abs_error"] / (np.abs(df_merged["true"]) + 1e-6)

    # Filtrage des erreurs relatives extrêmes
    df_rel_filtered = df_merged[df_merged["rel_error"] <= 5.0]

    output_path = os.path.join(output_dir, model_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Carte d’erreur absolue
    plot_prediction_map(
        df=df_merged,
        y_pred=df_merged["abs_error"].values,
        model_name=f"{model_name}_mean_absolute_error",
        date="2005",
        output_dir=output_path,
        cmap="Reds",
        data_type=data_type, 
        create_date_folder=False
    )

    # 2. Carte d’erreur relative
    plot_prediction_map(
        df=df_rel_filtered,
        y_pred=df_rel_filtered["rel_error"].values,
        model_name=f"{model_name}_mean_relative_error",
        date="2005",
        output_dir=output_path,
        cmap="Reds",
        data_type=data_type, 
        create_date_folder=False
    )


def plot_error_distributions(y_true, y_pred, output_dir, model_name, data_type, bins=50):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Création du dossier de sortie
    error_dist_path = os.path.join(output_dir, "errors_distribution")
    os.makedirs(error_dist_path, exist_ok=True)
    error_dist_file = os.path.join(error_dist_path, f"{model_name}_{data_type}_error_distributions.png")

    # Calcul des erreurs absolues et relatives
    abs_errors = np.abs(y_true - y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_errors = np.where(y_true != 0, abs_errors / np.abs(y_true), np.nan)

    # Filtrage des erreurs relatives > 400%
    rel_errors_filtered = 100*rel_errors[(~np.isnan(rel_errors)) & (rel_errors <= 4)]

    # Statistiques
    abs_median = np.median(abs_errors)
    abs_std = np.std(abs_errors)
    rel_median = np.median(rel_errors_filtered)
    rel_std = np.std(rel_errors_filtered)

    # Limites dynamiques pour les xlim
    abs_max = np.percentile(abs_errors, 99)  # Évite les outliers extrêmes
    rel_max = np.percentile(rel_errors_filtered, 99)

    # Marge pour lisibilité
    abs_xlim = (0, abs_max * 1.1)
    rel_xlim = (0, rel_max * 1.1)

    # Tracé
    plt.figure(figsize=(14, 6))

    # Erreurs absolues
    plt.subplot(1, 2, 1)
    sns.histplot(abs_errors, bins=bins, kde=True, color="skyblue")
    plt.axvline(abs_median, color='blue', linestyle='--', label=f"Median : {abs_median:.2f}")
    plt.axvline(abs_median + abs_std, color='green', linestyle='--', label=f"Std : {abs_std:.2f}")
    plt.axvline(abs_median - abs_std, color='green', linestyle='--')
    plt.xlim(abs_xlim)
    plt.title("Absolute errors distribution")
    plt.xlabel("Absolute error")
    plt.ylabel("Frequency")
    plt.legend()

    # Erreurs relatives
    plt.subplot(1, 2, 2)
    sns.histplot(rel_errors_filtered, bins=bins, kde=True, color="salmon")
    plt.axvline(rel_median, color='red', linestyle='--', label=f"Median : {rel_median:.2f}")
    plt.axvline(rel_median + rel_std, color='purple', linestyle='--', label=f"Std : {rel_std:.2f}")
    plt.axvline(rel_median - rel_std, color='purple', linestyle='--')
    plt.xlim(rel_xlim)
    plt.title("Relative errors distribution (≤ 400%)")
    plt.xlabel("Relative errors (%)")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.savefig(error_dist_file, dpi=300)
    plt.close()

    print(f"✅ Error distribution saved at: {error_dist_file}")


def plot_error_histogram_vs_modis(df_comparison, model_name, output_dir, data_type):
    plt.figure(figsize=(8, 6))
    plt.hist(df_comparison["diff_pred_vs_modis"].dropna(), bins=50, color="teal", edgecolor="black")
    plt.axvline(0, color='red', linestyle='--')
    plt.xlim(-40,40)
    plt.title(f"Difference between prediction and MODIS — {model_name}")
    plt.xlabel("Error (K)")
    plt.ylabel("Number of points")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{model_name}_{data_type}_hist_diff_vs_modis.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300)
    plt.close()


def plot_error_by_landcover(df, model_name, output_dir, data_type, land_cover_mapping):
    df = df.copy()
    df["land_cover_name"] = df["land_cover_class"].map(land_cover_mapping)

    error_by_cover = (df["diff_pred_vs_modis"].abs()).groupby(df["land_cover_name"]).mean().sort_values()

    plt.figure(figsize=(12,6))
    error_by_cover.plot(kind='bar')
    plt.title(f"Mean Absolute Error by Land Cover Class ({model_name} - {data_type})")
    plt.xlabel("Land Cover Class")
    plt.ylabel("Mean Absolute Error")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt_path = os.path.join(output_dir, f"{model_name}_{data_type}_error_by_landcover.png")
    os.makedirs(os.path.dirname(plt_path), exist_ok=True)
    plt.savefig(plt_path)
    plt.close()


def plot_daily_error_trend(df, model_name, output_dir, data_type):
    df['date'] = pd.to_datetime(df['date'])
    df['year_month'] = df['date'].dt.to_period('M')
    
    for period, group in df.groupby('year_month'):
        daily_error = group['diff_pred_vs_modis'].abs().groupby(group['date']).mean()

        plt.figure(figsize=(12, 6))
        plt.plot(daily_error.index, daily_error.values, marker='o')
        plt.title(f"Daily Mean Absolute Error vs MODIS - {model_name} - {period} ({data_type})")
        plt.xlabel("Date")
        plt.ylabel("Mean Absolute Error")
        plt.grid(True)
        plt.xticks(rotation=45)

        filename = f"{model_name}_{data_type}_daily_error_trend_{period}.png"
        os.makedirs(os.path.join(output_dir, str(period)), exist_ok=True)
        filepath = os.path.join(output_dir, str(period), filename)

        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()

