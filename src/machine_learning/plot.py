import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import seaborn as sns
import pandas as pd
import numpy as np



def plot_results(y_test, y_pred, output_path):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.4)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual LST (°C)")
    plt.ylabel("Predicted LST (°C)")
    plt.title("MODIS Surface Temperature Prediction")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=600)
    plt.close()
    print(f"✅ Regression plot saved at: {output_path}")

def plot_prediction_map(df, y_pred, model_name, date, output_dir="outputs/machine_learning/", 
                        cmap="coolwarm", vmin=None, vmax=None, binning=False, create_date_folder=True):
    
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
        colorbar_label = "Absolute Error (°C)"
    else:
        colorbar_label = "Predicted LST (°C)"

    plt.title(f"LST Prediction – {model_name} – {date}")
    plt.colorbar(scatter, label=colorbar_label, orientation="vertical", shrink=0.7, pad=0.05)
    plt.tight_layout()

    file_path = os.path.join(map_output_dir, f"{model_name}_map_{date}.png")
    plt.savefig(file_path, dpi=600)
    plt.close()
    print(f"✅ Map saved at: {file_path}")


def plot_error_map(df_day, model_name, date_str, output_base_dir="outputs/machine_learning"):
    import numpy as np

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
        cmap="Reds"
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
        cmap="Reds"
    )



def plot_mean_map(df_test_plot, model_name, output_dir="outputs/machine_learning/mean_values_maps"):
    import numpy as np

    # Moyenne des prédictions par point géographique
    df_mean_pred = df_test_plot.groupby(["lat", "lon"])["prediction"].mean().reset_index()

    # Création du dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    mean_map_path = os.path.join(output_dir, f"{model_name}_mean_value.png")

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
    plt.colorbar(scatter, label="Mean Predicted Temperature (°C)", orientation="vertical", shrink=0.7, pad=0.05)
    plt.tight_layout()

    plt.savefig(mean_map_path, dpi=600)
    plt.close()
    print(f"\n✅ Mean temperature map saved at: {mean_map_path}")

def plot_mean_error_map(df_test_plot, df_mean_true, model_name, output_dir="outputs/machine_learning/mean_values_maps"):
    import numpy as np

    # Moyenne des prédictions
    df_mean_pred = df_test_plot.groupby(["lat", "lon"])["prediction"].mean().reset_index()

    # Moyenne des vraies valeurs – très important
    df_mean_true = df_mean_true.groupby(["lat", "lon"])["LST_Celsius_mean"].mean().reset_index()
    df_mean_true = df_mean_true.rename(columns={"LST_Celsius_mean": "true"})

    # Fusion des deux
    df_merged = pd.merge(df_mean_pred, df_mean_true, on=["lat", "lon"], how="inner")

    # Supprimer les doublons potentiels (pas strictement nécessaire mais défensif)
    df_merged = df_merged.drop_duplicates(subset=["lat", "lon"])

    # Calcul des erreurs
    df_merged["abs_error"] = np.abs(df_merged["prediction"] - df_merged["true"])
    df_merged["rel_error"] = df_merged["abs_error"] / (np.abs(df_merged["true"]) + 1e-6)

    # Filtrage des erreurs relatives extrêmes
    df_rel_filtered = df_merged[df_merged["rel_error"] <= 5.0]

    # 1. Carte d’erreur absolue
    plot_prediction_map(
        df=df_merged,
        y_pred=df_merged["abs_error"].values,
        model_name=f"{model_name}_mean_absolute_error",
        date="2005",
        output_dir=output_dir,
        cmap="Reds",
        create_date_folder=False
    )

    # 2. Carte d’erreur relative
    plot_prediction_map(
        df=df_rel_filtered,
        y_pred=df_rel_filtered["rel_error"].values,
        model_name=f"{model_name}_mean_relative_error",
        date="2005",
        output_dir=output_dir,
        cmap="Reds",
        create_date_folder=False
    )


def plot_error_distributions(y_true, y_pred, output_path, name, bins=50):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    error_dist_path = os.path.join(output_path, "errors_distribution")
    os.makedirs(error_dist_path, exist_ok=True)                               
    error_dist_file = os.path.join(error_dist_path, f"{name}_error_distributions.png")

    abs_errors = np.abs(y_true - y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_errors = np.where(y_true != 0, abs_errors / np.abs(y_true), np.nan)

    # Filter relative errors > 4 (400%)
    rel_errors_filtered = rel_errors[(~np.isnan(rel_errors)) & (rel_errors <= 4)]

    plt.figure(figsize=(14,6))

    plt.subplot(1, 2, 1)
    sns.histplot(abs_errors, bins=bins, kde=True, color="skyblue")
    plt.title("Distribution of Absolute Errors")
    plt.xlabel("Absolute Error")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    sns.histplot(rel_errors_filtered, bins=bins, kde=True, color="salmon")
    plt.title("Distribution of Relative Errors (<= 200%)")
    plt.xlabel("Relative Error (fraction)")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(error_dist_file)
    plt.close()