import os
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature



def generate_full_heatmap_correlation(df, OUTPUT_DIR):
    heatmap_dir = os.path.join(OUTPUT_DIR, "correlation_heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    ## Full correlation matrix
    correlation_matrix = df.corr(numeric_only=True)
    
    # Mask for the upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix, 
        annot=True, fmt=".2f", cmap="coolwarm", 
        square=True, mask=mask, 
        cbar_kws={"shrink": .8}
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(heatmap_dir, "correlation_heatmap.png"), dpi=300)
    plt.close()
    print("✅ Correlation heatmap saved.")

    ## Targeted correlation
    cols_of_interest = ["LST_Kelvin","brightness_temp_19v_asc","brightness_temp_19v_desc","brightness_temp_37v_asc","brightness_temp_37v_desc","brightness_temp_19h_asc","brightness_temp_19h_desc","brightness_temp_37h_asc","brightness_temp_37h_desc","land_cover_class"]
    corr = df[cols_of_interest].corr()

    # Mask for the upper triangle in the targeted heatmap
    mask_subset = np.triu(np.ones_like(corr, dtype=bool), k=1)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr, annot=True, cmap="coolwarm", fmt=".2f", 
        mask=mask_subset,
        cbar_kws={"shrink": .8}
    )
    plt.title("Targeted Correlations")
    plt.tight_layout()
    plt.savefig(os.path.join(heatmap_dir, "correlation_subset_heatmap.png"), dpi=300)
    plt.close()
    print("✅ Targeted heatmap saved.")


def plot_feature_importances(model, feature_names, output_dir, model_name, data_type):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importances - {model_name} ({data_type})")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{model_name}_{data_type}_feature_importances.png"))
    plt.close()


def plot_prediction_map_comp(
    df, y_pred, model_name, date, output_dir, 
    cmap="inferno", data_type="unnormalized", 
    vmin=None, vmax=None, binning=False, create_date_folder=True
):
    print(f"🗺️ Generating map for model: {model_name} — {date}")

    df_plot = df.copy()
    df_plot["prediction"] = y_pred

    if binning:
        df_plot["lat_bin"] = df_plot["lat"].round(2)
        df_plot["lon_bin"] = df_plot["lon"].round(2)
        print(f"[DEBUG] Number of points before binning: {len(df_plot)}")
        df_grouped = df_plot.groupby(["lat_bin", "lon_bin"]).agg({"prediction": "mean"}).reset_index()
        df_grouped.rename(columns={"lat_bin": "latitude", "lon_bin": "longitude"}, inplace=True)
        print(f"[DEBUG] Number of points after binning: {len(df_grouped)}")
    else:
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

    # Label couleur
    model_name_lower = model_name.lower()
    if "relative_error" in model_name_lower:
        colorbar_label = "Relative Error (%)"
    elif "absolute_error" in model_name_lower:
        colorbar_label = "Absolute Error (°K)"
    elif "diff" in model_name_lower:
        colorbar_label = "LST Difference (°K)"
    else:
        colorbar_label = "Predicted LST (°K)"

    plt.title(f"LST – {model_name} – {date}")
    plt.colorbar(scatter, label=colorbar_label, orientation="vertical", shrink=0.7, pad=0.05)
    plt.tight_layout()

    filename = f"{model_name}_{data_type}_map_{date}.png"
    file_path = os.path.join(map_output_dir, filename)
    plt.savefig(file_path, dpi=600)
    plt.close()
    print(f"✅ Map saved at: {file_path}")

