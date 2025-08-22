from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd

def train_and_evaluate_rf_snow(df, features=None, target_col="is_snow_modis", random_state=42):
    if features is None:
        features = ["brightness_temp_19v", "brightness_temp_37v", "brightness_temp_19h", "brightness_temp_37h"]

    print("🔍 Preparing data for Random Forest snow classification...")
    df_clean = df.dropna(subset=features + [target_col])

    X = df_clean[features]
    y = df_clean[target_col]

    print(f"Dataset size: {len(df_clean)} samples")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=random_state
    )

    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n📊 Classification report (Random Forest Snow):")
    print(classification_report(y_test, y_pred, digits=3))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, X_test, y_test, y_pred, y_proba

def plot_rf_snow_map(df, output_dir="outputs/snow_detection"):
    print("🗺️ Generating RandomForest snow map...")
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-13, 38, 35, 65], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)

    snow_df = df[df["prediction"] == 1]
    if snow_df.empty:
        print("⚠️ No snow pixels predicted by RandomForest.")
        return

    sc = ax.scatter(
        snow_df["lon"], snow_df["lat"],
        c='royalblue', s=8, alpha=0.7,
        transform=ccrs.PlateCarree(),
        label="RF predicted snow"
    )

    plt.title("Detected Snow Pixels (RandomForest)")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(output_dir, "map_rf_snow_prediction.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"📸 RandomForest snow map saved to: {out_path}")


def plot_rf_snow_monthly_maps(df, output_dir="outputs/snow_detection/rf_monthly_maps"):
    print("🗓️ Generating monthly RF snow maps...")
    os.makedirs(output_dir, exist_ok=True)

    # Extraire le mois
    df = df.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.month

    for month in sorted(df["month"].unique()):
        df_month = df[df["month"] == month]
        snow_df = df_month[df_month["prediction"] == 1]

        if snow_df.empty:
            print(f"⚠️ No snow pixels predicted for month {month:02d}")
            continue

        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([-13, 38, 35, 65], crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)

        ax.scatter(
            snow_df["lon"], snow_df["lat"],
            c="royalblue", s=8, alpha=0.7,
            transform=ccrs.PlateCarree(),
            label="RF predicted snow"
        )

        plt.title(f"Snow Prediction (RF) - Month {month:02d}")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"snow_rf_month_{month:02d}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"📸 Saved: {out_path}")
