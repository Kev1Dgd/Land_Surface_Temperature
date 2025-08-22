import os
import calendar
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from sklearn.metrics import roc_curve, auc

def plot_snow_map(df, output_dir):
    print("🗺️ Generating snow map...")
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-13, 38, 35, 65], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)

    snow_df = df[df["is_snow_physical"] == 1]
    sc = ax.scatter(
        snow_df["lon"], snow_df["lat"],
        c='deepskyblue', s=8, alpha=0.7,
        transform=ccrs.PlateCarree(),
        label="Detected snow"
    )

    plt.title("Detected Snow Pixels (Physical Method)")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(output_dir, "map_snow_physical.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"📸 Snow map saved to: {out_path}")


def plot_snow_frequency_map(df_freq, output_dir):
    print("\n🧊 Generating snow frequency map...")
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-13, 38, 35, 65], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)

    sc = ax.scatter(
        df_freq["lon"], df_freq["lat"],
        c=100*df_freq["snow_freq"],
        cmap="Blues",
        s=10, alpha=0.8,
        transform=ccrs.PlateCarree()
    )

    cbar = plt.colorbar(sc, label="Snow Frequency (%)", orientation="vertical", shrink=0.7)
    plt.title("Snow Frequency Map (Physical Detection)")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "map_snow_frequency.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"📊 Snow frequency map saved to: {out_path}")


def plot_monthly_snow_maps(df, output_dir):
    print("\n🗓️ Generating monthly snow frequency maps...")
    monthly_dir = os.path.join(output_dir, "monthly_snow_maps")
    os.makedirs(monthly_dir, exist_ok=True)

    df["month"] = pd.to_datetime(df["date"]).dt.month

    for month in range(1, 13):
        df_month = df[df["month"] == month]
        if df_month.empty:
            print(f"⚠️ No data for month {month}. Skipping...")
            continue

        df_freq = df_month.groupby(["lat", "lon"])["is_snow_physical"].mean().reset_index()
        df_freq.rename(columns={"is_snow_physical": "snow_freq"}, inplace=True)

        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([-13, 38, 35, 65], crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)

        sc = ax.scatter(
            df_freq["lon"], df_freq["lat"],
            c=100*df_freq["snow_freq"],
            cmap="Blues",
            s=10, alpha=0.8,
            transform=ccrs.PlateCarree()
        )

        cbar = plt.colorbar(sc, label="Snow Frequency (%)", orientation="vertical", shrink=0.7)
        month_name = calendar.month_name[month]
        plt.title(f"Snow Frequency - {month_name}")
        plt.tight_layout()

        out_path = os.path.join(monthly_dir, f"snow_freq_{month:02d}_{month_name}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"🗺️ Saved: {out_path}")



def plot_monthly_snow_barplot(df, output_dir):
    print("\n📊 Generating monthly snow barplot...")

    df["month"] = pd.to_datetime(df["date"]).dt.month

    df_summary = df.groupby("month")[["is_snow_physical", "is_snow_modis"]].mean().reset_index()

    plt.figure(figsize=(10, 6))
    plt.plot(df_summary["month"], 100*df_summary["is_snow_physical"], marker='o', label="Physical Method", color='deepskyblue')
    plt.xticks(ticks=range(1,13), labels=[
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ])
    plt.xlabel("Month")
    plt.ylabel("Mean Snow Frequency (%)")
    plt.title("Monthly Snow Detection (Mean Frequency)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(output_dir, "monthly_snow_barplot.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"📊 Saved barplot to: {out_path}")