import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_station_tb(csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ Fichier introuvable : {csv_path}")
        return

    df = pd.read_csv(csv_path)

    if df.empty:
        print("⚠️ Fichier vide, rien à afficher.")
        return

    # Nettoyage des données
    df = df.dropna(subset=["TB"])
    df = df[df["TB"] > 0]

    # Affichage
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(df["station_lon"], df["station_lat"], c=df["TB"], cmap="plasma", s=100, edgecolor="k")

    for _, row in df.iterrows():
        ax.text(row["station_lon"] + 0.1, row["station_lat"], row["station"], fontsize=8)

    plt.colorbar(sc, label="TB 37GHz (K)")
    ax.set_title("Température de brillance AMSR-E 37GHz par station")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
