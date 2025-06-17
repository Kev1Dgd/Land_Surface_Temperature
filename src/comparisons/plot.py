import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_tb_comparison(daily_means, freq):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))  # 2x2 grilles
    axes = axes.flatten()  # facilite l'accès par index 0 à 3

    # Liste des comparaisons à faire : (subplot_index, pol/orb, pol/orb, orb1, orb2, titre)
    comparisons = [
        (0, ['horizontal', 'vertical'], 'ascending', 'polarization', f"{freq} — Ascending — Polarization Comparison"),
        (1, ['horizontal', 'vertical'], 'descending', 'polarization', f"{freq} — Descending — Polarization Comparison"),
        (2, ['horizontal'], ['ascending', 'descending'], 'orbit', f"{freq} — Horizontal polarization — Orbit Comparison"),
        (3, ['vertical'], ['ascending', 'descending'], 'orbit', f"{freq} — Vertical polarization — Orbit Comparison"),
    ]

    for idx, pols_or_orbits1, pols_or_orbits2, comp_type, title in comparisons:
        ax = axes[idx]
        y_vals = []

        # Récupération des dataframes et tracé
        dfs = []
        labels = []

        if comp_type == 'polarization':
            # pols_or_orbits1 = list de polarisations, pols_or_orbits2 = orbite commune
            orb = pols_or_orbits2
            for pol in pols_or_orbits1:
                df = daily_means.get((freq, pol, orb))
                if df is not None:
                    ax.plot(df['date'], df['brightness_temp'], label=pol.capitalize())
                    y_vals.extend(df['brightness_temp'].values)
                    dfs.append(df)
                    labels.append(pol)
        else:
            # comp_type == 'orbit'
            pol = pols_or_orbits1[0]
            for orb in pols_or_orbits2:
                df = daily_means.get((freq, pol, orb))
                if df is not None:
                    ax.plot(df['date'], df['brightness_temp'], label=orb.capitalize())
                    y_vals.extend(df['brightness_temp'].values)
                    dfs.append(df)
                    labels.append(orb)

        # Calcul stats si on a deux séries pour comparer
        if len(dfs) == 2:
            merged = pd.merge(dfs[0], dfs[1], on='date', suffixes=('_1', '_2'))
            diffs = merged['brightness_temp_1'] - merged['brightness_temp_2']

            mean_abs_diff = np.mean(np.abs(diffs))
            std_diff = np.std(diffs)
            corr = merged['brightness_temp_1'].corr(merged['brightness_temp_2'])

            stats_text = (
                f"Mean abs diff: {mean_abs_diff:.2f} K\n"
                f"Std diff: {std_diff:.2f} K\n"
                f"Corr: {corr:.2f}"
            )
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

        ax.set_title(title)
        ax.legend()
        ax.grid()
        ax.set_xlabel("Date")
        ax.set_ylabel("Mean TB (K)")

        if y_vals:
            margin = (max(y_vals) - min(y_vals)) * 0.05
            ax.set_ylim(min(y_vals) - margin, max(y_vals) + margin)

    plt.tight_layout()

    # 🔽 Sauvegarde du graphique
    output_dir = "outputs/comparisons"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"tb_comparison_{freq.replace('GHz','')}GHz.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Graphe enregistré dans : {output_path}")
