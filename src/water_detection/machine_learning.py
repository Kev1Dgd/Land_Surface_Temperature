import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_confusion_matrix_rf(y_test, y_pred_class, output_dir):
    output_file = os.path.join(output_dir, "confusion_matrix_rf.png")
    os.makedirs(output_dir, exist_ok=True)

    cm = confusion_matrix(y_test, y_pred_class)

    labels = [["TN", "FP"], ["FN", "TP"]]
    colors = np.array([
        ["#89f58f", "#f79898"],  
        ["#f79898", "#89f58f"]   
    ])

    fig, ax = plt.subplots(figsize=(6, 4))

    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color=colors[i, j]))
            ax.text(j + 0.5, i + 0.5, f"{cm[i, j]}", ha='center', va='center', fontsize=14, fontweight='bold')

    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(["no-water", "water"])
    ax.set_yticklabels(["no-water", "water"])
    ax.set_xlabel("Predicts")
    ax.set_ylabel("Real")
    ax.set_title("Confusion matrix - Water detection")

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.invert_yaxis()
    ax.set_aspect('equal')

    legend_patches = [
        mpatches.Patch(color="#89f58f", label="Correct (TP or TN)"),
        mpatches.Patch(color="#f79898", label="Error (FP or FN)"),
    ]
    ax.legend(handles=legend_patches, loc='lower right', frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"✅ Color matrix saved in {output_file}")


def plot_water_map(df_water_pred, output_dir):
    lon_min, lat_min, lon_max, lat_max = -12.984, 35.290, 38.018, 64.090

    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
    ax.scatter(df_water_pred["lon"], df_water_pred["lat"], color='royalblue', s=10, alpha=0.6, transform=ccrs.PlateCarree(), label="Water predicted")

    plt.title("Model predictions: areas detected as water")
    plt.legend(loc='upper right')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "water_machine_learning_map_rf.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=600)
    plt.close()
    print(f"✅ Prediction map saved in : {output_file}")


def plot_roc_curve_rf(y_test, y_proba, output_dir):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random model')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC curve - Water detection')
    plt.legend(loc='lower right')
    output_file = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"✅ ROC curve saved in: {output_file}")

