import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import numpy as np
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix


def compute_swf_from_tb(df, tb_col='brightness_temp_19h', lst_col='LST_Kelvin', epsilon_w=0.48, epsilon_s=0.95):

    print("📥 Starting SWF physical estimation...")
    df = df.copy()

    print(f"🔸 Using TB column: {tb_col}")
    print(f"🔸 Using LST column: {lst_col}")
    print(f"🔧 Emissivities → ε_water: {epsilon_w}, ε_soil: {epsilon_s}")

    T_skin = df[lst_col]
    TB = df[tb_col]

    numerator = TB - (epsilon_s * T_skin)
    denominator = T_skin * (epsilon_w - epsilon_s)
    denominator = denominator.replace(0, 1e-6)

    df["SWF_physical"] = numerator / denominator
    df["SWF_physical"] = df["SWF_physical"].clip(0, 1)

    print("✅ Physical SWF estimation complete.")
    print(f"📊 Min SWF: {df['SWF_physical'].min():.3f}, Max SWF: {df['SWF_physical'].max():.3f}")
    print(f"📈 Average SWF: {df['SWF_physical'].mean():.3f}")

    # == Save results ==
    output_path = os.path.join("outputs/water_detection", "swf_physical_estimates.csv")
    df[["lat", "lon", "date", "SWF_physical"]].to_csv(output_path, index=False)
    print(f"💾 File saved to: {output_path}")

    return df

def plot_swf_map(df, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-13, 38, 35, 65], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)

    sc = ax.scatter(
        df["lon"], df["lat"],
        c=df["SWF_physical"],
        cmap="Blues",
        s=8, alpha=0.8,
        transform=ccrs.PlateCarree()
    )

    plt.colorbar(sc, label="SWF_physical", orientation="vertical", shrink=0.7)
    plt.title("SWF_physical map (physical water detection)")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "map_swf_physical.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"🗺️ Map saved to: {out_path}")

def plot_swf_water_map(df, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-13, 38, 35, 65], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)

    df_water = df[df["SWF_physical"] > 0.5]

    sc = ax.scatter(
        df_water["lon"], df_water["lat"],
        c=df_water["SWF_physical"],
        cmap="Blues",
        s=8, alpha=0.8,
        transform=ccrs.PlateCarree()
    )

    plt.colorbar(sc, label="SWF_physical", orientation="vertical", shrink=0.7)
    plt.title("SWF_physical map (physical water areas)")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "water_map_swf_physical.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"🗺️ Map saved to: {out_path}")


def plot_confusion_matrix_swf(y_test, y_pred_class, output_dir):
    output_file = os.path.join(output_dir, "confusion_matrix_swf.png")
    os.makedirs(output_dir, exist_ok=True)

    cm = confusion_matrix(y_test, y_pred_class)

    colors = np.array([
        ["#89f58f", "#f79898"],  
        ["#f79898", "#89f58f"]   
    ])

    _, ax = plt.subplots(figsize=(6, 4))

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


def plot_roc_curve_swf(df, output_dir, true_label_col="is_water_modis", score_col="SWF_physical"):
    print("🔍 Starting ROC curve computation...")
    
    y_true = df[true_label_col]
    y_scores = df[score_col]
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    print(f"✅ ROC AUC computed: {roc_auc:.3f}")
    
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - SWF Physical')
    plt.legend(loc="lower right")
    plt.grid(True)
    out_path = os.path.join(output_dir, "ROC_curve_SWF_physical.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    print("🎉 ROC curve plotted successfully.")


def compare_swf_rf(y_test, y_pred_class_rf, y_proba_rf, y_true_swf, y_swf_scores, output_dir):
    output_path = os.path.join(output_dir, "compare_sw_rf.png")
    print("🔍 Starting fair comparison between SWF and RF on test set...")

    # === SWF method ===
    y_swf_pred = (y_swf_scores > 0.5).astype(int)
    print("\n📊 SWF Physical Method:")
    print(confusion_matrix(y_true_swf, y_swf_pred))
    print(classification_report(y_true_swf, y_swf_pred, digits=3))

    # === RF model ===
    print("\n📊 RandomForest Model:")
    print(confusion_matrix(y_test, y_pred_class_rf))
    print(classification_report(y_test, y_pred_class_rf, digits=3))

    # === ROC Curves ===
    fpr_swf, tpr_swf, _ = roc_curve(y_true_swf, y_swf_scores)
    auc_swf = auc(fpr_swf, tpr_swf)

    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
    auc_rf = auc(fpr_rf, tpr_rf)

    plt.figure(figsize=(8,6))
    plt.plot(fpr_swf, tpr_swf, label=f'SWF Physical (AUC = {auc_swf:.3f})', color='blue')
    plt.plot(fpr_rf, tpr_rf, label=f'RandomForest (AUC = {auc_rf:.3f})', color='green')
    plt.plot([0,1], [0,1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("✅ Comparison completed.")
