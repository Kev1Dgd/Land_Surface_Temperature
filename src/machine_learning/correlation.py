import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os


def generate_heatmap_correlation(df, OUTPUT_DIR):
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
    cols_of_interest = ["LST_Celsius", "brightness_temp_19v", "brightness_temp_37v", "brightness_temp_19h", "brightness_temp_37h", "land_cover_class"]
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
