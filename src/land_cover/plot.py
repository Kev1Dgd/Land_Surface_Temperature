import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import os

def plot_land_cover_map(nc_path, output_img_path):
    # Limites géographiques ciblées
    lon_min, lat_min, lon_max, lat_max = -12.984, 35.290, 38.018, 64.09

    # Chargement des données
    ds = xr.open_dataset(nc_path)
    var_name = list(ds.data_vars)[0]
    land_cover = ds[var_name]

    # Gestion dynamique de l'ordre des latitudes pour le slicing
    if land_cover.latitude.values[0] > land_cover.latitude.values[-1]:
        lat_slice = slice(lat_max, lat_min)  # latitudes décroissantes
    else:
        lat_slice = slice(lat_min, lat_max)  # latitudes croissantes

    land_cover = land_cover.sel(
        latitude=lat_slice,
        longitude=slice(lon_min, lon_max)
    )

    # Chargement des classes en anglais
    classes_df = pd.read_csv("data/processed/land_cover/land_cover_lookup.csv")
    id_to_label = dict(zip(classes_df["id"], classes_df["igbp_class_en"]))

    # Couleurs pour 18 classes max
    cmap = plt.get_cmap('tab20', 18)

    # Création de la figure
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    im = land_cover.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        add_colorbar=False
    )
    
    # Définir les limites géographiques
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Ajouter une légende personnalisée (à gauche)
    unique_classes = np.unique(land_cover.values)
    handles = [
        mpatches.Patch(color=cmap(i), label=f"{i}: {id_to_label.get(i, 'Unknown')}")
        for i in unique_classes if i in id_to_label
    ]
    plt.legend(handles=handles, title="IGBP Land Cover Classes", loc='center left', bbox_to_anchor=(-0.35, 0.5), fontsize='small')

    plt.title("Land Cover Classes (zoomed view)")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    plt.savefig(output_img_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✅ Map saved at : {output_img_path}")
