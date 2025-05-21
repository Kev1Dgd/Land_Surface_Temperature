import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import os

def plot_land_cover_map(nc_path, output_img_path):
    lon_min, lat_min, lon_max, lat_max = -12.984, 35.290, 38.018, 64.09

    # Data loading
    ds = xr.open_dataset(nc_path)
    var_name = list(ds.data_vars)[0]
    land_cover = ds[var_name]

    # Dynamic management of latitude order for slicing
    if land_cover.latitude.values[0] > land_cover.latitude.values[-1]:
        lat_slice = slice(lat_max, lat_min)  # decreasing latitudes
    else:
        lat_slice = slice(lat_min, lat_max)  # growing latitudes

    land_cover = land_cover.sel(
        latitude=lat_slice,
        longitude=slice(lon_min, lon_max)
    )

    # Loading classes in English
    classes_df = pd.read_csv("data/processed/land_cover/land_cover_lookup.csv")
    id_to_label = dict(zip(classes_df["id"], classes_df["igbp_class_en"]))

    # Colors for up to 18 classes
    cmap = plt.get_cmap('tab20', 18)

    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    im = land_cover.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        add_colorbar=False
    )
    
    # Define geographical limits
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')

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
    print(f"✅ Map saved in : {output_img_path}")
