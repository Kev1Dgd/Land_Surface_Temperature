import numpy as np
import pandas as pd

def clean_temp_fluxnet(value):
    if pd.isna(value):
        return np.nan
    # Supprimer tous les points
    value_clean = str(value).replace('.', '')
    try:
        temp = float(value_clean)
        if temp > 600:  
            temp = temp / 10000
        return temp
    except ValueError:
        return np.nan