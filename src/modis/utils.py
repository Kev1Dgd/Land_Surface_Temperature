import numpy as np

def convert_kelvin_to_celsius(lst_data):
    """Convert MODIS LST values from Kelvin to Celsius."""
    return (lst_data * 0.02) - 273.15

def clean_lst_data(lst_data):
    """Replace invalid or missing values in LST data."""
    lst_data[lst_data == 0] = np.nan
    return lst_data