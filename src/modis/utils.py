import numpy as np
import os


def clean_lst_data(lst_data):
    """Replace invalid or missing values in LST data."""
    lst_data[lst_data == 0] = np.nan
    return lst_data

def check_and_create_file(output_file, create_func, input_dir):
    if os.path.exists(output_file):
        print(f"{output_file} file already exists. No new generation carried out.")
        return
    else:
        print(f"📂 File {output_file} creation...")
        create_func(input_dir, output_file) 
