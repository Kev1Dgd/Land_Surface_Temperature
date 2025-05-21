import os
from glob import glob

def fix_amsre_headers():
    # Path to daily AMSRE files
    amsre_folder = "data/processed/amsre"
    header = "date,latitude,longitude,brightness_temp_37v,brightness_temp_19v\n"

    # Browse all merged_amsre_data_*.csv files in subfolders
    for root, dirs, files in os.walk(amsre_folder):
        for file in files:
            if file.startswith("merged_amsre_data_") and file.endswith(".csv"):
                file_path = os.path.join(root, file)

                # Read existing content
                with open(file_path, "r") as f:
                    content = f.read()

                # Check if the file already contains the
                if not content.startswith("date,latitude"):
                    print(f"✍️ Add header to {file_path}")
                    with open(file_path, "w") as f:
                        f.write(header + content)
                else:
                    print(f"✅ Header already present in {file_path}")