import os
import pandas as pd
from glob import glob
from collections import defaultdict


def load_amsre_subsets_from_saved(input_dir):
    subsets = {}
    for filepath in glob(os.path.join(input_dir, "*.csv")):
        filename = os.path.basename(filepath)
        parts = filename.replace(".csv", "").split("_")
        if len(parts) != 4:
            print(f"⚠️ Unexpected file name : {filename}")
            continue
        _, freq, pol, orb = parts
        key = (freq+"GHz", "horizontal" if pol=='h' else "vertical", "ascending" if orb=='asc' else "descending")
        df = pd.read_csv(filepath)
        subsets[key] = df
        print(f"✅ loads {filename} for {key}")
    return subsets


def process_and_save_daily_means(base_dir, output_dir, freqs):
    os.makedirs(output_dir, exist_ok=True)

    all_dfs = defaultdict(list)

    for pol_folder in ["horizontal_polarization", "vertical_polarization"]:
        pol_path = os.path.join(base_dir, pol_folder)
        pol_type = "horizontal" if "horizontal" in pol_folder else "vertical"

        for date_folder in os.listdir(pol_path):
            date_path = os.path.join(pol_path, date_folder)
            if not os.path.isdir(date_path):
                continue

            for file_path in glob(os.path.join(date_path, "*.csv")):
                filename = os.path.basename(file_path)

                # Keep only authorized frequencies
                if not any(filename.startswith(f"amsre_{f}GHz") for f in freqs):
                    continue

                parts = filename.split("_")
                freq = parts[1]  # '19GHz'
                orbit = parts[-1].replace(".csv", "")  

                key = (freq, pol_type, orbit)

                freq_simple = freq.replace("GHz", "")
                pol_simple = "h" if pol_type == "horizontal" else "v"
                orb_simple = "asc" if orbit.startswith("asc") else "desc"
                out_fname = f"TB_{freq_simple}_{pol_simple}_{orb_simple}.csv"
                out_path = os.path.join(output_dir, out_fname)

                if os.path.exists(out_path):
                    print(f"⏭️ File already exists, skip: {out_path}")
                    continue

                df = pd.read_csv(file_path)
                df['date'] = pd.to_datetime(date_folder)
                all_dfs[key].append(df)

                print(f"✅ Loads {filename} => {key}")

    for (freq, pol, orb), list_dfs in all_dfs.items():
        print(f"\n📊 Processing : {freq}, {pol}, {orb} ({len(list_dfs)} files)")

        df_all = pd.concat(list_dfs, ignore_index=True)

        temp_cols = [col for col in df_all.columns if "brightness_temp" in col]
        if not temp_cols:
            print(f"⚠️ No brightness_temp column in {freq} {pol} {orb} — ignored.")
            continue

        tb_col = temp_cols[0]

        daily_mean = df_all.groupby('date')[tb_col].mean().reset_index()
        daily_mean.columns = ['date', 'brightness_temp']

        # Delete abnormally low days: below (median - 3 * std)
        median = daily_mean['brightness_temp'].median()
        std = daily_mean['brightness_temp'].std()
        threshold = median - 3 * std

        initial_count = len(daily_mean)
        daily_mean = daily_mean[daily_mean['brightness_temp'] >= threshold]
        filtered_count = len(daily_mean)
        print(f"🧹 Deleted days : {initial_count - filtered_count} (daily TB < {threshold:.1f} K)")

        freq_simple = freq.replace("GHz", "")
        pol_simple = "h" if pol == "horizontal" else "v"
        orb_simple = "asc" if orb.startswith("asc") else "desc"
        out_fname = f"TB_{freq_simple}_{pol_simple}_{orb_simple}.csv"
        out_path = os.path.join(output_dir, out_fname)

        daily_mean.to_csv(out_path, index=False)
        print(f"💾 Saved in : {out_path}")
