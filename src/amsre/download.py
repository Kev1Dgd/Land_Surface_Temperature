from earthaccess import login, search_data, download
import os

def authenticate():
    """Authenticates the user to Earthdata."""
    print("🔐 Authentification Earthdata...")
    return login()

def download_amsre_ae_l2a(date="2005-07-01", output_dir="data/raw/amsre"):
    """Download AE_L2A files for a given date"""
    print(f"🔍 Search AMSR-E AE_L2A data for {date}")
    authenticate()
    results = search_data(
        short_name="AE_L2A",
        temporal=(date, date),
        cloud_hosted=True
    )
    os.makedirs(output_dir, exist_ok=True)
    files = download(results, local_path=output_dir)
    print(f"✅ {len(files)} files uploaded to {output_dir}")
    return files