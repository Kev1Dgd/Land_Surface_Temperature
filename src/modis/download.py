from earthaccess import login, search_data, download as earth_download
from pathlib import Path

def authenticate():
    """Authenticate the user with Earthdata."""
    print("🔐 Authentification Earthdata...")
    return login()

def search_modis_lst(start_date="2005-01-01", end_date="2005-12-31", bbox=(-10, 35, 10, 45)):
    """Search MOD11A1 Terra LST data for a specific time range and bounding box."""
    print(f"🔍 Searching MODIS LST from {start_date} to {end_date}...")
    results = search_data(
        short_name="MOD11A1",
        temporal=(start_date, end_date),
        bounding_box=bbox,
    )
    return results

def download_results(results, output_dir="data/raw/modis"):
    """Download search results into a local directory."""
    print(f"⬇️ Downloading {len(results)} MODIS files...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    downloaded_files = earth_download(results, local_path=output_dir)
    print(f"✅ Download complete: {len(downloaded_files)} files saved to {output_dir}")
    return downloaded_files