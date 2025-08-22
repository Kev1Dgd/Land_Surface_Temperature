from earthaccess import login, search_data, download
import os
import time

def authenticate():
    print("🔐 Authentification Earthdata...")
    return login()


def download_amsre_ae_l2a(date, output_dir="data/raw/amsre"):
    print(f"\n🔍 Search AMSR-E AE_L2A data for {date}")

    results = search_data(
        short_name="AE_L2A",
        temporal=(date, date),
        cloud_hosted=True
    )

    os.makedirs(output_dir, exist_ok=True)

    max_retries = 3
    delay = 5  

    for attempt in range(1, max_retries + 1):
        try:
            print(f"📥 Attempt no.{attempt} of download...")
            files = download(results, local_path=output_dir)
            print(f"✅ {len(files)} files uploaded to {output_dir}")
            return files
        except Exception as e:
            print(f"⚠️ Download error : {e}")
            if attempt < max_retries:
                print(f"🔁 New attempt in {delay} seconds...")
                time.sleep(delay)
            else:
                print("❌ Failed after 3 attempts. Abort.")
                raise