"""Download script for fetching raw dataset files."""

import os
import urllib.request
from pathlib import Path
from src.setup.constants import TAXONOMY_FILE, CSV_FILE

TAXONOMY_URL = "https://raw.githubusercontent.com/openfoodfacts/openfoodfacts-server/main/taxonomies/additives.txt"
CSV_URL = "https://raw.githubusercontent.com/suhasdissa/fssai-food-additives/main/additives.csv"

def download_file(url: str, dest: Path, name: str):
    print(f"Downloading {name}...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  [OK] Saved to {dest}")
    except Exception as e:
        print(f"  [X] Failed to download {name}: {e}")
        raise

def download_all():
    print("=" * 60)
    print(" 📥 Downloading Raw Datasets ")
    print("=" * 60)
    
    if not TAXONOMY_FILE.exists():
        download_file(TAXONOMY_URL, TAXONOMY_FILE, "Open Food Facts Taxonomy")
    else:
        print(f"  [SKIP] Taxonomy already exists: {TAXONOMY_FILE}")
        
    if not CSV_FILE.exists():
        download_file(CSV_URL, CSV_FILE, "SuhasDissa FSSAI Additives")
    else:
        print(f"  [SKIP] CSV already exists: {CSV_FILE}")

if __name__ == "__main__":
    download_all()
