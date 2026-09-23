"""
download_dataset.py — AI Civic Guardian
========================================
Downloads a filtered slice of Chicago 311 Service Requests
using the Socrata Open Data API (no login needed, free).

Instead of downloading 14 million rows (2+ GB),
this fetches only the complaint types we care about — ~10,000 rows total.

USAGE:
    python download_dataset.py

OUTPUT:
    dataset/text/311_Service_Requests.csv
"""

import os
import urllib.request
import urllib.parse
import json
import csv
import time

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset", "text")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "311_Service_Requests.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── API CONFIG ───────────────────────────────────────────────────────────────
# Chicago 311 dataset on Socrata
# Dataset ID: v6vf-nfxy
SOCRATA_BASE = "https://data.cityofchicago.org/resource/v6vf-nfxy.json"

# We fetch these SR_TYPE values — covers our 12 categories
# Fetching ~800 records per type to get a balanced dataset
QUERY_TYPES = [
    # Road & Pothole
    "Pothole in Street Complaint",
    "Alley Pothole Complaint",
    # Garbage & Sanitation
    "Sanitation Code Violation",
    "Fly Dumping Complaint",
    "Missed Garbage Pick-Up Complaint",
    # Animal & Pest
    "Rodent Baiting/Rat Complaint",
    "Dead Animal Pick-Up Request",
    "Stray Animal Complaint",
    # Water & Drainage
    "Water On Street Complaint",
    "Water in Basement Complaint",
    "Sewer Cleaning Inspection Request",
    "No Water Complaint",
    # Streetlight
    "Street Light Out Complaint",
    "Alley Light Out Complaint",
    "Street Light Pole Damage Complaint",
    # Graffiti
    "Graffiti Removal Request",
    # Building
    "Building Violation",
    "Vacant/Abandoned Building Complaint",
    # Tree
    "Tree Debris Clean-Up Request",
    "Tree Trim Request (NO LONGER BEING ACCEPTED)",
    # Traffic
    "Traffic Signal Out Complaint",
    "Sign Repair Request - All Other Signs",
    # Noise
    "Aircraft Noise Complaint",
]

RECORDS_PER_TYPE = 300   # fetch 300 per SR_TYPE → ~7000 rows total (faster)
LIMIT_PER_REQUEST = 300  # smaller request = faster response


# ─── CATEGORY MAPPING ─────────────────────────────────────────────────────────
CATEGORY_MAP = {
    # Road
    "Pothole in Street Complaint"                    : "Road & Pothole",
    "Alley Pothole Complaint"                        : "Road & Pothole",
    # Garbage
    "Sanitation Code Violation"                      : "Garbage & Sanitation",
    "Fly Dumping Complaint"                          : "Garbage & Sanitation",
    "Missed Garbage Pick-Up Complaint"               : "Garbage & Sanitation",
    # Animal
    "Rodent Baiting/Rat Complaint"                   : "Animal & Pest",
    "Dead Animal Pick-Up Request"                    : "Animal & Pest",
    "Stray Animal Complaint"                         : "Animal & Pest",
    # Water
    "Water On Street Complaint"                      : "Water & Drainage",
    "Water in Basement Complaint"                    : "Water & Drainage",
    "Sewer Cleaning Inspection Request"              : "Water & Drainage",
    "No Water Complaint"                             : "Water & Drainage",
    # Streetlight
    "Street Light Out Complaint"                     : "Streetlight",
    "Alley Light Out Complaint"                      : "Streetlight",
    "Street Light Pole Damage Complaint"             : "Streetlight",
    # Graffiti
    "Graffiti Removal Request"                       : "Graffiti & Vandalism",
    # Building
    "Building Violation"                             : "Building & Infrastructure",
    "Vacant/Abandoned Building Complaint"            : "Building & Infrastructure",
    # Tree
    "Tree Debris Clean-Up Request"                   : "Tree & Greenery",
    "Tree Trim Request (NO LONGER BEING ACCEPTED)"   : "Tree & Greenery",
    # Traffic
    "Traffic Signal Out Complaint"                   : "Traffic & Signals",
    "Sign Repair Request - All Other Signs"          : "Traffic & Signals",
    # Noise
    "Aircraft Noise Complaint"                       : "Noise Complaint",
}


def fetch_records(sr_type, limit=300):
    """Fetch records for a specific SR_TYPE from Socrata API."""
    params = urllib.parse.urlencode({
        "$where" : f"sr_type='{sr_type}'",
        "$limit" : limit,
        "$select": "sr_type",
    })
    url = SOCRATA_BASE + "?" + params

    for attempt in range(2):   # try twice
        try:
            req = urllib.request.Request(
                url,
                headers={"Accept": "application/json",
                         "User-Agent": "AI-Civic-Guardian/1.0"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data
        except Exception as e:
            if attempt == 0:
                time.sleep(1)   # wait 1s then retry
            else:
                print(f"    [WARN] Failed to fetch '{sr_type}': {e}")
    return []


def build_description(sr_type):
    """
    Generate a natural language description from SR_TYPE.
    This is used as the text input for classification.
    """
    # Clean up the SR_TYPE string into a complaint-like sentence
    cleaned = sr_type.lower()\
        .replace(" - ", " ")\
        .replace("/", " ")\
        .replace("-", " ")\
        .strip()

    templates = [
        f"There is a {cleaned} issue reported in the area",
        f"Complaint about {cleaned} near residential area",
        f"Reporting {cleaned} problem on the street",
        f"{cleaned} needs immediate attention",
        f"Issue reported: {cleaned}",
    ]

    # Use a simple rotating pattern
    return templates


def download():
    print("\n=== AI Civic Guardian — Dataset Downloader ===")
    print(f"Fetching ~{len(QUERY_TYPES) * RECORDS_PER_TYPE} records from Chicago 311 API...\n")

    all_rows = []

    for i, sr_type in enumerate(QUERY_TYPES):
        category = CATEGORY_MAP.get(sr_type, "General")
        print(f"[{i+1:02d}/{len(QUERY_TYPES)}] Fetching: {sr_type} → {category} ...", end=" ")

        records = fetch_records(sr_type, RECORDS_PER_TYPE)
        if records:
            templates = build_description(sr_type)
            for j, rec in enumerate(records):
                # Use sr_type as description (cleaned) — rotating templates
                desc = templates[j % len(templates)]
                all_rows.append({
                    "description": desc,
                    "category"   : category,
                    "sr_type"    : rec.get("sr_type", sr_type),
                })
            print(f"{len(records)} records")
        else:
            print("0 records (skipped)")

        # Be polite to the API
        time.sleep(0.1)

    print(f"\n[INFO] Total records collected: {len(all_rows)}")

    if not all_rows:
        print("[ERROR] No records downloaded. Check your internet connection.")
        return

    # Save to CSV
    fieldnames = ["description", "category", "sr_type"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Count per category
    from collections import Counter
    cat_counts = Counter(row["category"] for row in all_rows)
    print(f"\n[SAVED] {OUTPUT_CSV}")
    print(f"\nCategory distribution:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<30} {count}")

    print(f"\n[DONE] Dataset ready. Now run:")
    print(f"  python train_text_classifier.py")


if __name__ == "__main__":
    download()
