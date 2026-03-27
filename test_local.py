"""
test_local.py
Local verification of data_loader logic — no MinIO required.

Replaces MinIO I/O with local filesystem reads.
All matching, timestamp conversion, and data-building logic
is imported directly from data_loader.py so you test the real code.

Expected folder structure (mirrors MinIO layout):
    local_data/
        WorkObject1.xlsx
        WorkObject2.xlsx
        ...
        workobject1_frames/
            2024-05-20 08-22-58,996_normal_0.jpg
            2024-05-20 08-23-01,012_normal_0.jpg
            ...
        workobject2_frames/
            ...

Usage:
    python test_local.py
    python test_local.py --data-dir /path/to/local_data
"""

import os
import re
import sys
import argparse
from PIL import Image

# --- Import pure logic from data_loader.py (same file you'll deploy) ---
from data_loader import (
    excel_timestamp_to_filename_format,
    safe_float,
    build_text_prompt,
    build_target_response,
)

import pandas as pd


# ---------------------------------------------------------------------------
# Local replacements for MinIO I/O
# ---------------------------------------------------------------------------

def local_download_excel(excel_path):
    """Read an Excel file from local disk."""
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    return pd.read_excel(excel_path, engine="openpyxl")


def local_list_frame_files(folder_path):
    """
    List all .jpg files in a local folder.
    Returns a dict: { timestamp_string: full_file_path }

    Mirrors list_frame_files() in data_loader.py exactly.
    """
    frames = {}
    if not os.path.exists(folder_path):
        print(f"  WARNING: Frame folder not found: {folder_path}")
        return frames

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".jpg"):
            continue

        match = re.match(r"^(.+?)_(normal|pore|defect)_(\d+)\.jpg$", filename)
        if not match:
            continue

        timestamp_str = match.group(1)
        frames[timestamp_str] = os.path.join(folder_path, filename)

    return frames


def local_load_image(image_path):
    """Load an image from local disk and return as PIL Image."""
    return Image.open(image_path).convert("RGB")


# ---------------------------------------------------------------------------
# Core matching logic (mirrors load_workobject_data in data_loader.py)
# ---------------------------------------------------------------------------

def load_workobject_local(excel_path, frame_folder, input_columns, label_column, timestamp_column):
    """
    Local version of load_workobject_data.
    Uses local files instead of MinIO — logic is identical.
    """
    df = local_download_excel(excel_path)

    # Verify required columns
    missing = [c for c in input_columns + [label_column, timestamp_column] if c not in df.columns]
    if missing:
        print(f"  ERROR: Missing columns: {missing}")
        print(f"  Available: {list(df.columns)}")
        return []

    # Validate timestamp conversion on first row
    first_ts = df[timestamp_column].iloc[0]
    test_conversion = excel_timestamp_to_filename_format(first_ts)
    if test_conversion is None:
        print(f"  ERROR: Cannot convert timestamps. First value: '{first_ts}'")
        return []
    print(f"  Timestamp format auto-detected. Sample: '{first_ts}' -> '{test_conversion}'")

    # Load frames
    frames = local_list_frame_files(frame_folder)
    print(f"  Frames found: {len(frames)}")

    if test_conversion not in frames:
        sample_keys = list(frames.keys())[:3]
        print(f"  WARNING: First timestamp '{test_conversion}' not found in frames.")
        print(f"  Sample frame keys: {sample_keys}")

    # Match rows to frames
    paired_data = []
    skipped = 0
    seen_timestamps = set()

    for _, row in df.iterrows():
        ts_formatted = excel_timestamp_to_filename_format(row[timestamp_column])

        if ts_formatted is None:
            skipped += 1
            continue

        if ts_formatted in seen_timestamps:
            skipped += 1
            continue
        seen_timestamps.add(ts_formatted)

        if ts_formatted not in frames:
            skipped += 1
            continue

        pore_diam = safe_float(row[label_column])
        paired_data.append({
            "image_path":    frames[ts_formatted],
            "prompt":        build_text_prompt(row, input_columns),
            "target":        build_target_response(pore_diam),
            "pore_diameter": pore_diam,
            "label":         0 if pore_diam == 0 else 1,
        })

    print(f"  Skipped {skipped} rows (duplicate timestamps or no matching frame)")
    return paired_data


# ---------------------------------------------------------------------------
# Config — edit these to match your local folder layout
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = "./local_data"

# These must match the column names in your Excel files
INPUT_COLUMNS    = ["Layer", "Bead", "Current", "Wire Feed Speed", "Travel Speed"]
LABEL_COLUMN     = "Pore Diameter"
TIMESTAMP_COLUMN = "Timestamp"

# WorkObjects to test locally — add/remove as needed
WORKOBJECTS = [
    {
        "name":         "WorkObject1",
        "excel":        "WorkObject1.xlsx",
        "frame_folder": "workobject1_frames",
    },
    {
        "name":         "WorkObject2",
        "excel":        "WorkObject2.xlsx",
        "frame_folder": "workobject2_frames",
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(data_dir):
    print(f"Data directory: {os.path.abspath(data_dir)}")
    print("=" * 60)

    all_data = []

    for wo in WORKOBJECTS:
        excel_path    = os.path.join(data_dir, wo["excel"])
        frame_folder  = os.path.join(data_dir, wo["frame_folder"])

        print(f"\nLoading {wo['name']}...")
        print(f"  Excel : {excel_path}")
        print(f"  Frames: {frame_folder}")

        if not os.path.exists(excel_path):
            print(f"  SKIPPED — Excel file not found.")
            continue

        data = load_workobject_local(
            excel_path, frame_folder,
            INPUT_COLUMNS, LABEL_COLUMN, TIMESTAMP_COLUMN
        )

        n_normal = sum(1 for d in data if d["label"] == 0)
        n_defect = sum(1 for d in data if d["label"] == 1)
        print(f"  -> {len(data)} matched rows ({n_normal} normal, {n_defect} defect)")
        all_data.extend(data)

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"TOTAL MATCHED: {len(all_data)}")
    print(f"  Normal: {sum(1 for d in all_data if d['label'] == 0)}")
    print(f"  Defect: {sum(1 for d in all_data if d['label'] == 1)}")

    # --- Sample output ---
    if all_data:
        print("\n--- Sample (first matched row) ---")
        sample = all_data[0]
        print(f"Image : {sample['image_path']}")
        print(f"Prompt:\n{sample['prompt']}")
        print(f"Target: {sample['target']}")

        # Verify image actually opens
        try:
            img = local_load_image(sample["image_path"])
            print(f"Image size: {img.size} mode: {img.mode}  ✓")
        except Exception as e:
            print(f"  WARNING: Could not open image: {e}")

    # --- Defect samples ---
    defects = [d for d in all_data if d["label"] == 1]
    if defects:
        print(f"\n--- Defect samples ({len(defects)} total) ---")
        for d in defects[:5]:
            print(f"  {d['image_path']}  ->  {d['target']}")
    else:
        print("\n  No defect samples found — check label column values.")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local test for data_loader.py")
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Path to local data folder (default: {DEFAULT_DATA_DIR})"
    )
    args = parser.parse_args()
    run(args.data_dir)
