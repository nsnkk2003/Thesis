"""
fix_columns.py
Run this locally on your laptop.
1. Shows what columns each Excel file has
2. Renames columns to match the standard names
3. Saves fixed files

Usage: python fix_columns.py "C:\Users\yy57328\Downloads\Final_Data_100326"
"""

import sys
import os
import pandas as pd

# Standard column names we need (from config.yaml)
STANDARD_COLUMNS = [
    "Time",
    "Layer",
    "Bead",
    "Current",
    "Wire Feed Speed",
    "Throughput_Speed",
    "Laser Output Power",
    "Pyrometer1_Low",
    "Pyrometer2_Mid",
    "Pyrometer3_High",
    "AI_LaserVoltage",
    "Robot_DepositionWire_Speed",
    "Conductance",
    "Power_Wire",
    "PoreInfo",
    "pore_diameter",
]


def check_columns(folder_path):
    """Step 1: Show what columns each Excel file has."""
    print("=" * 60)
    print("STEP 1: Checking columns in each Excel file")
    print("=" * 60)

    excel_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".xlsx")])

    all_columns = {}
    for filename in excel_files:
        filepath = os.path.join(folder_path, filename)
        df = pd.read_excel(filepath, engine="openpyxl", nrows=3)  # read only 3 rows for speed
        cols = list(df.columns)
        all_columns[filename] = cols
        print(f"\n{filename}:")
        print(f"  Columns ({len(cols)}): {cols}")

    # Find columns that differ from file to file
    print("\n" + "=" * 60)
    print("STEP 2: Comparing columns across files")
    print("=" * 60)

    # Check which standard columns are missing in each file
    for filename, cols in all_columns.items():
        missing = [c for c in STANDARD_COLUMNS if c not in cols]
        extra = [c for c in cols if c not in STANDARD_COLUMNS]
        if missing:
            print(f"\n{filename}:")
            print(f"  MISSING: {missing}")
            if extra:
                print(f"  EXTRA (might be renamed versions): {extra}")
        else:
            print(f"\n{filename}: ✓ All standard columns present")

    # Also check first timestamp format
    print("\n" + "=" * 60)
    print("STEP 3: Checking timestamp format")
    print("=" * 60)
    for filename in excel_files:
        filepath = os.path.join(folder_path, filename)
        df = pd.read_excel(filepath, engine="openpyxl", nrows=3)
        if "Time" in df.columns:
            ts = df["Time"].iloc[0]
            print(f"\n{filename}:")
            print(f"  First timestamp: '{ts}' (type: {type(ts).__name__})")
        else:
            print(f"\n{filename}: 'Time' column not found!")
            # Check for similar column names
            time_like = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
            if time_like:
                print(f"  Possible time columns: {time_like}")

    return all_columns


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_columns.py <folder_path>")
        print('Example: python fix_columns.py "C:\\Users\\yy57328\\Downloads\\Final_Data_100326"')
        sys.exit(1)

    folder_path = sys.argv[1]

    if not os.path.exists(folder_path):
        print(f"ERROR: Folder not found: {folder_path}")
        sys.exit(1)

    all_columns = check_columns(folder_path)
