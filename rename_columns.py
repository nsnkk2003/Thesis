"""
rename_columns.py
Renames columns in all Excel files to match the standard names in config.yaml.
Saves fixed files in a new folder called 'Fixed_Data'.

Usage: python rename_columns.py "C:\Users\yy57328\Downloads\Final_Data_100326"
"""

import sys
import os
import pandas as pd

# Mapping: old name in Excel → new name we want in config
RENAME_MAP = {
    "AI_ProcadaCurrent": "Current",
    "Wire_Speed": "Wire Feed Speed",
    "TCP_Speed": "Throughput_Speed",
    "Laser_Power": "Laser Output Power",
    "Pyro_1_Low": "Pyrometer1_Low",
    "Pyro_2_Mid": "Pyrometer2_Mid",
    "Pyro_3_High": "Pyrometer3_High",
    "AI_ProcadaVoltage": "AI_LaserVoltage",
    "Robot_Wire_Speed": "Robot_DepositionWire_Speed",
    "Actual_Conductance": "Conductance",
    "Actual_Power_HotWire": "Power_Wire",
}

# Also rename file: Annom_WorkObject1.xlsx → WorkObject1.xlsx
def clean_filename(filename):
    return filename.replace("Annom_", "")


def rename_columns(folder_path):
    # Create output folder
    output_folder = os.path.join(folder_path, "Fixed_Data")
    os.makedirs(output_folder, exist_ok=True)

    excel_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".xlsx")])

    for filename in excel_files:
        filepath = os.path.join(folder_path, filename)
        print(f"\nProcessing: {filename}")

        # Read full Excel
        df = pd.read_excel(filepath, engine="openpyxl")
        print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")

        # Rename columns
        renamed = {}
        for old_name, new_name in RENAME_MAP.items():
            if old_name in df.columns:
                renamed[old_name] = new_name
                print(f"  Renamed: '{old_name}' → '{new_name}'")
            else:
                print(f"  WARNING: '{old_name}' not found in this file")

        df = df.rename(columns=renamed)

        # Clean filename (remove "Annom_" prefix)
        new_filename = clean_filename(filename)
        output_path = os.path.join(output_folder, new_filename)

        # Save
        df.to_excel(output_path, index=False, engine="openpyxl")
        print(f"  Saved: {output_path}")

    # Verify one file
    print("\n" + "=" * 60)
    print("VERIFICATION: Checking first fixed file")
    print("=" * 60)
    first_fixed = sorted([f for f in os.listdir(output_folder) if f.endswith(".xlsx")])[0]
    df_check = pd.read_excel(os.path.join(output_folder, first_fixed), engine="openpyxl", nrows=1)
    print(f"\n{first_fixed} columns:")
    print(f"  {list(df_check.columns)}")

    # Check all standard columns exist
    standard = ["Time", "Layer", "Bead", "Current", "Wire Feed Speed", "Throughput_Speed",
                "Laser Output Power", "Pyrometer1_Low", "Pyrometer2_Mid", "Pyrometer3_High",
                "AI_LaserVoltage", "Robot_DepositionWire_Speed", "Conductance", "Power_Wire",
                "PoreInfo", "pore_diameter"]
    missing = [c for c in standard if c not in df_check.columns]
    if missing:
        print(f"\n  STILL MISSING: {missing}")
    else:
        print(f"\n  ✓ All standard columns present!")

    print(f"\nFixed files saved in: {output_folder}")
    print(f"Upload the contents of this folder to MinIO (am-data bucket)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rename_columns.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    if not os.path.exists(folder_path):
        print(f"ERROR: Folder not found: {folder_path}")
        sys.exit(1)

    rename_columns(folder_path)
