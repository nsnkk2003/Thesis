"""
test_local.py
Local verification of the data matching logic — no MinIO, no imports from other files.

Expected folder structure:
    local_data/
        WorkObject1.xlsx
        workobject1_frames/
            2024-05-20 08-22-58,996_normal_0.jpg
            ...

Usage:
    python test_local.py
    python test_local.py --data-dir /path/to/local_data
"""

import os
import re
import argparse
import pandas as pd
from PIL import Image


# ---------------------------------------------------------------------------
# Edit these to match your Excel column names exactly
# ---------------------------------------------------------------------------
INPUT_COLUMNS    = ["Layer", "Bead", "Current", "Wire Feed Speed", "Travel Speed"]
LABEL_COLUMN     = "Pore Diameter"
TIMESTAMP_COLUMN = "Timestamp"

WORKOBJECTS = [
    {"name": "WorkObject1", "excel": "WorkObject1.xlsx", "frame_folder": "workobject1_frames"},
    {"name": "WorkObject2", "excel": "WorkObject2.xlsx", "frame_folder": "workobject2_frames"},
]
# ---------------------------------------------------------------------------


def timestamp_to_filename(ts):
    """
    Convert Excel timestamp to image filename prefix.

    Input  (pandas Timestamp or string): 2024-05-20 08:22:58.996000
                                      or 2024-05-20 08:22:58,996000  (European Excel)
    Output (string)                   : 2024-05-20 08-22-58,996

    The ms separator in the source may be . or , — both are handled.
    The output always uses , before ms to match the image filename format.
    """
    if isinstance(ts, str):
        ts = ts.replace(",", ".")  # normalise European comma → dot for pd.Timestamp
    dt = pd.Timestamp(ts)
    ms = dt.microsecond // 1000
    return dt.strftime("%Y-%m-%d %H-%M-%S") + f",{ms:03d}"


def list_frame_files(folder_path):
    """
    List all .jpg frames in a local folder.
    Returns dict: { timestamp_string: full_file_path }

    Filename format: 2024-05-20 08-22-58,996_normal_0.jpg
                 or: 2024-05-20 08-22-58,996_defect_0.jpg
                 or: 2024-05-20 08-22-58,996_pore_0.jpg
    """
    frames = {}
    if not os.path.exists(folder_path):
        print(f"  WARNING: Frame folder not found: {folder_path}")
        return frames
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".jpg"):
            continue
        match = re.match(r"^(.+?)_(normal|pore|defect)_\d+\.jpg$", filename)
        if not match:
            continue
        frames[match.group(1)] = os.path.join(folder_path, filename)
    return frames


def safe_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")  # European: "3.200,5"
        else:
            s = s.replace(",", "")                    # American: "3,200.5"
    elif "," in s:
        s = s.replace(",", ".")                       # European decimal: "3,2"
    return float(s)


def build_text_prompt(row):
    sensor_text = ", ".join(f"{col}={safe_float(row[col])}" for col in INPUT_COLUMNS)
    return (
        f"Analyze this Wire DED sensor reading and the corresponding melt pool image:\n"
        f"{sensor_text}\n"
        f"Is this a defect or normal? If defect, estimate the pore diameter."
    )


def build_target_response(pore_diameter):
    if pore_diameter == 0:
        return "NORMAL - no porosity detected."
    return f"DEFECT - pore detected with diameter: {pore_diameter}mm"


def load_workobject_local(excel_path, frame_folder):
    df = pd.read_excel(excel_path, engine="openpyxl")

    missing = [c for c in INPUT_COLUMNS + [LABEL_COLUMN, TIMESTAMP_COLUMN] if c not in df.columns]
    if missing:
        print(f"  ERROR: Missing columns: {missing}")
        print(f"  Available: {list(df.columns)}")
        return []

    frames = list_frame_files(frame_folder)
    print(f"  Frames found: {len(frames)}")

    paired_data = []
    skipped = 0
    seen_timestamps = set()

    for _, row in df.iterrows():
        ts = timestamp_to_filename(row[TIMESTAMP_COLUMN])

        if ts in seen_timestamps:
            skipped += 1
            continue
        seen_timestamps.add(ts)

        if ts not in frames:
            skipped += 1
            continue

        pore_diam = safe_float(row[LABEL_COLUMN])
        paired_data.append({
            "image_path":    frames[ts],
            "prompt":        build_text_prompt(row),
            "target":        build_target_response(pore_diam),
            "pore_diameter": pore_diam,
            "label":         0 if pore_diam == 0 else 1,
        })

    print(f"  Skipped {skipped} rows (duplicate timestamps or no matching frame)")
    return paired_data


def run(data_dir):
    print(f"Data directory: {os.path.abspath(data_dir)}")
    print("=" * 60)

    all_data = []

    for wo in WORKOBJECTS:
        excel_path   = os.path.join(data_dir, wo["excel"])
        frame_folder = os.path.join(data_dir, wo["frame_folder"])

        print(f"\nLoading {wo['name']}...")

        if not os.path.exists(excel_path):
            print(f"  SKIPPED — Excel file not found: {excel_path}")
            continue

        data = load_workobject_local(excel_path, frame_folder)
        n_normal = sum(1 for d in data if d["label"] == 0)
        n_defect = sum(1 for d in data if d["label"] == 1)
        print(f"  -> {len(data)} matched rows ({n_normal} normal, {n_defect} defect)")
        all_data.extend(data)

    print("\n" + "=" * 60)
    print(f"TOTAL MATCHED : {len(all_data)}")
    print(f"  Normal : {sum(1 for d in all_data if d['label'] == 0)}")
    print(f"  Defect : {sum(1 for d in all_data if d['label'] == 1)}")

    if all_data:
        sample = all_data[0]
        print(f"\n--- First matched sample ---")
        print(f"Image : {sample['image_path']}")
        print(f"Prompt:\n{sample['prompt']}")
        print(f"Target: {sample['target']}")
        try:
            img = Image.open(sample["image_path"]).convert("RGB")
            print(f"Image size: {img.size}, mode: {img.mode}  ✓")
        except Exception as e:
            print(f"  WARNING: Could not open image: {e}")

    defects = [d for d in all_data if d["label"] == 1]
    if defects:
        print(f"\n--- Defect samples ({len(defects)} total) ---")
        for d in defects[:5]:
            print(f"  {os.path.basename(d['image_path'])}  ->  {d['target']}")
    else:
        print("\n  No defect samples — check your label column values.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./local_data",
                        help="Path to local data folder (default: ./local_data)")
    args = parser.parse_args()
    run(args.data_dir)
