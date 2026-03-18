"""
data_loader.py
Reads Excel files and image frames from MinIO,
matches them by timestamp, and prepares datasets for fine-tuning.
Auto-detects timestamp format — handles European, American, ISO, etc.
"""

import os
import io
import re
import yaml
import pandas as pd
from PIL import Image
from minio import Minio
from datetime import datetime


def load_config(config_path="src/config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_minio_client(config):
    """Create and return a MinIO client."""
    return Minio(
        endpoint=config["minio"]["endpoint"],
        access_key=config["minio"]["access_key"],
        secret_key=config["minio"]["secret_key"],
        secure=config["minio"]["secure"],
    )


def download_excel_from_minio(client, bucket, excel_path):
    """Download an Excel file from MinIO and return as DataFrame."""
    response = client.get_object(bucket, excel_path)
    data = response.read()
    response.close()
    response.release_conn()
    df = pd.read_excel(io.BytesIO(data), engine="openpyxl")
    return df


def list_frame_files(client, bucket, folder_prefix):
    """
    List all .jpg files in a MinIO folder.
    Returns a dict: { (timestamp_string, index): full_path }

    Example:
      ("2024-05-21 16-06-29,305", 0) -> "workobject1_frames/2024-05-21 16-06-29,305_normal_0.jpg"
    """
    frames = {}
    objects = client.list_objects(bucket, prefix=folder_prefix + "/", recursive=True)

    for obj in objects:
        filename = os.path.basename(obj.object_name)
        if not filename.lower().endswith(".jpg"):
            continue

        # Extract timestamp, label, and index from filename
        # Format: "2024-05-21 16-06-29,305_normal_0.jpg"
        #     or: "2024-05-21 16-06-29,305_pore_0.jpg"
        match = re.match(r"^(.+?)_(normal|pore)_(\d+)\.jpg$", filename)
        if not match:
            continue

        timestamp_str = match.group(1)   # "2024-05-21 16-06-29,305"
        index = int(match.group(3))      # 0, 1, etc.

        key = (timestamp_str, index)
        frames[key] = obj.object_name

    return frames


def excel_timestamp_to_filename_format(ts):
    """
    Convert Excel timestamp to the format used in image filenames.
    Auto-detects the format — handles datetime objects, pandas Timestamps,
    and strings in any regional format (European, American, ISO).

    Target output: "2024-05-21 16-06-29,305"
    """
    # Step 1: Convert whatever we got into a standard datetime object
    dt = None

    if isinstance(ts, (datetime, pd.Timestamp)):
        dt = ts
    elif isinstance(ts, str):
        # Try multiple string formats that European/American Excel might produce
        formats_to_try = [
            "%Y-%m-%d %H:%M:%S.%f",       # 2024-05-21 16:06:29.305000
            "%Y-%m-%d %H:%M:%S,%f",        # 2024-05-21 16:06:29,305000 (European)
            "%d.%m.%Y %H:%M:%S.%f",        # 21.05.2024 16:06:29.305000 (European date)
            "%d.%m.%Y %H:%M:%S,%f",        # 21.05.2024 16:06:29,305000 (Full European)
            "%d/%m/%Y %H:%M:%S.%f",        # 21/05/2024 16:06:29.305000
            "%d/%m/%Y %H:%M:%S,%f",        # 21/05/2024 16:06:29,305000
            "%Y-%m-%d %H:%M:%S",           # 2024-05-21 16:06:29 (no milliseconds)
            "%d.%m.%Y %H:%M:%S",           # 21.05.2024 16:06:29
        ]
        for fmt in formats_to_try:
            try:
                dt = datetime.strptime(ts.strip(), fmt)
                break
            except ValueError:
                continue
        if dt is None:
            # Last resort: let pandas try to figure it out
            try:
                dt = pd.to_datetime(ts).to_pydatetime()
            except Exception:
                print(f"  WARNING: Could not parse timestamp: '{ts}'")
                return None
    else:
        # numpy datetime64 or other types
        try:
            dt = pd.Timestamp(ts).to_pydatetime()
        except Exception:
            print(f"  WARNING: Unknown timestamp type: {type(ts)} value: '{ts}'")
            return None

    # Step 2: Convert datetime to filename format
    # Target: "2024-05-21 16-06-29,305"
    ms = dt.microsecond // 1000  # get milliseconds (0-999)
    ts_str = dt.strftime("%Y-%m-%d %H-%M-%S") + f",{ms:03d}"

    return ts_str


def load_image_from_minio(client, bucket, image_path):
    """Download an image from MinIO and return as PIL Image."""
    response = client.get_object(bucket, image_path)
    data = response.read()
    response.close()
    response.release_conn()
    return Image.open(io.BytesIO(data)).convert("RGB")


def safe_float(value):
    """
    Convert any value to float, handling European format.
    Handles: 3.2, "3.2", "3,2", "3.200,5", "3,200.5", integers, etc.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        # If both comma and period exist, figure out which is decimal
        if "," in s and "." in s:
            # Whichever comes last is the decimal separator
            if s.rfind(",") > s.rfind("."):
                # European: "3.200,5" -> remove dots, replace comma with dot
                s = s.replace(".", "").replace(",", ".")
            else:
                # American: "3,200.5" -> remove commas
                s = s.replace(",", "")
        elif "," in s and "." not in s:
            # Only comma: European decimal "3,2" -> "3.2"
            s = s.replace(",", ".")
        # else: only dot or no separator, already fine
        return float(s)
    return float(value)


def build_text_prompt(row, input_columns):
    """
    Convert a row of sensor values into a text prompt for the model.
    Converts all values to proper floats (handles European format).

    Example output:
    "Analyze this Wire DED sensor reading and the corresponding melt pool image:
    Layer=45.0, Bead=2.0, Current=85.3, Wire Feed Speed=5.2, ...
    Is this a defect or normal? If defect, estimate the pore diameter."
    """
    sensor_parts = []
    for col in input_columns:
        val = safe_float(row[col])
        sensor_parts.append(f"{col}={val}")
    sensor_text = ", ".join(sensor_parts)

    prompt = (
        f"Analyze this Wire DED sensor reading and the corresponding melt pool image:\n"
        f"{sensor_text}\n"
        f"Is this a defect or normal? If defect, estimate the pore diameter."
    )
    return prompt


def build_target_response(pore_diameter):
    """
    Build the target response the model should learn to produce.

    pore_diameter = 0   -> "NORMAL - no porosity detected."
    pore_diameter > 0   -> "DEFECT - pore detected with diameter: 3.2mm"
    """
    if pore_diameter == 0 or pore_diameter == 0.0:
        return "NORMAL - no porosity detected."
    else:
        return f"DEFECT - pore detected with diameter: {pore_diameter}mm"


def load_workobject_data(client, bucket, excel_path, frame_folder,
                         input_columns, label_column, timestamp_column):
    """
    Load one work object: read Excel, match with images, return paired data.

    Returns list of dicts, each containing:
        image_path  - path to the image in MinIO
        prompt      - text question for the model
        target      - correct answer
        pore_diameter - raw label value
        label       - 0 (normal) or 1 (defect)
    """
    # Step 1: Read Excel
    df = download_excel_from_minio(client, bucket, excel_path)

    # --- Safety: verify columns exist ---
    missing_cols = []
    for col in input_columns + [label_column, timestamp_column]:
        if col not in df.columns:
            missing_cols.append(col)
    if missing_cols:
        print(f"  ERROR: Missing columns: {missing_cols}")
        print(f"  Available columns: {list(df.columns)}")
        return []

    # --- Auto-detect: test timestamp conversion on first row ---
    first_ts = df[timestamp_column].iloc[0]
    test_conversion = excel_timestamp_to_filename_format(first_ts)
    if test_conversion is None:
        print(f"  ERROR: Cannot convert timestamps. First value: '{first_ts}' (type: {type(first_ts)})")
        return []
    print(f"  Timestamp format auto-detected. Sample: '{first_ts}' -> '{test_conversion}'")

    # Step 2: List all frames in the folder
    frames = list_frame_files(client, bucket, frame_folder)
    print(f"  Frames found: {len(frames)}")

    # --- Auto-verify: check if first converted timestamp matches any frame ---
    first_key = (test_conversion, 0)
    if first_key not in frames:
        # Show what we're looking for vs what exists for debugging
        sample_frame_keys = list(frames.keys())[:3]
        print(f"  WARNING: First timestamp '{test_conversion}' not found in frames.")
        print(f"  Looking for key: {first_key}")
        print(f"  Sample frame keys: {sample_frame_keys}")
        print(f"  Continuing anyway — some rows may not have matching images.")

    # Step 3: Match rows with images
    paired_data = []
    skipped = 0

    # Track index per timestamp for duplicate timestamp handling
    timestamp_counts = {}

    for _, row in df.iterrows():
        ts = row[timestamp_column]
        ts_formatted = excel_timestamp_to_filename_format(ts)

        # Skip if timestamp couldn't be parsed
        if ts_formatted is None:
            skipped += 1
            continue

        # Determine index for this timestamp
        if ts_formatted not in timestamp_counts:
            timestamp_counts[ts_formatted] = 0
        else:
            timestamp_counts[ts_formatted] += 1
        current_index = timestamp_counts[ts_formatted]

        # Look up the matching image
        key = (ts_formatted, current_index)
        if key not in frames:
            # No matching image - skip this row
            skipped += 1
            continue

        image_path = frames[key]
        pore_diam = safe_float(row[label_column])

        paired_data.append({
            "image_path": image_path,
            "prompt": build_text_prompt(row, input_columns),
            "target": build_target_response(pore_diam),
            "pore_diameter": pore_diam,
            "label": 0 if pore_diam == 0 else 1,
        })

    if skipped > 0:
        print(f"  Skipped {skipped} rows (no matching image or unparseable timestamp)")

    return paired_data


def load_all_data(config):
    """
    Load all work objects and split into train and test sets.

    Returns:
        train_data: list of dicts (work objects for training)
        test_data:  list of dicts (work objects for testing)
    """
    client = get_minio_client(config)
    bucket = config["minio"]["bucket"]
    input_columns = config["data"]["input_columns"]
    label_column = config["data"]["label_column"]
    timestamp_column = config["data"]["timestamp_column"]

    excel_files = config["data"]["excel_files"]
    frame_folders = config["data"]["frame_folders"]
    train_indices = config["split"]["train_objects"]
    test_indices = config["split"]["test_objects"]

    train_data = []
    test_data = []

    for i, (excel_path, frame_folder) in enumerate(zip(excel_files, frame_folders)):
        print(f"Loading WorkObject {i+1}: {excel_path}...")

        data = load_workobject_data(
            client, bucket, excel_path, frame_folder,
            input_columns, label_column, timestamp_column
        )

        # Count stats
        n_normal = sum(1 for d in data if d["label"] == 0)
        n_defect = sum(1 for d in data if d["label"] == 1)
        print(f"  -> {len(data)} matched rows ({n_normal} normal, {n_defect} defect)")

        if i in train_indices:
            train_data.extend(data)
        elif i in test_indices:
            test_data.extend(data)
        else:
            print(f"  -> Skipped (not in train or test split)")

    print(f"\nTotal train: {len(train_data)} ({sum(1 for d in train_data if d['label']==1)} defects)")
    print(f"Total test:  {len(test_data)} ({sum(1 for d in test_data if d['label']==1)} defects)")

    return train_data, test_data


# ---- Quick test ----
if __name__ == "__main__":
    config = load_config()
    train_data, test_data = load_all_data(config)

    # Print a sample
    if train_data:
        sample = train_data[0]
        print(f"\nSample prompt:\n{sample['prompt']}")
        print(f"\nSample target:\n{sample['target']}")
        print(f"Image: {sample['image_path']}")
