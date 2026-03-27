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
    """
    Load configuration from YAML file.
    Falls back to config.template.yaml if config.yaml doesn't exist
    (on the cluster, credentials come from environment variables instead).
    """
    if not os.path.exists(config_path):
        template_path = config_path.replace("config.yaml", "config.template.yaml")
        if os.path.exists(template_path):
            print(f"  config.yaml not found, using {template_path}")
            print(f"  Credentials will be read from environment variables.")
            config_path = template_path
        else:
            raise FileNotFoundError(f"Neither {config_path} nor {template_path} found!")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_minio_client(config):
    """
    Create and return a MinIO client.
    Reads credentials from environment variables first (for Kubernetes),
    falls back to config.yaml values (for local testing).
    """
    endpoint   = os.environ.get("MINIO_ENDPOINT",   config["minio"]["endpoint"])
    access_key = os.environ.get("MINIO_ACCESS_KEY", config["minio"]["access_key"])
    secret_key = os.environ.get("MINIO_SECRET_KEY", config["minio"]["secret_key"])

    return Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=config["minio"]["secure"],
    )


def download_excel_from_minio(client, bucket, excel_path):
    """Download an Excel file from MinIO and return as DataFrame."""
    response = client.get_object(bucket, excel_path)
    data = response.read()
    response.close()
    response.release_conn()
    return pd.read_excel(io.BytesIO(data), engine="openpyxl")


def list_frame_files(client, bucket, folder_prefix):
    """
    List all .jpg files in a MinIO folder.
    Returns a dict: { timestamp_string: full_object_path }

    One frame per timestamp — the camera produces one image per moment.
    Filename format: "2024-05-21 16-06-29,305_normal_0.jpg"
                 or: "2024-05-21 16-06-29,305_defect_0.jpg"
                 or: "2024-05-21 16-06-29,305_pore_0.jpg"
    """
    frames = {}
    objects = client.list_objects(bucket, prefix=folder_prefix + "/", recursive=True)

    for obj in objects:
        filename = os.path.basename(obj.object_name)
        if not filename.lower().endswith(".jpg"):
            continue

        # FIX: regex covers normal, pore, AND defect labels
        match = re.match(r"^(.+?)_(normal|pore|defect)_(\d+)\.jpg$", filename)
        if not match:
            continue

        timestamp_str = match.group(1)  # "2024-05-21 16-06-29,305"
        frames[timestamp_str] = obj.object_name

    return frames


def excel_timestamp_to_filename_format(ts):
    """
    Convert Excel timestamp to the format used in image filenames.
    Auto-detects the format — handles datetime objects, pandas Timestamps,
    and strings in any regional format (European, American, ISO).

    Target output: "2024-05-21 16-06-29,305"
    """
    dt = None

    if isinstance(ts, (datetime, pd.Timestamp)):
        dt = ts
    elif isinstance(ts, str):
        formats_to_try = [
            "%Y-%m-%d %H:%M:%S.%f",    # 2024-05-21 16:06:29.305000
            "%Y-%m-%d %H:%M:%S,%f",    # 2024-05-21 16:06:29,305000 (European)
            "%d.%m.%Y %H:%M:%S.%f",   # 21.05.2024 16:06:29.305000
            "%d.%m.%Y %H:%M:%S,%f",   # 21.05.2024 16:06:29,305000
            "%d/%m/%Y %H:%M:%S.%f",   # 21/05/2024 16:06:29.305000
            "%d/%m/%Y %H:%M:%S,%f",   # 21/05/2024 16:06:29,305000
            "%Y-%m-%d %H:%M:%S",      # 2024-05-21 16:06:29 (no ms)
            "%d.%m.%Y %H:%M:%S",      # 21.05.2024 16:06:29
        ]
        for fmt in formats_to_try:
            try:
                dt = datetime.strptime(ts.strip(), fmt)
                break
            except ValueError:
                continue
        if dt is None:
            try:
                dt = pd.to_datetime(ts).to_pydatetime()
            except Exception:
                print(f"  WARNING: Could not parse timestamp: '{ts}'")
                return None
    else:
        try:
            dt = pd.Timestamp(ts).to_pydatetime()
        except Exception:
            print(f"  WARNING: Unknown timestamp type: {type(ts)} value: '{ts}'")
            return None

    ms = dt.microsecond // 1000
    return dt.strftime("%Y-%m-%d %H-%M-%S") + f",{ms:03d}"


def load_image_from_minio(client, bucket, image_path):
    """Download an image from MinIO and return as PIL Image."""
    response = client.get_object(bucket, image_path)
    data = response.read()
    response.close()
    response.release_conn()
    return Image.open(io.BytesIO(data)).convert("RGB")


def safe_float(value):
    """
    Convert any value to float, handling European decimal format.
    Handles: 3.2, "3.2", "3,2", "3.200,5", "3,200.5", integers, etc.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if "," in s and "." in s:
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".", "").replace(",", ".")  # European: "3.200,5"
            else:
                s = s.replace(",", "")                    # American: "3,200.5"
        elif "," in s:
            s = s.replace(",", ".")                       # European decimal: "3,2"
        return float(s)
    return float(value)


def build_text_prompt(row, input_columns):
    """
    Convert a row of sensor values into a text prompt for the model.

    Example output:
      "Analyze this Wire DED sensor reading and the corresponding melt pool image:
       Layer=45.0, Bead=2.0, Current=85.3, ...
       Is this a defect or normal? If defect, estimate the pore diameter."
    """
    sensor_text = ", ".join(f"{col}={safe_float(row[col])}" for col in input_columns)
    return (
        f"Analyze this Wire DED sensor reading and the corresponding melt pool image:\n"
        f"{sensor_text}\n"
        f"Is this a defect or normal? If defect, estimate the pore diameter."
    )


def build_target_response(pore_diameter):
    """
    Build the target response the model should learn to produce.

    pore_diameter = 0  -> "NORMAL - no porosity detected."
    pore_diameter > 0  -> "DEFECT - pore detected with diameter: 3.2mm"
    """
    if pore_diameter == 0:
        return "NORMAL - no porosity detected."
    return f"DEFECT - pore detected with diameter: {pore_diameter}mm"


def load_workobject_data(client, bucket, excel_path, frame_folder,
                         input_columns, label_column, timestamp_column):
    """
    Load one WorkObject: read Excel, match rows to images by timestamp, return paired data.

    The sensor logs many rows per second; the camera captures one frame per timestamp.
    Only the first Excel row for each unique timestamp is matched — duplicates are skipped.

    Returns list of dicts, each containing:
        image_path    - MinIO object path to the image
        prompt        - text question for the model
        target        - correct answer string
        pore_diameter - raw label value
        label         - 0 (normal) or 1 (defect)
    """
    df = download_excel_from_minio(client, bucket, excel_path)

    # Verify required columns exist
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

    # Load all frames from MinIO (keyed by timestamp string)
    frames = list_frame_files(client, bucket, frame_folder)
    print(f"  Frames found: {len(frames)}")

    # Warn early if even the first timestamp doesn't match
    if test_conversion not in frames:
        sample_keys = list(frames.keys())[:3]
        print(f"  WARNING: First timestamp '{test_conversion}' not found in frames.")
        print(f"  Sample frame keys: {sample_keys}")

    # Match Excel rows to frames — one match per unique timestamp
    paired_data = []
    skipped = 0
    seen_timestamps = set()

    for _, row in df.iterrows():
        ts_formatted = excel_timestamp_to_filename_format(row[timestamp_column])

        if ts_formatted is None:
            skipped += 1
            continue

        # FIX: skip duplicate timestamps — sensor logs faster than camera captures
        if ts_formatted in seen_timestamps:
            skipped += 1
            continue
        seen_timestamps.add(ts_formatted)

        if ts_formatted not in frames:
            skipped += 1
            continue

        pore_diam = safe_float(row[label_column])
        paired_data.append({
            "image_path":   frames[ts_formatted],
            "prompt":       build_text_prompt(row, input_columns),
            "target":       build_target_response(pore_diam),
            "pore_diameter": pore_diam,
            "label":        0 if pore_diam == 0 else 1,
        })

    print(f"  Skipped {skipped} rows (duplicate timestamps or no matching frame)")
    return paired_data


def load_all_data(config):
    """
    Load all WorkObjects and split into train and test sets.

    Returns:
        train_data: list of sample dicts
        test_data:  list of sample dicts
    """
    client           = get_minio_client(config)
    bucket           = config["minio"]["bucket"]
    input_columns    = config["data"]["input_columns"]
    label_column     = config["data"]["label_column"]
    timestamp_column = config["data"]["timestamp_column"]
    excel_files      = config["data"]["excel_files"]
    frame_folders    = config["data"]["frame_folders"]
    train_indices    = config["split"]["train_objects"]
    test_indices     = config["split"]["test_objects"]

    train_data, test_data = [], []

    for i, (excel_path, frame_folder) in enumerate(zip(excel_files, frame_folders)):
        print(f"Loading WorkObject {i+1}: {excel_path}...")
        data = load_workobject_data(
            client, bucket, excel_path, frame_folder,
            input_columns, label_column, timestamp_column
        )
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

    if train_data:
        sample = train_data[0]
        print(f"\nSample prompt:\n{sample['prompt']}")
        print(f"\nSample target:\n{sample['target']}")
        print(f"Image: {sample['image_path']}")
