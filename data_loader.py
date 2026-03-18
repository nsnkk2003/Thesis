"""
data_loader.py
Reads Excel files and image frames from MinIO,
matches them by timestamp, and prepares datasets for fine-tuning.
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

    Excel:    "2024-05-21 16:06:29.305"
    Filename: "2024-05-21 16-06-29,305"

    Rule: colons -> hyphens, period before milliseconds -> comma
    """
    if isinstance(ts, datetime):
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S.") + f"{ts.microsecond // 1000:03d}"
    elif isinstance(ts, pd.Timestamp):
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S.") + f"{ts.microsecond // 1000:03d}"
    else:
        ts_str = str(ts)

    # Replace colons with hyphens in the time part
    ts_str = ts_str.replace(":", "-")

    # Replace the last period (before milliseconds) with comma
    last_dot = ts_str.rfind(".")
    if last_dot != -1:
        ts_str = ts_str[:last_dot] + "," + ts_str[last_dot + 1:]

    return ts_str


def load_image_from_minio(client, bucket, image_path):
    """Download an image from MinIO and return as PIL Image."""
    response = client.get_object(bucket, image_path)
    data = response.read()
    response.close()
    response.release_conn()
    return Image.open(io.BytesIO(data)).convert("RGB")


def build_text_prompt(row, input_columns):
    """
    Convert a row of sensor values into a text prompt for the model.

    Example output:
    "Analyze this Wire DED sensor reading and the corresponding melt pool image:
    Layer=45, Bead=2, Current=85.3, Wire Feed Speed=5.2, ...
    Is this a defect or normal? If defect, estimate the pore diameter."
    """
    sensor_text = ", ".join(
        f"{col}={row[col]}" for col in input_columns
    )

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

    # Step 2: List all frames in the folder
    frames = list_frame_files(client, bucket, frame_folder)

    # Step 3: Match rows with images
    paired_data = []

    # Track index per timestamp for duplicate timestamp handling
    timestamp_counts = {}

    for _, row in df.iterrows():
        ts = row[timestamp_column]
        ts_formatted = excel_timestamp_to_filename_format(ts)

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
            continue

        image_path = frames[key]
        pore_diam = float(row[label_column])

        paired_data.append({
            "image_path": image_path,
            "prompt": build_text_prompt(row, input_columns),
            "target": build_target_response(pore_diam),
            "pore_diameter": pore_diam,
            "label": 0 if pore_diam == 0 else 1,
        })

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
