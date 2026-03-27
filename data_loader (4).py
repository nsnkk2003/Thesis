"""
data_loader.py
Reads Excel files and image frames from MinIO,
matches them by timestamp, and prepares datasets for fine-tuning.

Excel timestamp format : 2024-05-20 08:22:58.996000  (pandas Timestamp)
Image filename format  : 2024-05-20 08-22-58,996_normal_0.jpg
"""

import os
import io
import re
import yaml
import pandas as pd
from PIL import Image
from minio import Minio


def load_config(config_path="src/config.yaml"):
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
    return Minio(
        endpoint=os.environ.get("MINIO_ENDPOINT",   config["minio"]["endpoint"]),
        access_key=os.environ.get("MINIO_ACCESS_KEY", config["minio"]["access_key"]),
        secret_key=os.environ.get("MINIO_SECRET_KEY", config["minio"]["secret_key"]),
        secure=config["minio"]["secure"],
    )


def download_excel_from_minio(client, bucket, excel_path):
    response = client.get_object(bucket, excel_path)
    data = response.read()
    response.close()
    response.release_conn()
    return pd.read_excel(io.BytesIO(data), engine="openpyxl")


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
        ts = ts.replace(",", ".")   # normalise European comma → dot for pd.Timestamp
    dt = pd.Timestamp(ts)
    ms = dt.microsecond // 1000
    return dt.strftime("%Y-%m-%d %H-%M-%S") + f",{ms:03d}"


def list_frame_files(client, bucket, folder_prefix):
    """
    List all .jpg frames in a MinIO folder.
    Returns dict: { timestamp_string: full_object_path }

    Filename format: 2024-05-20 08-22-58,996_normal_0.jpg
                 or: 2024-05-20 08-22-58,996_defect_0.jpg
                 or: 2024-05-20 08-22-58,996_pore_0.jpg
    """
    frames = {}
    for obj in client.list_objects(bucket, prefix=folder_prefix + "/", recursive=True):
        filename = os.path.basename(obj.object_name)
        if not filename.lower().endswith(".jpg"):
            continue
        match = re.match(r"^(.+?)_(normal|pore)_\d+\.jpg$", filename)
        if not match:
            continue
        frames[match.group(1)] = obj.object_name
    return frames


def load_image_from_minio(client, bucket, image_path):
    response = client.get_object(bucket, image_path)
    data = response.read()
    response.close()
    response.release_conn()
    return Image.open(io.BytesIO(data)).convert("RGB")


def safe_float(value):
    """
    Convert sensor value to float, handling European decimal format.
    Handles: 3.2, "3,2", "3.200,5", "3,200.5", integers, etc.
    """
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


def build_text_prompt(row, input_columns):
    sensor_text = ", ".join(f"{col}={safe_float(row[col])}" for col in input_columns)
    return (
        f"Analyze this Wire DED sensor reading and the corresponding melt pool image:\n"
        f"{sensor_text}\n"
        f"Is this a defect or normal? If defect, estimate the pore diameter."
    )


def build_target_response(pore_diameter):
    if pore_diameter == 0:
        return "NORMAL - no porosity detected."
    return f"DEFECT - pore detected with diameter: {pore_diameter}mm"


def load_workobject_data(client, bucket, excel_path, frame_folder,
                         input_columns, label_column, timestamp_column):
    df = download_excel_from_minio(client, bucket, excel_path)

    missing = [c for c in input_columns + [label_column, timestamp_column] if c not in df.columns]
    if missing:
        print(f"  ERROR: Missing columns: {missing}")
        print(f"  Available: {list(df.columns)}")
        return []

    frames = list_frame_files(client, bucket, frame_folder)
    print(f"  Frames found: {len(frames)}")

    paired_data = []
    skipped = 0
    seen_timestamps = set()

    for _, row in df.iterrows():
        ts = timestamp_to_filename(row[timestamp_column])

        if ts in seen_timestamps:
            skipped += 1
            continue
        seen_timestamps.add(ts)

        if ts not in frames:
            skipped += 1
            continue

        pore_diam = safe_float(row[label_column])
        paired_data.append({
            "image_path":    frames[ts],
            "prompt":        build_text_prompt(row, input_columns),
            "target":        build_target_response(pore_diam),
            "pore_diameter": pore_diam,
            "label":         0 if pore_diam == 0 else 1,
        })

    print(f"  Skipped {skipped} rows (duplicate timestamps or no matching frame)")
    return paired_data


def load_all_data(config):
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


if __name__ == "__main__":
    config = load_config()
    train_data, test_data = load_all_data(config)
    if train_data:
        sample = train_data[0]
        print(f"\nSample prompt:\n{sample['prompt']}")
        print(f"Sample target: {sample['target']}")
        print(f"Image: {sample['image_path']}")
