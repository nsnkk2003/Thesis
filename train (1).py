"""
train.py
Fine-tunes LLaVA-NeXT (Mistral-7B) on Wire DED defect detection data
using LoRA + 4-bit quantization + local image cache + on-the-fly augmentation + weighted loss.
Auto-uploads checkpoints to MinIO and auto-resumes from latest checkpoint.
"""

import os
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torchvision.transforms as T

from data_loader import load_config, load_all_data, get_minio_client


# ============================================
# 1. MINIO -> LOCAL CACHE HELPERS
# ============================================
def safe_local_path(cache_dir: str, object_name: str) -> str:
    """
    Map a MinIO object path to a safe local file path.
    Keeps the same nested structure under cache_dir.
    """
    object_name = object_name.lstrip("/").replace("\\", "/")
    return os.path.join(cache_dir, object_name)


def download_images_to_cache(data, minio_client, bucket: str, cache_dir: str, force_download: bool = False):
    """
    Download all images referenced by `data` into a local cache directory.

    Args:
        data: list of dicts containing image_path
        minio_client: MinIO client
        bucket: MinIO bucket name
        cache_dir: local cache directory
        force_download: if True, re-download even if file exists
    """
    os.makedirs(cache_dir, exist_ok=True)

    unique_paths = sorted({item["image_path"] for item in data})
    total = len(unique_paths)
    downloaded = 0
    skipped = 0
    failed = 0

    print(f"\n--- Syncing {total} images to local cache: {cache_dir} ---")

    for i, object_name in enumerate(unique_paths, start=1):
        local_path = safe_local_path(cache_dir, object_name)
        local_dir = os.path.dirname(local_path)
        os.makedirs(local_dir, exist_ok=True)

        if os.path.exists(local_path) and not force_download:
            skipped += 1
            if i % 100 == 0 or i == total:
                print(f"  [{i}/{total}] skipped existing: {object_name}")
            continue

        try:
            minio_client.fget_object(bucket, object_name, local_path)
            downloaded += 1
            if i % 50 == 0 or i == total:
                print(f"  [{i}/{total}] downloaded: {object_name}")
        except Exception as e:
            failed += 1
            print(f"  ERROR downloading {object_name}: {e}")

    print(f"\nCache sync complete:")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped:    {skipped}")
    print(f"  Failed:     {failed}")

    if failed > 0:
        raise RuntimeError(f"Failed to download {failed} image(s) from MinIO.")


def load_image_from_cache(cache_dir: str, image_path: str) -> Image.Image:
    """
    Load image from local cache.
    """
    local_path = safe_local_path(cache_dir, image_path)

    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f"Cached image not found: {local_path}. "
            f"Make sure download_images_to_cache() ran successfully."
        )

    with Image.open(local_path) as img:
        return img.convert("RGB")


# ============================================
# 2. DATASET CLASS
# ============================================
class WireDEDDataset(Dataset):
    """
    Custom dataset that:
    - Stores paired data (image path + prompt + target)
    - Loads images from local disk cache
    - Applies augmentation to defect images
    - Formats everything for LLaVA-NeXT
    """

    def __init__(self, data, config, processor, cache_dir: str):
        self.processor = processor
        self.cache_dir = cache_dir
        self.max_length = int(os.environ.get("MAX_LENGTH", config["training"]["max_length"]))

        aug_config = config["augmentation"]
        self.augment = aug_config["enabled"]
        self.rotation = aug_config["image_rotation_degrees"]
        self.brightness_range = aug_config["image_brightness_range"]

        self.defect_transform = T.Compose([
            T.RandomRotation(degrees=self.rotation),
            T.ColorJitter(brightness=(self.brightness_range[0], self.brightness_range[1])),
        ])

        if aug_config["oversample_defects"]:
            self.data = self._oversample_defects(data)
        else:
            self.data = data

    def _oversample_defects(self, data):
        normal = [d for d in data if d["label"] == 0]
        defects = [d for d in data if d["label"] == 1]

        if len(defects) == 0:
            print("WARNING: No defect samples found in data!")
            return data

        target_defect_count = int(len(normal) * 0.25)
        repeat_factor = max(1, target_defect_count // len(defects))
        repeat_factor = min(repeat_factor, 100)

        oversampled_defects = defects * repeat_factor
        combined = normal + oversampled_defects
        random.shuffle(combined)

        print(f"  Oversampling: {len(defects)} defects x{repeat_factor} = {len(oversampled_defects)}")
        print(f"  Final dataset: {len(normal)} normal + {len(oversampled_defects)} defects = {len(combined)} total")
        return combined

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load from local disk, not MinIO
        image = load_image_from_cache(self.cache_dir, item["image_path"])

        image = image.resize((336,336))

        if self.augment and item["label"] == 1:
            image = self.defect_transform(image)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": item["prompt"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": item["target"]},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=False
        )

        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=False,
        )

        labels = inputs["input_ids"].clone().squeeze(0)
        input_ids = inputs["input_ids"].squeeze(0)

        inst_end_token = self.processor.tokenizer.encode("[/INST]", add_special_tokens=False)
        input_ids_list = input_ids.tolist()

        mask_end = 0
        for i in range(len(input_ids_list) - len(inst_end_token) + 1):
            if input_ids_list[i:i + len(inst_end_token)] == inst_end_token:
                mask_end = i + len(inst_end_token)

        labels[:mask_end] = -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "image_sizes": inputs["image_sizes"].squeeze(0),
            "labels": labels,
            "is_defect": item["label"],
        }


# ============================================
# 3. CUSTOM TRAINER WITH WEIGHTED LOSS
# ============================================
class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        is_defect = inputs.pop("is_defect")

        outputs = model(**inputs)
        loss = outputs.loss

        weights = torch.where(
            is_defect.bool(),
            torch.tensor(self.class_weights["defect"], device=loss.device),
            torch.tensor(self.class_weights["normal"], device=loss.device),
        )

        batch_weight = weights.mean()
        weighted_loss = loss * batch_weight

        return (weighted_loss, outputs) if return_outputs else weighted_loss


# ============================================
# 4. MINIO CHECKPOINT CALLBACK (auto-upload)
# ============================================
class MinIOCheckpointCallback(TrainerCallback):
    """
    Automatically uploads checkpoint folders to MinIO after each save.
    This way, if the container is stopped mid-training, we can resume
    from MinIO later without losing progress.
    """

    def __init__(self, minio_client, bucket, minio_prefix="trained_adapters/checkpoints"):
        self.minio_client = minio_client
        self.bucket = bucket
        self.minio_prefix = minio_prefix

    def on_save(self, args, state, control, **kwargs):
        """Called by the Trainer after every checkpoint save."""
        checkpoint_name = f"checkpoint-{state.global_step}"
        local_checkpoint_dir = os.path.join(args.output_dir, checkpoint_name)

        if not os.path.exists(local_checkpoint_dir):
            print(f"  WARNING: checkpoint folder not found at {local_checkpoint_dir}")
            return

        print(f"\n--- Uploading {checkpoint_name} to MinIO ---")
        uploaded = 0
        for root, dirs, files in os.walk(local_checkpoint_dir):
            for filename in files:
                local_path = os.path.join(root, filename)
                rel_path = os.path.relpath(local_path, local_checkpoint_dir)
                minio_path = f"{self.minio_prefix}/{checkpoint_name}/{rel_path}"
                try:
                    self.minio_client.fput_object(self.bucket, minio_path, local_path)
                    uploaded += 1
                except Exception as e:
                    print(f"  ERROR uploading {rel_path}: {e}")
        print(f"  Uploaded {uploaded} files to {self.minio_prefix}/{checkpoint_name}/")


# ============================================
# 5. CHECKPOINT DOWNLOAD / RESUME HELPER
# ============================================
def download_latest_checkpoint_from_minio(minio_client, bucket, output_dir,
                                           minio_prefix="trained_adapters/checkpoints"):
    """
    Check MinIO for existing checkpoints. If found, download the latest one
    to output_dir so training can resume from it.
    Returns the local path to the latest checkpoint, or None if no checkpoint exists.
    """
    print(f"\n--- Checking MinIO for existing checkpoints ---")
    try:
        objects = list(minio_client.list_objects(bucket, prefix=minio_prefix + "/", recursive=True))
    except Exception as e:
        print(f"  Could not list checkpoints: {e}")
        return None

    if not objects:
        print(f"  No checkpoints found. Training will start from scratch.")
        return None

    # Find all unique checkpoint folder names (e.g., "checkpoint-8749")
    checkpoint_steps = set()
    for obj in objects:
        # Path format: trained_adapters/checkpoints/checkpoint-XXXX/file
        parts = obj.object_name.split("/")
        for part in parts:
            if part.startswith("checkpoint-"):
                try:
                    step = int(part.split("-")[1])
                    checkpoint_steps.add(step)
                except (IndexError, ValueError):
                    continue

    if not checkpoint_steps:
        print(f"  No valid checkpoint folders found.")
        return None

    # Pick the highest step number
    latest_step = max(checkpoint_steps)
    latest_name = f"checkpoint-{latest_step}"
    print(f"  Found latest checkpoint: {latest_name}")

    # Download all files for this checkpoint
    local_checkpoint_dir = os.path.join(output_dir, latest_name)
    os.makedirs(local_checkpoint_dir, exist_ok=True)

    print(f"  Downloading to {local_checkpoint_dir}...")
    downloaded = 0
    for obj in objects:
        if f"/{latest_name}/" not in "/" + obj.object_name:
            continue
        # Get path relative to the checkpoint folder
        rel_path = obj.object_name.split(f"{latest_name}/", 1)[1]
        local_path = os.path.join(local_checkpoint_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            minio_client.fget_object(bucket, obj.object_name, local_path)
            downloaded += 1
        except Exception as e:
            print(f"  ERROR downloading {rel_path}: {e}")

    print(f"  Downloaded {downloaded} files. Resume path: {local_checkpoint_dir}")
    return local_checkpoint_dir


# ============================================
# 6. MAIN TRAINING FUNCTION
# ============================================
def main():
    print("=" * 50)
    print("Wire DED Defect Detection - Fine-Tuning")
    print("=" * 50)

    config = load_config()
    print(f"\nModel: {config['model']['name']}")
    print(f"LoRA r={config['lora']['r']}, alpha={config['lora']['alpha']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")

    print("\n--- Loading data metadata ---")
    train_data, test_data = load_all_data(config)

    if len(train_data) == 0:
        print("ERROR: No training data loaded! Check your config and MinIO connection.")
        return

    print("\n--- Loading processor ---")
    processor = LlavaNextProcessor.from_pretrained(config["model"]["name"])
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print("\n--- Preparing local image cache ---")
    minio_client = get_minio_client(config)
    bucket = config["minio"]["bucket"]

    # Docker-friendly persistent cache location
    cache_dir = os.environ.get("IMAGE_CACHE_DIR", "/data/image_cache")
    force_download = os.environ.get("FORCE_DOWNLOAD_IMAGES", "false").lower() == "true"

    # Download both train and test image references
    all_data = train_data + test_data
    download_images_to_cache(
        data=all_data,
        minio_client=minio_client,
        bucket=bucket,
        cache_dir=cache_dir,
        force_download=force_download,
    )

    print("\n--- Loading model (4-bit quantized) ---")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = LlavaNextForConditionalGeneration.from_pretrained(
        config["model"]["name"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    model = prepare_model_for_kbit_training(model)

    print("\n--- Attaching LoRA adapters ---")
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\n--- Creating datasets ---")
    train_dataset = WireDEDDataset(train_data, config, processor, cache_dir=cache_dir)
    print(f"Training dataset size: {len(train_dataset)}")

    print("\n--- Setting up training ---")
    output_dir = config["training"]["output_dir"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        warmup_ratio=config["training"]["warmup_ratio"],
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
    )

    class_weights = {
        "normal": config["class_weight"]["normal"],
        "defect": config["class_weight"]["defect"],
    }

    # Check for existing checkpoint in MinIO BEFORE creating the trainer
    resume_checkpoint_path = download_latest_checkpoint_from_minio(
        minio_client=minio_client,
        bucket=bucket,
        output_dir=output_dir,
    )

    trainer = WeightedLossTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Add MinIO upload callback so every saved checkpoint goes to S3 automatically
    trainer.add_callback(MinIOCheckpointCallback(
        minio_client=minio_client,
        bucket=bucket,
    ))

    print("\n--- Starting training ---")
    print(f"Total training examples: {len(train_dataset)}")
    print(f"Effective batch size: {config['training']['batch_size']} x 4 = {config['training']['batch_size'] * 4}")
    print(f"Class weights: normal={class_weights['normal']}, defect={class_weights['defect']}")
    print(f"Image cache dir: {cache_dir}")
    if resume_checkpoint_path:
        print(f"RESUMING from checkpoint: {resume_checkpoint_path}")
    else:
        print("Starting fresh training (no checkpoint found)")
    print()

    if resume_checkpoint_path:
        trainer.train(resume_from_checkpoint=resume_checkpoint_path)
    else:
        trainer.train()

    print("\n--- Saving LoRA adapter ---")
    adapter_dir = os.path.join(output_dir, "lora_adapter")
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print(f"Adapter saved to {adapter_dir}")

    if config["training"]["save_to_minio"]:
        print("\n--- Uploading adapter to MinIO ---")
        upload_adapter_to_minio(minio_client, config["minio"]["bucket"], adapter_dir)

    print("\n--- Training complete! ---")


def upload_adapter_to_minio(client, bucket, adapter_dir):
    minio_prefix = "trained_adapters/lora_adapter"

    for root, dirs, files in os.walk(adapter_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            rel_path = os.path.relpath(local_path, adapter_dir)
            minio_path = f"{minio_prefix}/{rel_path}"

            client.fput_object(bucket, minio_path, local_path)
            print(f"  Uploaded: {minio_path}")

    print(f"  All adapter files uploaded to MinIO: {minio_prefix}/")


if __name__ == "__main__":
    main()
