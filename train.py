"""
train.py
Fine-tunes LLaVA-NeXT (Mistral-7B) on Wire DED defect detection data
using LoRA + 4-bit quantization + on-the-fly augmentation + weighted loss.
"""

import os
import io
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from data_loader import load_config, load_all_data, get_minio_client, load_image_from_minio
import torchvision.transforms as T


# ============================================
# 1. DATASET CLASS
# ============================================
class WireDEDDataset(Dataset):
    """
    Custom dataset that:
    - Stores paired data (image path + prompt + target)
    - Downloads images from MinIO on-the-fly
    - Applies augmentation to defect images
    - Formats everything for LLaVA-NeXT
    """

    def __init__(self, data, config, processor, minio_client):
        """
        Args:
            data: list of dicts from data_loader (image_path, prompt, target, label)
            config: full config dict
            processor: LLaVA-NeXT processor
            minio_client: MinIO client for downloading images
        """
        self.processor = processor
        self.minio_client = minio_client
        self.bucket = config["minio"]["bucket"]
        self.max_length = config["training"]["max_length"]

        # Augmentation settings
        aug_config = config["augmentation"]
        self.augment = aug_config["enabled"]
        self.sensor_noise = aug_config["sensor_noise_percent"]
        self.rotation = aug_config["image_rotation_degrees"]
        self.brightness_range = aug_config["image_brightness_range"]

        # Image augmentation transforms (only applied to defect samples)
        self.defect_transform = T.Compose([
            T.RandomRotation(degrees=self.rotation),
            T.ColorJitter(brightness=(self.brightness_range[0], self.brightness_range[1])),
        ])

        # Oversample defect examples if enabled
        if aug_config["oversample_defects"]:
            self.data = self._oversample_defects(data, config)
        else:
            self.data = data

    def _oversample_defects(self, data, config):
        """
        Duplicate defect samples so they appear roughly as often as normal samples.
        We don't duplicate to exact 50/50 — we use the class weight ratio.
        Example: if weight=50, we duplicate defects ~50x.
        But we cap it to avoid extreme duplication.
        """
        normal = [d for d in data if d["label"] == 0]
        defects = [d for d in data if d["label"] == 1]

        if len(defects) == 0:
            print("WARNING: No defect samples found in data!")
            return data

        # Calculate how many times to repeat defects
        # Goal: make defect count roughly 20-30% of total (not 50/50, that's too aggressive)
        target_defect_count = int(len(normal) * 0.25)
        repeat_factor = max(1, target_defect_count // len(defects))
        repeat_factor = min(repeat_factor, 100)  # cap at 100x to be safe

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

        # Step 1: Download image from MinIO
        image = load_image_from_minio(self.minio_client, self.bucket, item["image_path"])

        # Step 1.5: Resize to 336x336 to prevent AnyRes multi-patch splitting
        # Without this: image produces ~3117 tokens (multiple patches)
        # With this:    image produces ~1365 tokens (single patch)
        # This makes training ~2.3x faster with no meaningful quality loss
        # for melt pool defect patterns
        image = image.resize((336, 336))

        # Step 2: Apply augmentation if this is a defect AND augmentation is enabled
        if self.augment and item["label"] == 1:
            image = self.defect_transform(image)

        # Step 3: Build the conversation in LLaVA format
        # The processor.apply_chat_template handles the [INST]...[/INST] formatting
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

        # Step 4: Apply chat template to get the formatted text
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=False
        )

        # Step 5: Process image + text together
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        # Step 6: Create labels (same as input_ids, but mask the user prompt part)
        # The model should only learn to predict the assistant's response
        labels = inputs["input_ids"].clone().squeeze(0)
        input_ids = inputs["input_ids"].squeeze(0)

        # Find where the assistant response starts (after [/INST])
        # Mask everything before that with -100 (ignored in loss)
        # We look for the [/INST] token pattern in the input
        inst_end_token = self.processor.tokenizer.encode("[/INST]", add_special_tokens=False)
        input_ids_list = input_ids.tolist()

        # Find the last occurrence of [/INST] tokens
        mask_end = 0
        for i in range(len(input_ids_list) - len(inst_end_token) + 1):
            if input_ids_list[i:i+len(inst_end_token)] == inst_end_token:
                mask_end = i + len(inst_end_token)

        # Mask everything up to and including [/INST]
        labels[:mask_end] = -100

        # Also mask padding tokens
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "image_sizes": inputs["image_sizes"].squeeze(0),
            "labels": labels,
            "is_defect": item["label"],  # used for weighted loss
        }


# ============================================
# 2. CUSTOM TRAINER WITH WEIGHTED LOSS
# ============================================
class WeightedLossTrainer(Trainer):
    """
    Custom trainer that applies higher loss weight to defect samples.
    This makes the model pay more attention to defects during training.
    """

    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights  # {"normal": 1.0, "defect": 50.0}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract the defect flag before passing to model
        is_defect = inputs.pop("is_defect")

        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss

        # Apply per-sample weights based on whether it's a defect
        # Create weight tensor: 1.0 for normal, 50.0 for defect
        weights = torch.where(
            is_defect.bool(),
            torch.tensor(self.class_weights["defect"], device=loss.device),
            torch.tensor(self.class_weights["normal"], device=loss.device),
        )

        # Average weight for this batch
        batch_weight = weights.mean()
        weighted_loss = loss * batch_weight

        return (weighted_loss, outputs) if return_outputs else weighted_loss


# ============================================
# 3. MAIN TRAINING FUNCTION
# ============================================
def main():
    print("=" * 50)
    print("Wire DED Defect Detection - Fine-Tuning")
    print("=" * 50)

    # --- Load config ---
    config = load_config()
    print(f"\nModel: {config['model']['name']}")
    print(f"LoRA r={config['lora']['r']}, alpha={config['lora']['alpha']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")

    # --- Load data ---
    print("\n--- Loading data from MinIO ---")
    train_data, test_data = load_all_data(config)

    if len(train_data) == 0:
        print("ERROR: No training data loaded! Check your config and MinIO connection.")
        return

    # --- Load processor ---
    print("\n--- Loading processor ---")
    processor = LlavaNextProcessor.from_pretrained(config["model"]["name"])

    # Make sure pad token is set
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # --- Load model with 4-bit quantization ---
    print("\n--- Loading model (4-bit quantized) ---")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",            # normalized float 4-bit
        bnb_4bit_compute_dtype=torch.float16,  # compute in float16 for speed
        bnb_4bit_use_double_quant=True,        # double quantization saves more memory
    )

    model = LlavaNextForConditionalGeneration.from_pretrained(
        config["model"]["name"],
        quantization_config=bnb_config,
        device_map="auto",         # automatically places layers on available GPUs
        torch_dtype=torch.float16,
    )

    # Prepare model for LoRA training (handles quantization compatibility)
    model = prepare_model_for_kbit_training(model)

    # --- Attach LoRA adapters ---
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
    model.print_trainable_parameters()  # shows how many params are trainable

    # --- Create datasets ---
    print("\n--- Creating datasets ---")
    minio_client = get_minio_client(config)

    train_dataset = WireDEDDataset(train_data, config, processor, minio_client)
    print(f"Training dataset size: {len(train_dataset)}")

    # --- Set up training arguments ---
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
        save_total_limit=2,              # keep only last 2 checkpoints
        fp16=True,                       # use mixed precision for speed
        gradient_accumulation_steps=4,   # simulate larger batch size
        gradient_checkpointing=True,     # trade compute for memory
        dataloader_num_workers=2,
        remove_unused_columns=False,     # we have custom columns (is_defect)
        report_to="none",               # disable wandb/tensorboard for now
    )

    # --- Create trainer with weighted loss ---
    class_weights = {
        "normal": config["class_weight"]["normal"],
        "defect": config["class_weight"]["defect"],
    }

    trainer = WeightedLossTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # --- Train ---
    print("\n--- Starting training ---")
    print(f"Total training examples: {len(train_dataset)}")
    print(f"Effective batch size: {config['training']['batch_size']} x 4 (gradient accumulation) = {config['training']['batch_size'] * 4}")
    print(f"Class weights: normal={class_weights['normal']}, defect={class_weights['defect']}")
    print()

    trainer.train()

    # --- Save LoRA adapter ---
    print("\n--- Saving LoRA adapter ---")
    adapter_dir = os.path.join(output_dir, "lora_adapter")
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print(f"Adapter saved to {adapter_dir}")

    # --- Upload to MinIO if configured ---
    if config["training"]["save_to_minio"]:
        print("\n--- Uploading adapter to MinIO ---")
        upload_adapter_to_minio(minio_client, config["minio"]["bucket"], adapter_dir)

    print("\n--- Training complete! ---")


def upload_adapter_to_minio(client, bucket, adapter_dir):
    """Upload all adapter files to MinIO under 'trained_adapters/' prefix."""
    minio_prefix = "trained_adapters/lora_adapter"

    for root, dirs, files in os.walk(adapter_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            # Relative path from adapter_dir
            rel_path = os.path.relpath(local_path, adapter_dir)
            minio_path = f"{minio_prefix}/{rel_path}"

            client.fput_object(bucket, minio_path, local_path)
            print(f"  Uploaded: {minio_path}")

    print(f"  All adapter files uploaded to MinIO: {minio_prefix}/")


if __name__ == "__main__":
    main()
