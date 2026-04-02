"""
test_speed.py
Tests different approaches to reduce training time.
Run on the GPU machine with: python src/test_speed.py

Tests:
1. Image token count WITH and WITHOUT resize
2. Training speed at different max_length values
3. Reports which settings are safe and fastest
"""

import torch
import time
from PIL import Image
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig,
)


def test_image_tokens():
    """Test how many tokens different image sizes produce."""
    print("=" * 60)
    print("TEST 1: Image token counts at different resolutions")
    print("=" * 60)

    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

    # Create test images at different sizes
    test_sizes = [
        (336, 336),
        (672, 672),
        (224, 224),
        (512, 512),
        (1024, 1024),
    ]

    prompt_text = (
        "Analyze this Wire DED sensor reading and the corresponding melt pool image:\n"
        "Layer=45.0, Bead=2.0, Current=85.3, Wire Feed Speed=5.2, "
        "Throughput_Speed=8.0, Laser Output Power=3500.0, Pyrometer1_Low=245.6, "
        "Pyrometer2_Mid=0.0, Pyrometer3_High=890.2, AI_LaserVoltage=5.3, "
        "Robot_DepositionWire_Speed=5.8, Conductance=45.2, Power_Wire=250.3\n"
        "Is this a defect or normal? If defect, estimate the pore diameter."
    )

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "NORMAL - no porosity detected."},
            ],
        },
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=False)

    for width, height in test_sizes:
        # Create a random test image
        img = Image.new("RGB", (width, height), color=(128, 128, 128))

        inputs = processor(
            text=text,
            images=img,
            return_tensors="pt",
            padding=False,
        )

        token_count = inputs["input_ids"].shape[1]
        print(f"  Image {width}x{height}: {token_count} total tokens")

    # Also test with actual melt pool image size if available
    print("\n  Recommendation:")
    print("  - If 336x336 gives <1024 tokens → use max_length=1024")
    print("  - If 336x336 gives <1500 tokens → use max_length=1536")
    print("  - Pick the smallest max_length that fits your token count + 50 buffer")

    return processor


def test_training_speed(processor):
    """Run a quick 10-step training at different max_lengths to measure speed."""
    print("\n" + "=" * 60)
    print("TEST 2: Training speed at different settings")
    print("=" * 60)

    # Load model
    print("\nLoading model (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Test different configurations
    configs_to_test = [
        {"image_size": (336, 336), "max_length": 1024, "batch_size": 4},
        {"image_size": (336, 336), "max_length": 1024, "batch_size": 8},
        {"image_size": (672, 672), "max_length": 2800, "batch_size": 4},
    ]

    prompt_text = (
        "Analyze this Wire DED sensor reading and the corresponding melt pool image:\n"
        "Layer=45.0, Bead=2.0, Current=85.3, Wire Feed Speed=5.2, "
        "Throughput_Speed=8.0, Laser Output Power=3500.0, Pyrometer1_Low=245.6, "
        "Pyrometer2_Mid=0.0, Pyrometer3_High=890.2, AI_LaserVoltage=5.3, "
        "Robot_DepositionWire_Speed=5.8, Conductance=45.2, Power_Wire=250.3\n"
        "Is this a defect or normal? If defect, estimate the pore diameter."
    )

    for cfg in configs_to_test:
        img_size = cfg["image_size"]
        max_len = cfg["max_length"]
        batch_sz = cfg["batch_size"]

        print(f"\n--- Testing: image={img_size}, max_length={max_len}, batch={batch_sz} ---")

        # Create fake batch
        img = Image.new("RGB", img_size, color=(128, 128, 128))

        conversation = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "NORMAL - no porosity detected."},
            ]},
        ]

        text = processor.apply_chat_template(conversation, add_generation_prompt=False)

        try:
            # Process one example
            inputs = processor(
                text=text,
                images=img,
                return_tensors="pt",
                padding="max_length",
                max_length=max_len,
                truncation=True,
            )

            token_count = inputs["input_ids"].shape[1]
            print(f"  Token count: {token_count}")

            # Create a batch by repeating
            batch = {
                "input_ids": inputs["input_ids"].repeat(batch_sz, 1).to(model.device),
                "attention_mask": inputs["attention_mask"].repeat(batch_sz, 1).to(model.device),
                "pixel_values": inputs["pixel_values"].repeat(batch_sz, 1, 1, 1).to(model.device),
                "image_sizes": inputs["image_sizes"].repeat(batch_sz, 1).to(model.device),
                "labels": inputs["input_ids"].clone().repeat(batch_sz, 1).to(model.device),
            }

            # Time 10 forward+backward passes
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

            torch.cuda.synchronize()
            start = time.time()

            for step in range(10):
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if step == 0:
                    print(f"  First step loss: {loss.item():.4f}")

            torch.cuda.synchronize()
            elapsed = time.time() - start

            per_step = elapsed / 10
            print(f"  10 steps in {elapsed:.1f}s ({per_step:.1f}s/step)")

            # Estimate full training time
            # Assume ~100k samples, 3 epochs, gradient_accumulation=4
            total_steps = (100000 * 3) / (batch_sz * 4)
            estimated_hours = (total_steps * per_step) / 3600
            print(f"  Estimated full training: {estimated_hours:.1f} hours ({estimated_hours/24:.1f} days)")

            # GPU memory
            mem_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  Peak GPU memory: {mem_used:.1f} GB")

            torch.cuda.reset_peak_memory_stats()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  OUT OF MEMORY — this config doesn't fit on the GPU")
                torch.cuda.empty_cache()
            else:
                print(f"  ERROR: {e}")


if __name__ == "__main__":
    processor = test_image_tokens()
    test_training_speed(processor)
