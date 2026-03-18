"""
evaluate.py
Evaluates the fine-tuned LLaVA-NeXT model on the test set (WorkObject 9, 10, 11).
Computes F1-score, Precision, Recall, and Confusion Matrix.
Saves results to MinIO.
"""

import os
import json
import torch
from datetime import datetime
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig,
)
from peft import PeftModel
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from data_loader import load_config, load_all_data, get_minio_client, load_image_from_minio


def load_trained_model(config):
    """
    Load the base LLaVA-NeXT model + merge the trained LoRA adapter on top.
    Returns the model and processor ready for inference.
    """
    model_name = config["model"]["name"]
    adapter_dir = os.path.join(config["training"]["output_dir"], "lora_adapter")

    print(f"Loading base model: {model_name}")
    print(f"Loading LoRA adapter from: {adapter_dir}")

    # Load processor
    processor = LlavaNextProcessor.from_pretrained(adapter_dir)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Load base model in 4-bit (same config as training)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Load LoRA adapter on top of base model
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()  # set to evaluation mode (no dropout, no learning)

    print("Model loaded successfully.\n")
    return model, processor


def predict_single(model, processor, image, prompt, max_new_tokens=50):
    """
    Run inference on a single image + prompt.
    Returns the model's text response.
    """
    # Build conversation — only user message, no assistant (model must generate)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    # add_generation_prompt=True tells the processor to add the [/INST] token
    # so the model knows it should start generating
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True
    )

    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt",
    )

    # Move inputs to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate response — do_sample=False means deterministic (same input = same output)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Decode only the newly generated tokens (skip the input)
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response.strip()


def parse_prediction(response_text):
    """
    Parse the model's text response into a binary label.

    Looks for keywords in the response:
    - Contains "DEFECT" or "pore" -> predicted label = 1 (defect)
    - Contains "NORMAL" or "no porosity" -> predicted label = 0 (normal)
    - If unclear -> defaults to 0 (normal) to be conservative

    Returns: (predicted_label, raw_response)
    """
    text_lower = response_text.lower()

    if "defect" in text_lower or "pore detected" in text_lower:
        return 1, response_text
    elif "normal" in text_lower or "no porosity" in text_lower:
        return 0, response_text
    else:
        # Model gave an unexpected response — default to normal
        print(f"  WARNING: Unexpected response: '{response_text}' -> defaulting to NORMAL")
        return 0, response_text


def evaluate(config):
    """
    Main evaluation function.
    Loads model, runs predictions on test set, computes metrics.
    """
    print("=" * 50)
    print("Wire DED Defect Detection - Evaluation")
    print("=" * 50)

    # Load data (we only need test_data)
    print("\n--- Loading test data ---")
    _, test_data = load_all_data(config)

    if len(test_data) == 0:
        print("ERROR: No test data loaded!")
        return

    # Load trained model
    print("\n--- Loading trained model ---")
    model, processor = load_trained_model(config)

    # MinIO client for loading images
    minio_client = get_minio_client(config)
    bucket = config["minio"]["bucket"]

    # Run predictions
    print(f"\n--- Running predictions on {len(test_data)} test examples ---")
    true_labels = []
    pred_labels = []
    raw_responses = []

    for i, item in enumerate(test_data):
        # Progress update every 100 examples
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Processing {i+1}/{len(test_data)}...")

        # Load the image
        image = load_image_from_minio(minio_client, bucket, item["image_path"])

        # Get model prediction
        response = predict_single(model, processor, image, item["prompt"])

        # Parse response into label
        pred_label, raw_response = parse_prediction(response)

        true_labels.append(item["label"])
        pred_labels.append(pred_label)
        raw_responses.append(raw_response)

    # Compute metrics
    print("\n--- Results ---")

    f1 = f1_score(true_labels, pred_labels, pos_label=1, zero_division=0)
    precision = precision_score(true_labels, pred_labels, pos_label=1, zero_division=0)
    recall = recall_score(true_labels, pred_labels, pos_label=1, zero_division=0)
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])

    print(f"F1-Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  Predicted:    Normal  Defect")
    print(f"  Actual Normal:  {cm[0][0]:>5}   {cm[0][1]:>5}")
    print(f"  Actual Defect:  {cm[1][0]:>5}   {cm[1][1]:>5}")

    print(f"\nDetailed Report:")
    print(classification_report(
        true_labels, pred_labels,
        target_names=["Normal", "Defect"],
        zero_division=0,
    ))

    # Count how many defects were caught vs missed
    total_defects = sum(true_labels)
    caught_defects = sum(1 for t, p in zip(true_labels, pred_labels) if t == 1 and p == 1)
    missed_defects = sum(1 for t, p in zip(true_labels, pred_labels) if t == 1 and p == 0)
    false_alarms = sum(1 for t, p in zip(true_labels, pred_labels) if t == 0 and p == 1)

    print(f"Total real defects in test set: {total_defects}")
    print(f"Defects correctly caught: {caught_defects}")
    print(f"Defects missed: {missed_defects}")
    print(f"False alarms (normal flagged as defect): {false_alarms}")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "test_size": len(test_data),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "confusion_matrix": cm.tolist(),
        "total_defects": total_defects,
        "caught_defects": caught_defects,
        "missed_defects": missed_defects,
        "false_alarms": false_alarms,
    }

    # Save locally
    output_dir = config["training"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Upload to MinIO
    if config["training"]["save_to_minio"]:
        print("Uploading results to MinIO...")
        minio_client.fput_object(
            bucket,
            "trained_adapters/evaluation_results.json",
            results_path,
        )
        print("Results uploaded to MinIO: trained_adapters/evaluation_results.json")

    print("\n--- Evaluation complete! ---")
    return results


if __name__ == "__main__":
    config = load_config()
    evaluate(config)
