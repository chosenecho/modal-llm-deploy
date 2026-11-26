"""
One-off cloud job on Modal: quantize BF16 full-fine-tuned weights in your Modal Volume
to FP8 (~27 GB) so they run cheaply on a single 48 GB L40S instead of an 80 GB A100.

Usage:
  1. Upload BF16 weights to Modal Volume:
       modal volume put qwen38-27b-weights ./local-bf16-dir /qwen38-27b-bf16
  2. Run quantization in the cloud (takes a few minutes on A100-80GB):
       modal run quantize_fp8.py
  3. (Optional) Remove the BF16 folder from the volume afterwards to save storage:
       modal volume rm -r qwen38-27b-weights /qwen38-27b-bf16
"""

import os
import modal

VOLUME_NAME = "qwen38-27b-weights"
MODEL_DIR = "/models"
INPUT_SUBDIR = "qwen38-27b-bf16"
OUTPUT_SUBDIR = "qwen38-27b-fp8"

quant_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch",
    "transformers",
    "accelerate",
    "llmcompressor",
    "safetensors",
)

app = modal.App("qwen38-27b-quantize-fp8")
model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=quant_image,
    gpu="A100-80GB",
    volumes={MODEL_DIR: model_volume},
    timeout=3600,
)
def quantize_to_fp8(
    input_subdir: str = INPUT_SUBDIR,
    output_subdir: str = OUTPUT_SUBDIR,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    src = os.path.join(MODEL_DIR, input_subdir)
    dst = os.path.join(MODEL_DIR, output_subdir)

    if not os.path.isdir(src):
        raise RuntimeError(f"Source BF16 directory not found: {src}")

    print(f"Loading BF16 model from {src} ...")
    model = AutoModelForCausalLM.from_pretrained(
        src,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=True)

    # Dynamic FP8 (data-free, near-lossless, fast)
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["lm_head"],
    )

    print("Running FP8_DYNAMIC quantization ...")
    oneshot(model=model, recipe=recipe)

    os.makedirs(dst, exist_ok=True)
    model.save_pretrained(dst, save_compressed=True)
    tokenizer.save_pretrained(dst)

    model_volume.commit()
    print(f"Saved FP8 weights to volume path /{output_subdir}")
