"""
Modal Serverless GPU deployment for private full-fine-tuned Qwen3.8-27B using vLLM.

Usage:
  1. modal secret create qwen-api-key VLLM_API_KEY="sk-your-strong-secret"
  2. modal volume create qwen38-27b-weights
  3. modal volume put qwen38-27b-weights ./local-model-dir /qwen38-27b-fp8
  4. modal deploy serve_vllm.py
"""

import os
import subprocess
import modal

# =============================================================================
# Configuration — adjust GPU & path to match your weight precision
# =============================================================================
APP_NAME = "qwen38-27b-vllm"
VOLUME_NAME = "qwen38-27b-weights"
SECRET_NAME = "qwen-api-key"

# Choose preset matching your uploaded weights:
#   - FP8 weights (~27 GB):   GPU="L40S" (48 GB),   MODEL_SUBDIR="qwen38-27b-fp8"
#   - AWQ/INT4 (~16 GB):      GPU="L4"   (24 GB),   MODEL_SUBDIR="qwen38-27b-awq"
#   - BF16 weights (~54 GB):  GPU="A100-80GB",      MODEL_SUBDIR="qwen38-27b-bf16"
GPU_TYPE = "L40S"
MODEL_DIR = "/models"
MODEL_SUBDIR = "qwen38-27b-fp8"
SERVED_MODEL_NAME = "qwen3.8-27b-ft"

# Context length & cost tuning for <=100 calls/day
MAX_MODEL_LEN = 16384
SCALEDOWN_WINDOW_SECONDS = 60  # Scale to 0 after 60s of no requests (30-120s recommended)
VLLM_PORT = 8000

# Optional: set to True if your model is a VL checkpoint and you only need text
DISABLE_MULTIMODAL = True

# Optional: tool calling parser (verify parser name supported by your vLLM version)
ENABLE_TOOL_CALLING = True
TOOL_CALL_PARSER = "qwen3_xml"  # Fallback: "hermes" depending on your chat template / vLLM build

# =============================================================================
# Modal Image & Resources
# =============================================================================
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install(
        "vllm",
        "huggingface_hub[hf_transfer]",
        "transformers",
        "accelerate",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # Keep compile caches off ephemeral paths if needed
            "VLLM_LOGGING_LEVEL": "INFO",
        }
    )
)

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=vllm_image,
    gpu=GPU_TYPE,
    volumes={MODEL_DIR: model_volume},
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    min_containers=0,
    timeout=600,
)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=VLLM_PORT, startup_timeout=300)
def serve():
    model_path = os.path.join(MODEL_DIR, MODEL_SUBDIR)
    api_key = os.environ.get("VLLM_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"Missing VLLM_API_KEY in Modal Secret '{SECRET_NAME}'. "
            "Refusing to expose an unauthenticated GPU endpoint."
        )
    if not os.path.isdir(model_path):
        raise RuntimeError(
            f"Model directory not found: {model_path}. "
            f"Upload weights first via: modal volume put {VOLUME_NAME} <local_dir> /{MODEL_SUBDIR}"
        )

    cmd = [
        "vllm",
        "serve",
        model_path,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--api-key",
        api_key,
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        "0.90",
        "--trust-remote-code",
        # Critical for low-frequency Serverless: skips CUDA Graph capture to cut cold start
        # time significantly (and reduces billed cold-start seconds).
        "--enforce-eager",
    ]

    if DISABLE_MULTIMODAL:
        cmd.extend(["--limit-mm-per-prompt", '{"image": 0, "video": 0}'])

    if ENABLE_TOOL_CALLING:
        cmd.extend(
            [
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                TOOL_CALL_PARSER,
            ]
        )

    print("Starting vLLM:", " ".join(c if c != api_key else "***" for c in cmd))
    subprocess.Popen(cmd)


@app.function(
    image=vllm_image,
    volumes={MODEL_DIR: model_volume},
    secrets=[modal.Secret.from_name("hf-token")],
    timeout=3600,
)
def download_from_private_hf(repo_id: str, target_subdir: str = MODEL_SUBDIR):
    """
    Optional helper: pull private weights from Hugging Face directly into Modal Volume
    (much faster than uploading from a home connection).

    Run:
      modal secret create hf-token HF_TOKEN="hf_xxx"
      modal run serve_vllm.py::download_from_private_hf --repo-id your-org/your-private-qwen38-27b
    """
    from huggingface_hub import snapshot_download

    target_dir = os.path.join(MODEL_DIR, target_subdir)
    os.makedirs(target_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
        token=os.environ["HF_TOKEN"],
    )
    model_volume.commit()
    print(f"Downloaded {repo_id} to volume path /{target_subdir}")
