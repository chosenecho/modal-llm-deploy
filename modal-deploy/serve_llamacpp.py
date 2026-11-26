"""
Alternative Modal Serverless GPU deployment using llama.cpp (llama-server + GGUF).

Best when:
  - You already converted your full-fine-tuned Qwen3.8-27B weights to GGUF
    (e.g. Q4_K_M / UD-IQ4_XS ~15-17 GB, or Q8_0 ~28 GB), OR
  - You want ultra-fast cold starts and guaranteed compatibility with Qwen3.8's
    hybrid architecture (matching your local llama.cpp setup).

Usage:
  1. modal secret create qwen-api-key LLAMA_API_KEY="sk-your-strong-secret"
  2. modal volume create qwen38-27b-weights
  3. modal volume put qwen38-27b-weights ./Qwen3.8-27B-ft-Q4_K_M.gguf /gguf/Qwen3.8-27B-ft-Q4_K_M.gguf
  4. modal deploy serve_llamacpp.py
"""

import os
import subprocess
import modal

# =============================================================================
# Configuration
# =============================================================================
APP_NAME = "qwen38-27b-llamacpp"
VOLUME_NAME = "qwen38-27b-weights"
SECRET_NAME = "qwen-api-key"

# 24 GB L4 (~$0.80/hr) fits Q4_K_M / UD-IQ4_XS (~15-17 GB) + 16K-32K context comfortably.
# Use "L40S" (48 GB) if serving Q8_0 (~28 GB) or very large context.
GPU_TYPE = "L4"
MODEL_DIR = "/models"
GGUF_REL_PATH = "gguf/Qwen3.8-27B-ft-Q4_K_M.gguf"
MODEL_ALIAS = "qwen3.8-27b-ft"

CTX_SIZE = 16384
PARALLEL_SLOTS = 2  # Total context is split across slots unless --no-kv-unified is tuned
SCALEDOWN_WINDOW_SECONDS = 60
SERVER_PORT = 8080
# Require Modal-Key / Modal-Secret at Modal's edge proxy before waking GPU containers
REQUIRE_PROXY_AUTH = True

# Official prebuilt CUDA llama-server image (pinned tag recommended for production)
LLAMA_IMAGE = modal.Image.from_registry(
    "ghcr.io/ggml-org/llama.cpp:server-cuda",
    add_python="3.12",
)

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=LLAMA_IMAGE,
    gpu=GPU_TYPE,
    volumes={MODEL_DIR: model_volume},
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    min_containers=0,
    timeout=600,
)
@modal.concurrent(max_inputs=4)
@modal.web_server(port=SERVER_PORT, startup_timeout=240, requires_proxy_auth=REQUIRE_PROXY_AUTH)
def serve():
    model_file = os.path.join(MODEL_DIR, GGUF_REL_PATH)
    api_key = os.environ.get("LLAMA_API_KEY") or os.environ.get("VLLM_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"Missing LLAMA_API_KEY (or VLLM_API_KEY) in Modal Secret '{SECRET_NAME}'."
        )
    if not os.path.isfile(model_file):
        raise RuntimeError(
            f"GGUF file not found: {model_file}. "
            f"Upload via: modal volume put {VOLUME_NAME} <local.gguf> /{GGUF_REL_PATH}"
        )

    cmd = [
        "/app/llama-server",
        "-m",
        model_file,
        "--alias",
        MODEL_ALIAS,
        "-ngl",
        "999",
        "-c",
        str(CTX_SIZE),
        "-np",
        str(PARALLEL_SLOTS),
        "--cont-batching",
        "-fa",
        "on",
        "-ctk",
        "q8_0",
        "-ctv",
        "q8_0",
        "--no-mmproj",
        "--jinja",
        "--api-key",
        api_key,
        "--host",
        "0.0.0.0",
        "--port",
        str(SERVER_PORT),
    ]

    print("Starting llama-server:", " ".join(c if c != api_key else "***" for c in cmd))
    subprocess.Popen(cmd)
