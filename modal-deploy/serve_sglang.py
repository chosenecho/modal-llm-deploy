"""
Modal Serverless GPU deployment for private full-fine-tuned Qwen3.8-27B
using FP8 Safetensors + SGLang.

Hardware & Cost Profile (<= 100 calls/day, cold-start optimized):
  - GPU: L40S (48 GB, sm_89 native FP8 tensor cores, ~$1.95/hr billed per second)
  - Scale-to-zero: min_containers=0, scaledown_window=60s, max_containers=1 (strict cost cap)
  - Cold-start optimization: GPU Memory Snapshot + pre-snapshot warmup + small-batch decode CUDA Graph + persistent JIT cache Volume

Quick Start:
  1. pip install modal && modal setup
  2. modal secret create qwen-api-key SGLANG_API_KEY="sk-your-strong-secret"
  3. modal volume create qwen38-27b-weights
  4. modal volume put qwen38-27b-weights ./your-fp8-safetensors-dir /qwen38-27b-fp8
  5. modal deploy serve_sglang.py
"""

import os
import subprocess
import modal

# =============================================================================
# Configuration — tune after running `python inspect_weights.py <weight_dir>`
# =============================================================================
APP_NAME = "qwen38-27b-sglang"
WEIGHTS_VOLUME_NAME = "qwen38-27b-weights"
CACHE_VOLUME_NAME = "qwen38-27b-sglang-cache"
SECRET_NAME = "qwen-api-key"

# L40S (48 GB, Ada Lovelace sm_89) has native FP8 tensor cores and fits
# ~28.8 GiB FineGrained FP8 weights + KV cache + hybrid recurrent state buffers comfortably.
GPU_TYPE = "L40S"
MODEL_DIR = "/models"

# Model mapping (baked into image env at deploy time).
# Override via shell env vars, local .env.local (gitignored), or PROFILE_MODELS below:
#   SGLANG_MODEL_SUBDIR       - directory inside the Modal Volume (/models/<subdir>)
#   SGLANG_SERVED_MODEL_NAME  - model ID exposed by /v1/models and /v1/chat/completions
DEFAULT_MODEL_SUBDIR = "qwen38-27b-fp8"
DEFAULT_SERVED_MODEL_NAME = "qwen3.8-27b-ft"
PROFILE_MODELS: dict[str, tuple[str, str]] = {}


def _read_dotenv_file(path: str) -> dict[str, str]:
    values: dict[str, str] = {}
    if not os.path.isfile(path):
        return values
    with open(path, "r", encoding="utf-8-sig") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key:
                values[key] = val
    return values


def _resolve_model_config() -> tuple[str, str]:
    local_env = _read_dotenv_file(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.local")
    )
    env_subdir = os.environ.get("SGLANG_MODEL_SUBDIR") or local_env.get("SGLANG_MODEL_SUBDIR")
    env_served = os.environ.get("SGLANG_SERVED_MODEL_NAME") or local_env.get("SGLANG_SERVED_MODEL_NAME")
    if env_subdir and env_served:
        return env_subdir, env_served
    profile = os.environ.get("MODAL_PROFILE") or getattr(modal.config, "_profile", None) or ""
    default_subdir, default_served = PROFILE_MODELS.get(
        profile, (DEFAULT_MODEL_SUBDIR, DEFAULT_SERVED_MODEL_NAME)
    )
    return env_subdir or default_subdir, env_served or default_served


MODEL_SUBDIR, SERVED_MODEL_NAME = _resolve_model_config()

# Context & memory settings
CONTEXT_LENGTH = 32768
MEM_FRACTION_STATIC = 0.82  # Safe headroom for hybrid linear/full attention + VL buffers on 48 GB L40S
MAX_RUNNING_REQUESTS = 4

# Quantization:
#   - None: auto-detect from model's config.json (FineGrained FP8 e4m3, block [128, 128])
#   - "fp8": force online dynamic FP8 quantization if uploading BF16 weights directly
QUANTIZATION: str | None = None

# Parser defaults for Qwen3 / Qwen3.8 checkpoints (verify with inspect_weights.py):
#   - <tool_call><function=...><parameter=...></function></tool_call> -> qwen3_coder
#   - <think>...</think> -> qwen3
TOOL_CALL_PARSER: str | None = "qwen3_coder"
REASONING_PARSER: str | None = "qwen3"

# Optional custom chat template path inside the mounted model directory (auto-loaded from chat_template.jinja)
CHAT_TEMPLATE_FILE: str | None = None

# Serverless cost & scaling guardrails
SCALEDOWN_WINDOW_SECONDS = 60   # Keep warm for 60s between turns to balance cost and cold starts
MAX_CONTAINERS = 1              # Strict single-GPU cap (~$1.95/hr max burn rate)
SGLANG_PORT = 8000
ENABLE_GPU_SNAPSHOT = True      # Modal CPU+GPU memory snapshot for ~15-30s cold starts
CUDA_GRAPH_MAX_BS = 4           # Captured once during snapshot; faster decode with zero restore penalty

# Official prebuilt SGLang image (sglang 0.5.23, transformers 5.19.0, cu130) with Qwen3_5 / Qwen3.8 support
SGLANG_IMAGE_TAG = "lmsysorg/sglang:v0.5.23"

# =============================================================================
# Modal Image & Persistent Volumes
# =============================================================================
CACHE_DIR = "/cache"

sglang_image = (
    modal.Image.from_registry(SGLANG_IMAGE_TAG)
    .entrypoint([])
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # Persist FlashInfer / Triton / TorchInductor JIT caches across cold starts
            "FLASHINFER_WORKSPACE_BASE": f"{CACHE_DIR}/flashinfer",
            "TRITON_CACHE_DIR": f"{CACHE_DIR}/triton",
            "TORCHINDUCTOR_CACHE_DIR": f"{CACHE_DIR}/torchinductor",
            "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "0",
            "SGLANG_MODEL_SUBDIR": MODEL_SUBDIR,
            "SGLANG_SERVED_MODEL_NAME": SERVED_MODEL_NAME,
        }
    )
)

app = modal.App(APP_NAME)
weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME_NAME, create_if_missing=True)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)


def _wait_and_warmup_sglang(proc: subprocess.Popen, port: int, api_key: str, model_name: str, timeout: int = 540) -> None:
    """Wait for SGLang /health and run one tiny completion so JIT kernels are warm before snapshotting."""
    import json
    import time
    import urllib.request

    t0 = time.monotonic()
    health_url = f"http://127.0.0.1:{port}/health"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Connection": "close",
    }

    while True:
        rc = proc.poll()
        if rc is not None:
            raise RuntimeError(f"SGLang exited prematurely with code {rc} during startup.")
        if time.monotonic() - t0 > timeout:
            raise TimeoutError(f"SGLang did not become healthy within {timeout}s.")
        try:
            req = urllib.request.Request(health_url, headers={"Authorization": f"Bearer {api_key}", "Connection": "close"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    break
        except Exception:
            time.sleep(1.0)

    print(f"[snapshot-prep] SGLang /health ready in {time.monotonic() - t0:.1f}s; running warmup request...", flush=True)

    warmup_payload = json.dumps(
        {
            "model": model_name,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
            "temperature": 0.0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode("utf-8")
    t_warm = time.monotonic()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=warmup_payload,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=240) as resp:
        resp.read()
    print(f"[snapshot-prep] Warmup completed in {time.monotonic() - t_warm:.1f}s.", flush=True)


@app.cls(
    image=sglang_image,
    gpu=GPU_TYPE,
    volumes={
        MODEL_DIR: weights_volume,
        CACHE_DIR: cache_volume,
    },
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    min_containers=0,
    max_containers=MAX_CONTAINERS,
    timeout=600,
    enable_memory_snapshot=ENABLE_GPU_SNAPSHOT,
    experimental_options={"enable_gpu_snapshot": True} if ENABLE_GPU_SNAPSHOT else None,
)
@modal.concurrent(max_inputs=8)
class SGLangServer:
    @modal.enter(snap=ENABLE_GPU_SNAPSHOT)
    def start(self):
        model_path = os.path.join(MODEL_DIR, MODEL_SUBDIR)
        api_key = os.environ.get("SGLANG_API_KEY") or os.environ.get("VLLM_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"Missing SGLANG_API_KEY (or VLLM_API_KEY) in Modal Secret '{SECRET_NAME}'. "
                "Refusing to expose an unauthenticated GPU endpoint."
            )
        if not os.path.isdir(model_path):
            raise RuntimeError(
                f"Model directory not found: {model_path}. "
                f"Upload weights first via: modal volume put {WEIGHTS_VOLUME_NAME} <local_dir> /{MODEL_SUBDIR}"
            )

        for sub in ("flashinfer", "triton", "torchinductor"):
            os.makedirs(os.path.join(CACHE_DIR, sub), exist_ok=True)

        cmd = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_path,
            "--served-model-name",
            SERVED_MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(SGLANG_PORT),
            "--api-key",
            api_key,
            "--context-length",
            str(CONTEXT_LENGTH),
            "--mem-fraction-static",
            str(MEM_FRACTION_STATIC),
            "--max-running-requests",
            str(MAX_RUNNING_REQUESTS),
            "--trust-remote-code",
        ]

        if ENABLE_GPU_SNAPSHOT and CUDA_GRAPH_MAX_BS > 0:
            # Capture small-batch decode CUDA graphs once during snapshot creation for faster decode.
            cmd.extend([
                "--cuda-graph-max-bs-decode",
                str(CUDA_GRAPH_MAX_BS),
                "--cuda-graph-backend-prefill=disabled",
            ])
        else:
            cmd.append("--disable-cuda-graph")

        if QUANTIZATION:
            cmd.extend(["--quantization", QUANTIZATION])

        if TOOL_CALL_PARSER:
            cmd.extend(["--tool-call-parser", TOOL_CALL_PARSER])

        if REASONING_PARSER:
            cmd.extend(["--reasoning-parser", REASONING_PARSER])

        if CHAT_TEMPLATE_FILE:
            template_path = os.path.join(model_path, CHAT_TEMPLATE_FILE)
            if not os.path.isfile(template_path):
                raise RuntimeError(f"Chat template file not found: {template_path}")
            cmd.extend(["--chat-template", template_path])

        print("Starting SGLang:", " ".join(c if c != api_key else "***" for c in cmd), flush=True)
        self.proc = subprocess.Popen(cmd)

        if ENABLE_GPU_SNAPSHOT:
            _wait_and_warmup_sglang(self.proc, SGLANG_PORT, api_key, SERVED_MODEL_NAME)
            try:
                cache_volume.commit()
            except Exception as exc:
                print(f"[snapshot-prep] cache_volume.commit warning (non-fatal): {exc}", flush=True)

    @modal.web_server(port=SGLANG_PORT, startup_timeout=600)
    def serve(self):
        # SGLang is started in @modal.enter(snap=True) and already listening on 0.0.0.0:SGLANG_PORT
        # after GPU memory snapshot restore.
        pass

    @modal.exit()
    def stop(self):
        proc = getattr(self, "proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()


@app.function(
    image=modal.Image.debian_slim(python_version="3.12").pip_install(
        "huggingface_hub[hf_transfer]"
    ).env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}),
    cpu=4.0,
    memory=8192,
    volumes={MODEL_DIR: weights_volume},
    timeout=3600,
)
def download_from_hf(
    repo_id: str,
    target_subdir: str = MODEL_SUBDIR,
    token: str | None = None,
):
    """
    Pull FP8 Safetensors weights from Hugging Face directly into Modal Volume
    (runs on Modal cloud bandwidth, much faster than local upload).

    Usage:
      modal run serve_sglang.py::download_from_hf --repo-id your-org/your-repo [--token hf_xxx]
    """
    from huggingface_hub import snapshot_download

    target_dir = os.path.join(MODEL_DIR, target_subdir)
    os.makedirs(target_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
        token=token or os.environ.get("HF_TOKEN"),
    )
    weights_volume.commit()
    print(f"Downloaded {repo_id} to Modal Volume path /{target_subdir}")


downloader_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("aria2")
    .pip_install("requests")
)


@app.function(
    image=downloader_image,
    cpu=2.0,
    memory=4096,
    volumes={MODEL_DIR: weights_volume},
    timeout=1800,
    retries=modal.Retries(max_retries=2, initial_delay=3.0),
)
def download_modelscope_file(
    model_id: str,
    rel_path: str,
    expected_size: int,
    expected_sha: str = "",
    target_subdir: str = MODEL_SUBDIR,
    token: str | None = None,
) -> str:
    """Download a single ModelScope file using 16-connection aria2c + SHA256 verification."""
    import hashlib
    import time
    from pathlib import Path

    weights_volume.reload()
    target_dir = Path(MODEL_DIR) / target_subdir
    dest = target_dir / rel_path
    if dest.is_file() and (not expected_size or dest.stat().st_size == expected_size):
        return f"SKIP {rel_path} (already present)"

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    url = f"https://www.modelscope.cn/models/{model_id}/resolve/master/{rel_path}"

    cmd = [
        "aria2c",
        "--dir", str(dest.parent),
        "--out", part.name,
        "--allow-overwrite=true",
        "--auto-file-renaming=false",
        "--continue=true",
        "--max-connection-per-server=16",
        "--split=16",
        "--min-split-size=8M",
        "--max-tries=5",
        "--retry-wait=3",
        "--connect-timeout=30",
        "--timeout=120",
        "--console-log-level=warn",
        "--summary-interval=0",
        "--user-agent=Mozilla/5.0",
    ]
    if token:
        cmd.extend(["--header", f"Authorization: Bearer {token}"])
    cmd.append(url)

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"aria2c failed for {rel_path}: {proc.stderr[-500:] or proc.stdout[-500:]}")

    actual_size = part.stat().st_size
    if expected_size and actual_size != expected_size:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"Size mismatch for {rel_path}: got {actual_size}, expected {expected_size}")

    if expected_sha:
        hasher = hashlib.sha256()
        with part.open("rb") as fh:
            for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
                hasher.update(chunk)
        actual_sha = hasher.hexdigest().lower()
        if actual_sha != expected_sha.lower():
            part.unlink(missing_ok=True)
            raise RuntimeError(f"SHA256 mismatch for {rel_path}: got {actual_sha}, expected {expected_sha}")

    part.replace(dest)
    dt = max(time.time() - t0, 0.001)
    mbps = (actual_size / (1024 * 1024)) / dt

    for attempt in range(1, 6):
        try:
            weights_volume.commit()
            break
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 * attempt)

    msg = f"OK {rel_path} ({actual_size / (1024**3):.2f} GiB in {dt:.1f}s, {mbps:.1f} MB/s)"
    print(msg, flush=True)
    return msg


@app.function(
    image=downloader_image,
    cpu=2.0,
    memory=4096,
    volumes={MODEL_DIR: weights_volume},
    timeout=3600,
)
def download_from_modelscope(
    model_id: str,
    target_subdir: str = MODEL_SUBDIR,
    token: str | None = None,
    manifest_json: str = "",
):
    """
    Fast parallel fan-out downloader from ModelScope into Modal Volume.
    Spawns 1 Modal container per missing file, each using 16-connection aria2c + SHA256 check.

    Usage:
      modal run serve_sglang.py --model-id your-org/your-model [--target-subdir ...]
    """
    import json
    from pathlib import Path
    import requests

    weights_volume.reload()
    target_dir = Path(MODEL_DIR) / target_subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    # Clean up stale .incomplete / .part / .aria2 files from prior interrupted runs
    cleaned = False
    for pat in ("*.incomplete", "*.part", "*.aria2"):
        for stale in target_dir.rglob(pat):
            try:
                stale.unlink()
                cleaned = True
            except OSError:
                pass
    if cleaned:
        weights_volume.commit()

    if manifest_json:
        files = json.loads(manifest_json)
    else:
        headers = {"User-Agent": "Mozilla/5.0"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        api_url = f"https://www.modelscope.cn/api/v1/models/{model_id}/repo/files?Recursive=True"
        resp = requests.get(api_url, headers=headers, timeout=30)
        resp.raise_for_status()
        raw_files = resp.json().get("Data", {}).get("Files", [])
        files = [f for f in raw_files if f.get("Type") == "blob"]

    pending = []
    for f in files:
        dest = target_dir / f["Path"]
        expected_size = int(f.get("Size", 0))
        if dest.is_file() and dest.stat().st_size == expected_size:
            continue
        pending.append(f)

    total_count = len(files)
    print(
        f"ModelScope repo {model_id}: {total_count} files total, "
        f"{total_count - len(pending)} already present, {len(pending)} fanning out in parallel...",
        flush=True,
    )

    if pending:
        args_iter = [
            (
                model_id,
                f["Path"],
                int(f.get("Size", 0)),
                f.get("Sha256") or "",
                target_subdir,
                token,
            )
            for f in pending
        ]
        done = total_count - len(pending)
        for msg in download_modelscope_file.starmap(args_iter):
            done += 1
            print(f"[{done}/{total_count}] {msg}", flush=True)

    weights_volume.reload()
    all_files = [p for p in target_dir.rglob("*") if p.is_file() and not p.name.endswith((".part", ".aria2", ".incomplete"))]
    shards = sorted(target_dir.glob("*.safetensors"))
    total_gib = sum(p.stat().st_size for p in all_files) / (1024 ** 3)
    print(
        f"DONE {model_id} -> /{target_subdir}: "
        f"{len(all_files)} files ({len(shards)} safetensors shards, {total_gib:.2f} GiB)",
        flush=True,
    )


@app.local_entrypoint()
def sync_modelscope(
    model_id: str,
    target_subdir: str = MODEL_SUBDIR,
    token: str = "",
):
    """Fetch ModelScope file list locally and fan out parallel downloads on Modal."""
    import json
    import urllib.request

    headers = {"User-Agent": "Mozilla/5.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    api_url = f"https://www.modelscope.cn/api/v1/models/{model_id}/repo/files?Recursive=True"
    req = urllib.request.Request(api_url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    raw_files = data.get("Data", {}).get("Files", [])
    files = [
        {"Path": f["Path"], "Size": int(f.get("Size", 0)), "Sha256": f.get("Sha256") or ""}
        for f in raw_files
        if f.get("Type") == "blob"
    ]
    print(f"Fetched local manifest for {model_id}: {len(files)} files -> target /{target_subdir}", flush=True)
    download_from_modelscope.remote(
        model_id=model_id,
        target_subdir=target_subdir,
        token=token or None,
        manifest_json=json.dumps(files),
    )
