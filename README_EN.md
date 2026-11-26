# modal-llm-deploy

Serverless GPU deployment templates and tooling for private LLMs on [Modal](https://modal.com): **SGLang + FP8 Safetensors + Modal GPU Memory Snapshots** with scale-to-zero billing, ~20–30s cold-start restoration, OpenAI-compatible API, `<think>` reasoning separation, and tool calling.

[中文文档](README.md) · [Modal Directory Quick Reference](modal-deploy/README.md) · [MIT License](LICENSE)

> **Status**: The primary entrypoint [`modal-deploy/serve_sglang.py`](modal-deploy/serve_sglang.py) has been verified end-to-end on Modal `L40S` (48 GB) with a 28.8 GiB FineGrained FP8 27B checkpoint (401 auth rejection, `/v1/models`, streaming `<think>` separation, tool calling, and GPU snapshot restore). `serve_vllm.py` and `serve_llamacpp.py` are provided as alternative engine templates.

---

## Why This Instead of Renting a Dedicated GPU

For individual developers, researchers, or small teams running **low-to-medium traffic** (tens to hundreds of chat/agent calls per day), renting an always-on 48 GB GPU wastes most of the monthly bill on idle time.

This repository packages practical production guardrails for Modal Serverless GPUs:

| Challenge | How This Repo Solves It |
| :--- | :--- |
| **Idle GPU burn** | `min_containers=0` scales to zero automatically (\$0 compute when idle), per-second `L40S` billing (~\$1.95/hr active), and `MAX_CONTAINERS=1` enforces a strict single-GPU spend cap. |
| **Slow 27B cold starts** | Combines **Modal CPU+GPU Memory Snapshots**, pre-snapshot warmup, small-batch decode-only CUDA Graphs, and a persistent JIT cache Volume—cutting cold start from ~140–170s down to **~20–30s** on snapshot restore. |
| **Slow home uploads** | Built-in parallel cloud weight ingestion from **ModelScope** (1 Modal container per shard × 16-connection `aria2c` + size/SHA256 verification) and **Hugging Face** directly into Modal Volumes. |
| **BF16 won't fit on 48 GB** | One-shot cloud quantization script [`quantize_fp8.py`](modal-deploy/quantize_fp8.py) converts ~54 GB BF16 weights to ~27–29 GB `FP8_DYNAMIC` on a per-second `A100-80GB` so serving fits on a single `L40S`. |
| **Template & shard surprises** | Zero-dependency local preflight inspector [`inspect_weights.py`](modal-deploy/inspect_weights.py) and 4-step endpoint verifier [`smoke_test.py`](modal-deploy/smoke_test.py). |

---

## Quick Start (SGLang + L40S Primary Stack)

Prerequisites: Python 3.10+ and a [Modal](https://modal.com) account.

### 1. Install and Authenticate Modal CLI

```bash
pip install modal
modal setup
```

### 2. (Recommended) Preflight-Check Local Weights

Inspect safetensors shard completeness, FP8 `quantization_config`, and `<think>` / `<tool_call>` chat template tags without installing PyTorch:

```bash
python modal-deploy/inspect_weights.py /path/to/your-fp8-model-dir
```

### 3. Create API Key Secret and Weights Volume

The server **refuses to start without a Bearer API key**, preventing unauthenticated GPU exposure:

```bash
modal secret create qwen-api-key SGLANG_API_KEY="sk-replace-with-your-strong-secret"
modal volume create qwen38-27b-weights
```

### 4. Sync Weights into the Modal Volume (Pick One)

- **Option A — Parallel fan-out from ModelScope (recommended, ~2 min for 28.8 GiB)**
  ```bash
  modal run modal-deploy/serve_sglang.py \
    --model-id your-org/your-fp8-model \
    --target-subdir qwen38-27b-fp8 \
    --token your-modelscope-token
  ```
  > Omit `--token` for public repos. Fetches the file manifest locally, then fans out one Modal container per missing file using 16-connection `aria2c` with exact size and SHA256 checks.

- **Option B — Cloud pull from Hugging Face**
  ```bash
  modal run modal-deploy/serve_sglang.py::download_from_hf \
    --repo-id your-org/your-fp8-repo \
    --target-subdir qwen38-27b-fp8 \
    --token hf_your_token
  ```

- **Option C — Direct upload from local disk**
  ```bash
  modal volume put qwen38-27b-weights ./local-fp8-dir /qwen38-27b-fp8
  ```

> **Have BF16 weights (~54 GB) instead of FP8?** Upload to `/qwen38-27b-bf16` and run the cloud FP8 quantization job:
> ```bash
> modal run modal-deploy/quantize_fp8.py \
>   --input-subdir qwen38-27b-bf16 \
>   --output-subdir qwen38-27b-fp8
> ```

### 5. Configure Model Mapping and Deploy

Copy [`modal-deploy/.env.example`](modal-deploy/.env.example) to `modal-deploy/.env.local` (gitignored) or pass `SGLANG_MODEL_SUBDIR` / `SGLANG_SERVED_MODEL_NAME` via environment variables:

```bash
cp modal-deploy/.env.example modal-deploy/.env.local
# Edit SGLANG_MODEL_SUBDIR and SGLANG_SERVED_MODEL_NAME if needed
modal deploy modal-deploy/serve_sglang.py
```

Modal prints the deployed HTTPS endpoint (e.g. `https://<workspace>--qwen38-27b-sglang-sglangserver-serve.modal.run`).

### 6. Smoke Test (and Build the Initial GPU Snapshot)

```bash
python modal-deploy/smoke_test.py \
  --url https://<workspace>--qwen38-27b-sglang-sglangserver-serve.modal.run \
  --api-key "sk-replace-with-your-strong-secret" \
  --model qwen3.8-27b-ft
```

The smoke test verifies:
1. Unauthenticated `GET /v1/models` returns `401/403` (first call builds & persists the GPU memory snapshot)
2. Authenticated `GET /v1/models` lists the expected model ID
3. Streaming `POST /v1/chat/completions` (reports TTFT, total latency, `reasoning_content`, and `content`)
4. Structured OpenAI tool-calling argument parsing

For OpenAI-compatible SDKs and coding agents, set **Base URL** to `https://<your-endpoint>.modal.run/v1` and set client HTTP timeout to `>= 120s` to allow for cold-start restoration.

---

## Cold-Start Optimization Architecture

[`modal-deploy/serve_sglang.py`](modal-deploy/serve_sglang.py) layers six cold-start optimizations:

1. **Modal CPU + GPU Memory Snapshots** (`enable_memory_snapshot=True`, `experimental_options={"enable_gpu_snapshot": True}`, `@modal.enter(snap=True)`): captures host RAM + GPU VRAM once after initial startup; subsequent wakeups from zero restore directly from the snapshot instead of reading weights and re-initializing CUDA.
2. **Pre-Snapshot Warmup** (`_wait_and_warmup_sglang`): waits for `/health` and sends a local `max_tokens=4` completion before snapshotting, forcing FlashInfer / Triton / CUDA JIT compilation to finish inside the snapshot rather than stalling the first real user request.
3. **Persistent JIT Cache Volume** (`qwen38-27b-sglang-cache` mounted at `/cache`): persists `FLASHINFER_WORKSPACE_BASE`, `TRITON_CACHE_DIR`, and `TORCHINDUCTOR_CACHE_DIR` across full cold starts and image updates.
4. **Trimmed Decode-Only CUDA Graphs** (`--cuda-graph-max-bs-decode 4`, `--cuda-graph-backend-prefill=disabled`): avoids bloated multi-batch graph capture while preserving fast decode throughput with zero capture cost on snapshot restore.
5. **Single-GPU FP8 + Co-located Volume**: ~28.8 GiB FP8 weights fit cleanly on one 48 GB `L40S`, avoiding multi-GPU TP/NCCL initialization and public-internet weight downloads.
6. **60-Second Scaledown Window** (`SCALEDOWN_WINDOW_SECONDS = 60`): keeps the container warm across multi-turn chats and tool-call loops (~0.4–0.8s warm TTFT) before scaling back to zero.

> **Observed timings** (Modal `us-east`, single `L40S`, 28.77 GiB FineGrained FP8 27B checkpoint): initial snapshot creation ~`140–171s`, snapshot restore ~`29s`, warm streaming TTFT `0.4–0.8s`.

---

## Repository Layout & Engine Options

```text
.
├── modal-deploy/
│   ├── serve_sglang.py     # Primary: Modal L40S + SGLang + FP8 + GPU Snapshot + ModelScope parallel sync
│   ├── serve_vllm.py       # Alternative: Modal + vLLM (--enforce-eager for faster cold starts)
│   ├── serve_llamacpp.py   # Alternative: Modal L4 (24 GB) + llama-server + GGUF
│   ├── quantize_fp8.py     # One-off cloud job: BF16 -> FP8_DYNAMIC on A100-80GB
│   ├── inspect_weights.py  # Zero-dependency local weight & chat_template inspector
│   ├── smoke_test.py       # Zero-dependency 4-step endpoint verifier
│   ├── .env.example        # Local deploy-time env override template
│   └── README.md           # Concise Modal directory guide
├── PDR.md                  # Reference: local AMD RX 7800 XT + llama.cpp design review (Chinese)
├── PDP.md                  # Reference: local self-hosting rollout & benchmark plan (Chinese)
├── LICENSE                 # MIT License
├── README.md               # Chinese README
└── README_EN.md            # English README
```

| Script | Weight Format | Recommended GPU | Best For |
| :--- | :--- | :--- | :--- |
| [`serve_sglang.py`](modal-deploy/serve_sglang.py) | FP8 Safetensors (~27–29 GB) | `L40S` (48 GB) | **Primary stack**: 32K context, RadixAttention prefix caching, GPU snapshots, agent tool calling |
| [`serve_vllm.py`](modal-deploy/serve_vllm.py) | FP8 / AWQ / BF16 Safetensors | `L40S` / `L4` / `A100-80GB` | vLLM workflows or mixed-precision checkpoints |
| [`serve_llamacpp.py`](modal-deploy/serve_llamacpp.py) | GGUF (`Q4_K_M` / `UD-IQ4_XS`) | `L4` (24 GB, ~\$0.80/hr) | Existing GGUF checkpoints, lowest hourly GPU rate, parity with local `llama.cpp` |

Root [`PDR.md`](PDR.md) and [`PDP.md`](PDP.md) document a companion local self-hosting design (AMD RX 7800 XT 16 GB + `llama.cpp` Vulkan) kept here as an architectural reference when comparing local hardware vs. Modal serverless deployment.

---

## Tuning & Cost Guardrails

All core knobs live at the top of [`modal-deploy/serve_sglang.py`](modal-deploy/serve_sglang.py):

- **`SGLANG_MODEL_SUBDIR` / `SGLANG_SERVED_MODEL_NAME`**: override via shell env, gitignored `modal-deploy/.env.local`, or `PROFILE_MODELS` so private checkpoint names never need to be committed.
- **`SCALEDOWN_WINDOW_SECONDS`** (default `60`): increase to `180`–`300` for fewer cold starts during slower interactive sessions, or lower to `30` to minimize idle seconds.
- **`MAX_CONTAINERS`** (default `1`): hard concurrency ceiling against client retry storms.
- **Modal Billing Cap**: set a monthly spending limit in Modal Console under **Settings → Billing**.
- **Multi-profile deploys**: run `MODAL_PROFILE=<profile> modal deploy modal-deploy/serve_sglang.py` to manage multiple Modal workspaces cleanly.

---

## License

Released under the [MIT License](LICENSE).
