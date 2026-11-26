# Modal 弹性部署目录说明

完整文档、冷启动优化原理与引擎对比请见仓库根目录：[`README.md`](../README.md)（[English](../README_EN.md)）。

## 文件清单

| 文件 | 用途 |
| :--- | :--- |
| [`serve_sglang.py`](serve_sglang.py) | **主力部署脚本**：Modal `L40S` (48 GB) + SGLang (`v0.5.23`) + FP8 Safetensors，集成 **Modal 边缘网关鉴权**（`requires_proxy_auth=True`，未授权请求不唤醒 GPU）+ Bearer API Key 双层鉴权、`min_containers=0`、`scaledown_window=60s`、`max_containers=1`、**CPU+GPU 显存快照**、快照前 Warmup、小 batch decode CUDA Graph、持久化 JIT 缓存卷，以及 ModelScope 并发拉取 / HF 云端拉取。 |
| [`inspect_weights.py`](inspect_weights.py) | **权重预检脚本（零依赖）**：本地检查分片完整性、FP8 `quantization_config`、架构及 `<tool_call>` / `<think>` 模板，输出推荐参数。 |
| [`smoke_test.py`](smoke_test.py) | **上线验收脚本（零依赖）**：自动验证未授权 401/403 拦截、`/v1/models`、流式输出 TTFT 与推理分离、以及 OpenAI Tool Calling 闭环。 |
| [`quantize_fp8.py`](quantize_fp8.py) | **云端量化脚本**：在按秒计费的 Modal `A100-80GB` 上将 BF16 权重一键转为 `FP8_DYNAMIC`。 |
| [`serve_vllm.py`](serve_vllm.py) | **备选引擎模板**：Modal + vLLM（开启 `--enforce-eager` 压缩冷启动）。 |
| [`serve_llamacpp.py`](serve_llamacpp.py) | **备选引擎模板**：Modal `L4` (24 GB) + `llama-server` + GGUF。 |
| [`.env.example`](.env.example) | 本地覆盖配置模板（复制为 `.env.local` 后生效，已被 `.gitignore` 忽略）。 |

## 常用命令速查（在仓库根目录执行）

```bash
# 1. 本地权重预检
python modal-deploy/inspect_weights.py <本地权重目录>

# 2. 创建 Secret 与 Volume
modal secret create qwen-api-key SGLANG_API_KEY="sk-your-strong-secret"
modal volume create qwen38-27b-weights

# 3. 同步权重（三选一）
# 3a) ModelScope 云端并发拉取（16 线程 aria2c + SHA256 校验）
modal run modal-deploy/serve_sglang.py --model-id <org/repo> --target-subdir qwen38-27b-fp8 [--token <ms-token>]
# 3b) Hugging Face 云端拉取
modal run modal-deploy/serve_sglang.py::download_from_hf --repo-id <org/repo> --target-subdir qwen38-27b-fp8 [--token <hf-token>]
# 3c) 本地目录上传
modal volume put qwen38-27b-weights <本地FP8目录> /qwen38-27b-fp8

# 4. 部署（可选先复制 .env.example 为 .env.local 覆盖模型目录与名称）
modal deploy modal-deploy/serve_sglang.py

# 5. 全链路 4 项验收（先在 Modal 控制台 Settings -> Proxy Auth Tokens 生成 wk-/ws-；--url 不要带 /v1 后缀）
python modal-deploy/smoke_test.py \
  --url https://<workspace>--qwen38-27b-sglang-sglangserver-serve.modal.run \
  --api-key "sk-your-strong-secret" \
  --modal-key "wk-your-proxy-key" \
  --modal-secret "ws-your-proxy-secret" \
  --model qwen3.8-27b-ft
```
