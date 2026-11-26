# modal-llm-deploy

面向 [Modal](https://modal.com) Serverless GPU 平台的私有大语言模型弹性部署模板与工具链：基于 **SGLang + FP8 Safetensors + Modal GPU 显存快照**，实现闲置缩容到零、冷启动 ~20–30 秒唤醒、按秒计费与 OpenAI 兼容 API（支持 `<think>` 推理分离与 Tool Calling）。

[English](README_EN.md) · [Modal 目录快速说明](modal-deploy/README.md) · [MIT License](LICENSE)

> **当前状态**：主力脚本 [`modal-deploy/serve_sglang.py`](modal-deploy/serve_sglang.py) 已在 Modal `L40S`（48 GB）上以 28.8 GiB FineGrained FP8 27B 权重完成端到端实测验证（401 鉴权拦截、`/v1/models`、流式 `<think>` 分离、Tool Calling 闭环与 GPU Snapshot 恢复）。`serve_vllm.py` 与 `serve_llamacpp.py` 作为备选引擎模板提供。

---

## 为什么选这套方案（而不是直接包月租卡）

对于个人、科研或小团队的**低中频调用场景**（如每天几十到几百次对话 / Agent 调用），常驻包租一张 48 GB GPU（如 L40S / A6000）每月通常需要数千人民币，且大部分时间都在空转。

本仓库针对 Modal Serverless GPU 做了工程化收口：

| 痛点 | 本仓库的解法 |
| :--- | :--- |
| **常驻租卡太贵** | `min_containers=0` 闲置自动缩容到零（GPU 算力费 \$0），单卡 L40S 按秒计费（约 \$1.95/hr），`MAX_CONTAINERS=1` 锁死峰值消费上限。 |
| **27B 冷启动太慢** | 启用 **Modal CPU+GPU 显存快照** + 快照前主动 Warmup + 裁剪小 batch decode CUDA Graph + 持久化 JIT 缓存卷，将完整启动 140–170s 压缩至快照恢复 **~20–30s**。 |
| **国内/私有权重上传慢** | 内置 ModelScope **云端多容器并发拉取**（每分片 1 个 Modal 容器 × 16 线程 `aria2c` + SHA256 校验 + 断点续传）与 Hugging Face 云端拉取，绕开本地上行带宽瓶颈。 |
| **BF16 装不下 48 GB 卡** | 提供一键云端 [`quantize_fp8.py`](modal-deploy/quantize_fp8.py)，在按秒计费的 A100-80GB 上几分钟完成 `FP8_DYNAMIC` 压缩，把 54 GB 权重压到 ~27–29 GB 跑进单张 L40S。 |
| **上线前容易踩格式坑** | 提供零依赖权重预检脚本 [`inspect_weights.py`](modal-deploy/inspect_weights.py) 与上线 4 项验收脚本 [`smoke_test.py`](modal-deploy/smoke_test.py)，无需安装 PyTorch 即可校验分片、量化配置与模板标签。 |

---

## 快速开始（SGLang + L40S 主线）

前置条件：Python 3.10+、已注册的 [Modal](https://modal.com) 账号（含免费试用额度）。

### 1. 安装并登录 Modal CLI

```bash
pip install modal
modal setup
```

### 2. （推荐）本地预检模型权重

如果权重在本地，先用零依赖脚本检查分片完整性、FP8 元数据以及 `<think>` / `<tool_call>` 聊天模板：

```bash
python modal-deploy/inspect_weights.py /path/to/your-fp8-model-dir
```

### 3. 创建鉴权密钥与权重存储卷

服务默认**强制校验 Bearer API Key**，缺失 Secret 时拒绝启动，避免暴露裸奔 GPU 接口：

```bash
modal secret create qwen-api-key SGLANG_API_KEY="sk-replace-with-your-strong-secret"
modal volume create qwen38-27b-weights
```

### 4. 将权重同步到 Modal Volume（三选一）

- **方式 A：ModelScope 云端并发拉取（推荐，28.8 GiB 约 2 分钟完成）**
  ```bash
  modal run modal-deploy/serve_sglang.py \
    --model-id your-org/your-fp8-model \
    --target-subdir qwen38-27b-fp8 \
    --token your-modelscope-token
  ```
  > 公开模型可省略 `--token`。脚本会在本地拉取文件清单后，在 Modal 云端为每个缺失分片并发启动容器，使用 16 连接 `aria2c` 下载并逐文件校验大小与 SHA256。

- **方式 B：Hugging Face 云端拉取**
  ```bash
  modal run modal-deploy/serve_sglang.py::download_from_hf \
    --repo-id your-org/your-fp8-repo \
    --target-subdir qwen38-27b-fp8 \
    --token hf_your_token
  ```

- **方式 C：本地目录直接上传**
  ```bash
  modal volume put qwen38-27b-weights ./local-fp8-dir /qwen38-27b-fp8
  ```

> **如果手上只有 BF16 权重（~54 GB）**：先上传到 `/qwen38-27b-bf16`，再运行云端一次性量化任务转为 FP8：
> ```bash
> modal run modal-deploy/quantize_fp8.py \
>   --input-subdir qwen38-27b-bf16 \
>   --output-subdir qwen38-27b-fp8
> ```

### 5. 配置模型名并部署

如果使用自定义目录名或对外暴露的模型 ID，可复制模板文件 [`modal-deploy/.env.example`](modal-deploy/.env.example) 为 `modal-deploy/.env.local`（已被 `.gitignore` 忽略），或通过环境变量传入：

```bash
cp modal-deploy/.env.example modal-deploy/.env.local
# 按需修改 SGLANG_MODEL_SUBDIR 与 SGLANG_SERVED_MODEL_NAME
modal deploy modal-deploy/serve_sglang.py
```

部署成功后，Modal 会输出 HTTPS 端点（形如 `https://<workspace>--qwen38-27b-sglang-sglangserver-serve.modal.run`）。

### 6. 全链路验收（并触发首次 GPU 显存快照）

```bash
python modal-deploy/smoke_test.py \
  --url https://<workspace>--qwen38-27b-sglang-sglangserver-serve.modal.run \
  --api-key "sk-replace-with-your-strong-secret" \
  --model qwen3.8-27b-ft
```

脚本依次验证：
1. 未携带 API Key 访问 `/v1/models` 返回 `401/403`（首次调用会构建并保存 GPU 显存快照）
2. 携带正确 Key 获取 `/v1/models`
3. 流式 `/v1/chat/completions`（报告 TTFT、端到端耗时、`<think>` 推理内容与正文）
4. OpenAI Function / Tool Calling 参数结构化解析

接入任意 OpenAI 兼容客户端时，将 **Base URL** 设为 `https://<你的地址>.modal.run/v1`，**HTTP 超时**建议设为 `≥ 120s` 以兼容冷启动恢复。

---

## 冷启动加速架构

[`modal-deploy/serve_sglang.py`](modal-deploy/serve_sglang.py) 组合了 6 层优化：

1. **Modal CPU + GPU 显存快照**（`enable_memory_snapshot=True`、`experimental_options={"enable_gpu_snapshot": True}`、`@modal.enter(snap=True)`）：首次部署启动后保存完整进程内存与显存状态；缩容到 0 后再次被请求唤醒时直接还原显存快照，跳过读盘、模型反序列化与 CUDA 初始化。
2. **快照前强制 Warmup**（`_wait_and_warmup_sglang`）：在快照落盘前轮询 `/health` 并发送一条 `max_tokens=4` 的本地推理请求，确保 FlashInfer / Triton / CUDA 算子在快照前完成编译加载，消除恢复后的首包编译卡顿。
3. **持久化 JIT 缓存卷**（`qwen38-27b-sglang-cache` → `/cache`）：将 `FLASHINFER_WORKSPACE_BASE`、`TRITON_CACHE_DIR`、`TORCHINDUCTOR_CACHE_DIR` 全部持久化并在预热后 `commit()`，即使快照失效或升级镜像也无需重编算子。
4. **裁剪小 batch decode CUDA Graph**（`--cuda-graph-max-bs-decode 4` + `--cuda-graph-backend-prefill=disabled`）：避免大 batch 建图拖慢快照与吃显存，同时保留高吞吐 decode，快照恢复零额外开销。
5. **FP8 单卡装入 + 同区域 Volume**：28.8 GiB 权重直接装入单张 48 GB L40S，免除多卡切分与 NCCL 初始化，且零公网拉取。
6. **60 秒保活窗口**（`SCALEDOWN_WINDOW_SECONDS = 60`）：单轮多步工具调用或连续追问直接命中热容器（实测流式 TTFT ~0.4–0.8s），空闲 60 秒后自动归零止损。

> **实测参考**（Modal `us-east` 单卡 `L40S`，28.77 GiB FineGrained FP8 27B 模型）：首次构建快照约 `140–171s`，快照恢复冷启动约 `29s`，热态流式 TTFT `0.4–0.8s`。

---

## 仓库结构与引擎选型

```text
.
├── modal-deploy/
│   ├── serve_sglang.py     # 主力方案：Modal L40S + SGLang + FP8 + GPU Snapshot + ModelScope 并发拉取
│   ├── serve_vllm.py       # 备选模板：Modal + vLLM（--enforce-eager 降低冷启动耗时）
│   ├── serve_llamacpp.py   # 备选模板：Modal L4 (24 GB) + llama-server + GGUF
│   ├── quantize_fp8.py     # 云端一次性任务：A100-80GB 上将 BF16 转为 FP8_DYNAMIC
│   ├── inspect_weights.py  # 零依赖本地权重预检（分片完整性 / 量化配置 / chat_template）
│   ├── smoke_test.py       # 零依赖上线 4 项验收脚本
│   ├── .env.example        # 本地环境变量覆盖示例
│   └── README.md           # Modal 目录速查说明
├── PDR.md                  # 附：本地主机（AMD RX 7800 XT + llama.cpp）多用户推理设计评审
├── PDP.md                  # 附：本地主机推理服务阶段部署与压测计划
├── LICENSE                 # MIT License
├── README.md               # 中文主文档
└── README_EN.md            # English Documentation
```

| 模板脚本 | 适用权重格式 | 推荐 GPU | 适用场景 |
| :--- | :--- | :--- | :--- |
| [`serve_sglang.py`](modal-deploy/serve_sglang.py) | FP8 Safetensors（~27–29 GB） | `L40S` (48 GB) | **主力推荐**：长上下文（32K）、前缀缓存复用（RadixAttention）、GPU 快照快启、Agent Tool Calling |
| [`serve_vllm.py`](modal-deploy/serve_vllm.py) | FP8 / AWQ / BF16 Safetensors | `L40S` / `L4` / `A100-80GB` | 偏好 vLLM 生态或需多精度灵活切换时使用 |
| [`serve_llamacpp.py`](modal-deploy/serve_llamacpp.py) | GGUF（如 `Q4_K_M` / `UD-IQ4_XS`） | `L4` (24 GB, ~\$0.80/hr) | 已有 GGUF 权重、追求更低单价 GPU 或与本地 `llama.cpp` 行为对齐 |

根目录附带的 [`PDR.md`](PDR.md) 与 [`PDP.md`](PDP.md) 记录了本地消费级显卡（AMD RX 7800 XT 16 GB + `llama.cpp` Vulkan）自建方案的设计与压测对照，供在「云端弹性部署 vs 本地常驻部署」之间做架构选型参考。

---

## 常用调优与成本护栏

关键参数均集中在 [`modal-deploy/serve_sglang.py`](modal-deploy/serve_sglang.py) 顶部：

- **`SGLANG_MODEL_SUBDIR` / `SGLANG_SERVED_MODEL_NAME`**：支持通过环境变量、`modal-deploy/.env.local` 或 `PROFILE_MODELS` 映射覆盖，无需把私有模型名写进 Git 仓库。
- **`SCALEDOWN_WINDOW_SECONDS`**（默认 `60`）：交互间隔较长且希望减少冷启次数时可调至 `180`–`300`；极致省钱可调低至 `30`。
- **`MAX_CONTAINERS`**（默认 `1`）：硬性单卡上限，防止客户端重试风暴拉起多卡导致费用失控。
- **`CONTEXT_LENGTH` / `MEM_FRACTION_STATIC`**（默认 `32768` / `0.82`）：可根据模型实际架构（如混合线性注意力状态缓冲）与并发需求调整。
- **平台侧账单保护**：建议在 Modal 控制台 **Settings → Billing** 设置月度消费上限（Spending Limit）。
- **多账号/多 Profile 切换**：配合 `MODAL_PROFILE=<profile>` 可将同一套脚本独立部署到不同 Modal Workspace，并在 `PROFILE_MODELS` 中按 Profile 映射不同模型。

---

## License

本项目采用 [MIT License](LICENSE) 开源。
