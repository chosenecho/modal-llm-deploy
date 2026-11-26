"""
Zero-dependency local preflight inspector for Hugging Face Safetensors weights.
Run this on your merged FP8 (or BF16) model directory before deploying to Modal + SGLang.

Usage:
  python inspect_weights.py <path-to-model-dir>
"""

import json
import os
import sys
from pathlib import Path


def format_gb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GiB"


def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def inspect_dir(model_dir: Path) -> int:
    if not model_dir.is_dir():
        print(f"[ERROR] Directory not found: {model_dir}")
        return 1

    print(f"=== Inspecting Model Directory: {model_dir.resolve()} ===\n")

    # 1. File inventory & total size
    safetensors_files = sorted(model_dir.glob("*.safetensors"))
    all_files = [p for p in model_dir.rglob("*") if p.is_file()]
    total_bytes = sum(p.stat().st_size for p in all_files)
    weights_bytes = sum(p.stat().st_size for p in safetensors_files)

    print(f"Total files:       {len(all_files)} ({format_gb(total_bytes)})")
    print(f"Safetensors count: {len(safetensors_files)} ({format_gb(weights_bytes)})")
    if not safetensors_files:
        print("[WARN] No .safetensors files found in root directory!")

    # 2. config.json check
    config = load_json(model_dir / "config.json")
    if config is None:
        print("\n[ERROR] Missing config.json — required by SGLang.")
        return 1

    archs = config.get("architectures", [])
    model_type = config.get("model_type")
    torch_dtype = config.get("torch_dtype") or config.get("dtype")
    max_pos = config.get("max_position_embeddings") or (
        config.get("text_config", {}).get("max_position_embeddings")
    )
    quant_cfg = config.get("quantization_config")
    has_vision = "vision_config" in config or any("VL" in str(a) for a in archs)

    print("\n--- config.json ---")
    print(f"architectures:           {archs}")
    print(f"model_type:              {model_type}")
    print(f"torch_dtype:             {torch_dtype}")
    print(f"max_position_embeddings: {max_pos}")
    print(f"multimodal / VL:         {has_vision}")
    print(f"quantization_config:     {json.dumps(quant_cfg, ensure_ascii=False) if quant_cfg else 'None (unquantized / BF16?)'}")

    # 3. Check index file consistency if sharded
    index_json = load_json(model_dir / "model.safetensors.index.json")
    if index_json and "weight_map" in index_json:
        expected_shards = sorted(set(index_json["weight_map"].values()))
        actual_shards = sorted(p.name for p in safetensors_files)
        missing = sorted(set(expected_shards) - set(actual_shards))
        print(f"\nIndex shards expected: {len(expected_shards)}, present: {len(actual_shards)}")
        if missing:
            print(f"[ERROR] Missing safetensors shards: {missing}")
            return 1

    # 4. Tokenizer & chat_template inspection
    tok_cfg = load_json(model_dir / "tokenizer_config.json") or {}
    chat_template = tok_cfg.get("chat_template")
    if not chat_template:
        jinja_path = model_dir / "chat_template.jinja"
        if jinja_path.is_file():
            chat_template = jinja_path.read_text(encoding="utf-8")

    print("\n--- tokenizer / template ---")
    print(f"tokenizer_class:   {tok_cfg.get('tokenizer_class')}")
    print(f"chat_template:     {'present (' + str(len(chat_template)) + ' chars)' if isinstance(chat_template, str) else type(chat_template).__name__}")

    tpl_text = chat_template if isinstance(chat_template, str) else json.dumps(chat_template or "")
    has_think = "<think>" in tpl_text
    has_xml_tool = "<function=" in tpl_text or "<parameter=" in tpl_text
    has_json_tool = "<tool_call>" in tpl_text

    print(f"Detected <think> tags:        {has_think}")
    print(f"Detected <tool_call> tags:    {has_json_tool}")
    print(f"Detected XML <function=> tag: {has_xml_tool}")

    # 5. Deployment recommendations for Modal + SGLang
    print("\n--- Modal + SGLang Recommendations ---")
    weight_gib = weights_bytes / (1024 ** 3)
    if quant_cfg:
        quant_method = quant_cfg.get("quant_method") or quant_cfg.get("format") or "unknown"
        print(f"* Pre-quantized checkpoint detected (method={quant_method}).")
        print("  -> Keep QUANTIZATION = None in serve_sglang.py (SGLang auto-detects from config.json).")
    else:
        print("* No quantization_config in config.json.")
        if weight_gib > 40:
            print("  -> Looks like BF16 (~54 GiB). Recommended: run quantize_fp8.py to convert to FP8 (~27 GiB) before serving on L40S.")
        else:
            print("  -> Check whether FP8 metadata was stripped during export.")

    if weight_gib <= 35:
        print(f"* Weight size ({weight_gib:.1f} GiB) fits on Modal GPU_TYPE = 'L40S' (48 GB).")
    else:
        print(f"* Weight size ({weight_gib:.1f} GiB) exceeds 48 GB L40S; quantize to FP8 or use A100-80GB / H100.")

    if has_xml_tool:
        print("* Tool format looks like Qwen3 XML (<function=...>) -> verify SGLang --tool-call-parser (e.g. qwen3_coder / qwen).")
    elif has_json_tool:
        print("* Tool format looks like <tool_call> JSON -> try TOOL_CALL_PARSER = 'qwen25' in serve_sglang.py.")

    if has_think:
        print("* Reasoning tags (<think>) detected -> consider REASONING_PARSER = 'qwen3' if you want reasoning_content separated.")

    est_storage_usd = (total_bytes / (1024 ** 3)) * 0.15
    print(f"* Estimated Modal Volume storage cost: ~${est_storage_usd:.2f}/month.")

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_weights.py <path-to-model-dir>")
        sys.exit(2)
    sys.exit(inspect_dir(Path(sys.argv[1])))
