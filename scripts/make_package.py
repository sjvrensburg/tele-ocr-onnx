"""Assemble the transformers.js model folder in output/: configs, tokenizer, chat template, and the
`transformers.js_config` block (external-data chunk counts per ONNX file, fp16 KV cache)."""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "models" / "TeleOCR"
OUT = ROOT / "output"

COPY = [
    "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json", "chat_template.jinja",
]


def main():
    for name in COPY:
        shutil.copy2(SRC / name, OUT / name)

    # TeleOCR's preprocessor_config.json relies on Qwen2VLImageProcessor's Python class defaults, which
    # transformers.js does not share (it would skip resizing and normalization and resample bilinearly).
    # Spell the Python defaults out; this does not change Python behaviour.
    pre = json.loads((SRC / "preprocessor_config.json").read_text())
    for k, v in {"do_resize": True, "resample": 3, "do_rescale": True, "rescale_factor": 1 / 255,
                 "do_normalize": True, "do_convert_rgb": True}.items():
        pre.setdefault(k, v)
    (OUT / "preprocessor_config.json").write_text(json.dumps(pre, indent=2) + "\n")

    # transformers.js' Qwen2-VL processor only reads the chat template from tokenizer_config.json.
    tok_cfg = json.loads((SRC / "tokenizer_config.json").read_text())
    tok_cfg["chat_template"] = (SRC / "chat_template.jinja").read_text()
    (OUT / "tokenizer_config.json").write_text(json.dumps(tok_cfg, indent=2, ensure_ascii=False) + "\n")

    # Model card + the browser preprocessing helper (bit-exact PIL bicubic resize, see README).
    shutil.copy2(ROOT / "docs" / "MODEL_CARD.md", OUT / "README.md")
    (OUT / "js").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "web" / "demo" / "pil_resize.js", OUT / "js" / "pil_resize.js")
    # Standalone plain-ONNX-Runtime pipeline (Python).
    (OUT / "ort").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "ort" / "teleocr_ort.py", OUT / "ort" / "teleocr_ort.py")

    config = json.loads((SRC / "config.json").read_text())
    # The custom PyTorch code is not shipped (and not needed by ONNX runtimes).
    config.pop("auto_map", None)
    chunks = {}
    for onnx_file in sorted((OUT / "onnx").glob("*.onnx")):
        n = len(list(onnx_file.parent.glob(onnx_file.name + "_data*")))
        if n:
            chunks[onnx_file.name] = n
    config["transformers.js_config"] = {
        "use_external_data_format": chunks,
        "kv_cache_dtype": {"fp16": "float16", "q4f16": "float16"},
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config["transformers.js_config"], indent=2))


if __name__ == "__main__":
    main()
