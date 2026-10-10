#!/usr/bin/env python3
"""TeleOCR with plain ONNX Runtime: no transformers.js, no PyTorch, no transformers.

    pip install onnxruntime tokenizers numpy pillow huggingface_hub
    python teleocr_ort.py page.png --task text
    python teleocr_ort.py table.png --task table --dtype fp32 --provider cpu

Runs the three graphs of this repo (vision_encoder, embed_tokens, decoder_model_merged) and does the
glue transformers.js would otherwise do: Qwen2-VL preprocessing (PIL bicubic smart_resize + patchify),
the chat prompt, 3D MRoPE position ids, splicing image features into the token embeddings, and a
KV-cached generation loop (greedy, with the checkpoint's repetition penalty).

Works with any ONNX Runtime build/EP: CPUExecutionProvider (all dtypes), CUDA, DirectML, WebGPU, ...
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer

REPO_ID = "stefanj0/TeleOCR-ONNX"

PROMPTS = {
    "text": "Please output the text content from the image.",
    "table": "This is the image of a table. Please output the table in OTSL format.",
    "formula": "Please write out the expression of the formula in the image using LaTeX format.",
    "code": "The image contains a code snippet, please output the parsing result.",
    "layout": "Analyze the image layout.",
    "layout_distorted": "\nMulti-point Layout Segmentation Analysis.",
    "figure": "This is a scientific figure. Please extract the table implied by this figure.",
}
# Layout prompts expect the page resized to 1036x1036 first (as in the original model card).
PRE_RESIZE = {"layout": (1036, 1036), "layout_distorted": (1036, 1036)}

PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "dml": "DmlExecutionProvider",
    "webgpu": "WebGpuExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
}


# ----------------------------------------------------------------------------------------------
# Preprocessing (identical to transformers' Qwen2VLImageProcessor with this repo's config)
# ----------------------------------------------------------------------------------------------
def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    h_bar = round(height / factor) * factor  # Python round (half to even), as in transformers
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def preprocess(image: Image.Image, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """-> pixel_values float32 [num_patches, 1176], image_grid_thw int64 [1, 3]"""
    p, m, tp = cfg["patch_size"], cfg["merge_size"], cfg["temporal_patch_size"]
    image = image.convert("RGB")
    h, w = smart_resize(image.height, image.width, p * m, cfg["min_pixels"], cfg["max_pixels"])
    if (w, h) != image.size:
        image = image.resize((w, h), Image.BICUBIC)
    x = np.asarray(image).astype(np.float64) * cfg.get("rescale_factor", 1 / 255)
    x = ((x - np.array(cfg["image_mean"])) / np.array(cfg["image_std"])).astype(np.float32)
    x = np.repeat(x.transpose(2, 0, 1)[None], tp, axis=0)  # [T=2, C, H, W]: an image is one frame, tiled
    gt, gh, gw = 1, h // p, w // p
    x = x.reshape(gt, tp, 3, gh // m, m, p, gw // m, m, p).transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return np.ascontiguousarray(x.reshape(gt * gh * gw, 3 * tp * p * p)), np.array([[gt, gh, gw]], dtype=np.int64)


def rope_positions(ids: list[int], grids: np.ndarray, cfg: dict) -> np.ndarray:
    """Qwen2.5-VL 3D (t, h, w) position ids for one sequence -> int64 [3, 1, S]."""
    m = cfg["vision_config"]["spatial_merge_size"]
    image_token = cfg["image_token_id"]
    pos = np.zeros((3, len(ids)), dtype=np.int64)
    i, nxt, img = 0, 0, 0
    while i < len(ids):
        if ids[i] == image_token:
            t, gh, gw = (int(v) for v in grids[img])
            lh, lw = gh // m, gw // m
            n = t * lh * lw
            tt, hh, ww = np.meshgrid(np.arange(t), np.arange(lh), np.arange(lw), indexing="ij")
            pos[:, i:i + n] = nxt + np.stack([tt.ravel(), hh.ravel(), ww.ravel()])
            nxt += max(t, lh, lw)
            i += n
            img += 1
        else:
            pos[:, i] = nxt
            nxt += 1
            i += 1
    return pos[:, None, :]


# ----------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------
class TeleOCR:
    def __init__(self, model_dir: str | Path, dtype: str = "q4f16", provider: str = "cpu"):
        d = Path(model_dir)
        self.cfg = json.loads((d / "config.json").read_text())
        self.pre = json.loads((d / "preprocessor_config.json").read_text())
        gen = json.loads((d / "generation_config.json").read_text())
        self.eos = set(gen["eos_token_id"] if isinstance(gen["eos_token_id"], list) else [gen["eos_token_id"]])
        self.repetition_penalty = gen.get("repetition_penalty", 1.0)
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        suffix = "" if dtype == "fp32" else f"_{dtype}"
        so = ort.SessionOptions()
        so.log_severity_level = 3
        providers = [PROVIDERS.get(provider, provider)]
        if providers[0] != "CPUExecutionProvider":
            providers.append("CPUExecutionProvider")
        load = lambda name: ort.InferenceSession(str(d / "onnx" / f"{name}{suffix}.onnx"), so, providers=providers)
        self.vision, self.embed, self.decoder = load("vision_encoder"), load("embed_tokens"), load("decoder_model_merged")
        kv = next(i for i in self.decoder.get_inputs() if i.name == "past_key_values.0.key")
        self.kv_dtype = np.float16 if "float16" in kv.type else np.float32
        self.n_layers = sum(1 for i in self.decoder.get_inputs() if i.name.endswith(".key"))
        self.n_kv_heads, self.head_dim = kv.shape[1], kv.shape[3]

    def prompt_ids(self, prompt: str, n_image_tokens: int) -> list[int]:
        # chat_template.jinja with the default system prompt and one image before the text
        text = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
                "<|vision_start|><|image_pad|><|vision_end|>" + prompt + "<|im_end|>\n<|im_start|>assistant\n")
        ids = self.tok.encode(text, add_special_tokens=False).ids
        k = ids.index(self.cfg["image_token_id"])
        return ids[:k] + [self.cfg["image_token_id"]] * n_image_tokens + ids[k + 1:]

    def __call__(self, image: Image.Image, prompt: str, max_new_tokens: int = 4096,
                 repetition_penalty: float | None = None, stream=None) -> tuple[str, dict]:
        penalty = self.repetition_penalty if repetition_penalty is None else repetition_penalty
        t0 = time.perf_counter()
        pixel_values, grid = preprocess(image, self.pre)
        feats = self.vision.run(["image_features"], {"pixel_values": pixel_values, "image_grid_thw": grid})[0]
        ids = self.prompt_ids(prompt, feats.shape[0])
        position_ids = rope_positions(ids, grid, self.cfg)

        input_ids = np.array([ids], dtype=np.int64)
        embeds = self.embed.run(["inputs_embeds"], {"input_ids": input_ids})[0]
        embeds[input_ids == self.cfg["image_token_id"]] = feats  # float32 on both sides

        feeds = {"inputs_embeds": embeds, "attention_mask": np.ones_like(input_ids), "position_ids": position_ids}
        empty = np.zeros((1, self.n_kv_heads, 0, self.head_dim), dtype=self.kv_dtype)
        for i in range(self.n_layers):
            feeds[f"past_key_values.{i}.key"] = feeds[f"past_key_values.{i}.value"] = empty
        out_names = [o.name for o in self.decoder.get_outputs()]
        next_pos = int(position_ids.max()) + 1
        seen = set(ids)
        generated: list[int] = []
        t_first = None
        while len(generated) < max_new_tokens:
            outs = dict(zip(out_names, self.decoder.run(out_names, feeds)))
            logits = outs["logits"][0, -1].astype(np.float32)
            if penalty != 1.0:  # HF RepetitionPenaltyLogitsProcessor over prompt + generated ids
                idx = np.fromiter(seen, dtype=np.int64)
                sel = logits[idx]
                logits[idx] = np.where(sel > 0, sel / penalty, sel * penalty)
            tok = int(logits.argmax())
            if t_first is None:
                t_first = time.perf_counter()
            if tok in self.eos:
                break
            generated.append(tok)
            seen.add(tok)
            if stream:
                stream(tok)
            emb = self.embed.run(["inputs_embeds"], {"input_ids": np.array([[tok]], dtype=np.int64)})[0]
            total = len(ids) + len(generated)
            p = next_pos + len(generated) - 1
            feeds = {"inputs_embeds": emb, "attention_mask": np.ones((1, total), dtype=np.int64),
                     "position_ids": np.full((3, 1, 1), p, dtype=np.int64)}
            for i in range(self.n_layers):
                feeds[f"past_key_values.{i}.key"] = outs[f"present.{i}.key"]
                feeds[f"past_key_values.{i}.value"] = outs[f"present.{i}.value"]
        t1 = time.perf_counter()
        text = self.tok.decode(generated, skip_special_tokens=True).strip()
        stats = {"visual_tokens": int(feats.shape[0]), "prompt_tokens": len(ids), "new_tokens": len(generated),
                 "time_to_first_token_s": round(t_first - t0, 3),
                 "decode_tok_per_s": round((len(generated) - 1) / (t1 - t_first), 1) if len(generated) > 1 else None}
        return text, stats


def download(repo_id: str, dtype: str, local_dir: str | Path | None = None) -> Path:
    """Fetch configs + the chosen dtype's graphs into a plain folder (default ~/.cache/teleocr-onnx/<repo>).

    A real folder, not the symlinked HF cache: recent onnxruntime resolves symlinks and rejects external
    data that lives in a different blob directory than the .onnx file.
    """
    from huggingface_hub import snapshot_download

    suffix = "" if dtype == "fp32" else f"_{dtype}"
    graphs = [f"onnx/{n}{suffix}.onnx*" for n in ("vision_encoder", "embed_tokens", "decoder_model_merged")]
    local_dir = Path(local_dir or Path.home() / ".cache" / "teleocr-onnx" / repo_id.replace("/", "--"))
    return Path(snapshot_download(repo_id, allow_patterns=["*.json", *graphs], local_dir=local_dir))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--task", choices=list(PROMPTS), default="text")
    ap.add_argument("--prompt", help="custom prompt (overrides --task)")
    ap.add_argument("--model", default=REPO_ID, help="local folder or Hub repo id")
    ap.add_argument("--download-dir", help="where Hub files go (default ~/.cache/teleocr-onnx/<repo>)")
    ap.add_argument("--dtype", choices=["q4f16", "fp16", "fp32"], default="q4f16")
    ap.add_argument("--provider", default="cpu", help="cpu | cuda | dml | webgpu | coreml | <full EP name>")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--repetition-penalty", type=float, default=None, help="default: generation_config.json (1.05)")
    ap.add_argument("--json", action="store_true", help="print {text, stats} as JSON instead of streaming")
    args = ap.parse_args()

    model_dir = Path(args.model) if Path(args.model).is_dir() else download(args.model, args.dtype, args.download_dir)
    model = TeleOCR(model_dir, args.dtype, args.provider)
    image = Image.open(args.image)
    if args.task in PRE_RESIZE and not args.prompt:
        image = image.convert("RGB").resize(PRE_RESIZE[args.task], Image.BICUBIC)

    tokens, shown = [], ""

    def stream(tok):  # print decoded text incrementally
        nonlocal shown
        tokens.append(tok)
        s = model.tok.decode(tokens, skip_special_tokens=True)
        if not s.endswith("\ufffd"):  # wait until multi-byte characters are complete
            sys.stdout.write(s[len(shown):])
            sys.stdout.flush()
            shown = s

    text, stats = model(image, args.prompt or PROMPTS[args.task], args.max_new_tokens, args.repetition_penalty,
                        stream=None if args.json else stream)
    if args.json:
        print(json.dumps({"text": text, "stats": stats}, ensure_ascii=False, indent=1))
    else:
        print(f"\n\n{stats}", file=sys.stderr)


if __name__ == "__main__":
    main()
