"""Verify ort/teleocr_ort.py (plain ONNX Runtime pipeline) against the PyTorch golden data.

1. glue: pixel_values, prompt ids and 3D position ids must equal what transformers produced
2. generation: greedy output (repetition_penalty=1.0, like the golden run) per dtype vs PyTorch text
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ort"))
from teleocr_ort import PRE_RESIZE, PROMPTS, TeleOCR, preprocess, rope_positions  # noqa: E402

FILES = {"text": "text.png", "formula": "formula.png", "table": "table.png", "layout": "layout.jpg"}


def load_image(name):
    im = Image.open(ROOT / "models/TeleOCR/assets" / FILES[name])
    if name in PRE_RESIZE:
        im = im.convert("RGB").resize(PRE_RESIZE[name], Image.BICUBIC)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "output"))
    ap.add_argument("--dtype", nargs="*", default=["fp32", "fp16", "q4f16"])
    ap.add_argument("--samples", nargs="*", default=list(FILES))
    ap.add_argument("--provider", default="cpu")
    args = ap.parse_args()
    golden = json.loads((ROOT / "golden/summary.json").read_text())
    ok = True

    model = TeleOCR(args.model, args.dtype[0], args.provider)
    for name in args.samples:
        g = np.load(ROOT / f"golden/{name}.npz")
        pv, grid = preprocess(load_image(name), model.pre)
        ids = model.prompt_ids(PROMPTS[name], int(grid.prod()) // 4)
        pos = rope_positions(ids, grid, model.cfg)
        checks = {
            "pixel_values": pv.shape == g["pixel_values"].shape and np.abs(pv - g["pixel_values"]).max() < 1e-5,
            "grid": np.array_equal(grid, g["image_grid_thw"]),
            "input_ids": ids == g["input_ids"][0].tolist(),
            "position_ids": np.array_equal(pos, g["position_ids"]),
        }
        ok &= all(checks.values())
        print(f"glue {name:8s} " + " ".join(f"{k}={'OK' if v else 'MISMATCH'}" for k, v in checks.items()))

    for dtype in args.dtype:
        model = TeleOCR(args.model, dtype, args.provider)
        for name in args.samples:
            ref = golden[name]
            text, stats = model(load_image(name), PROMPTS[name], max_new_tokens=ref["new_tokens"], repetition_penalty=1.0)
            exact = text == ref["text"].strip()
            print(f"gen  {dtype:6s} {name:8s} exact={exact}  {stats}")
            if not exact:
                a, b = text, ref["text"].strip()
                i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                print(f"     first diff at char {i}: ort={a[max(0, i - 10):i + 10]!r} torch={b[max(0, i - 10):i + 10]!r}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
