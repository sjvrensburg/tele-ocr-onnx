"""Check vision_encoder*.onnx against the PyTorch TeleOCR vision tower.

Cases: the golden images (real processor output) and synthetic grids, including grids that are not
multiples of the 8-patch window and a multi-image batch with different sizes.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[1]


def cos_sim(a, b):
    a, b = a.reshape(len(a), -1).astype(np.float64), b.reshape(len(b), -1).astype(np.float64)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "output/onnx/vision_encoder.onnx"))
    ap.add_argument("--min-cos", type=float, default=0.9999)
    args = ap.parse_args()

    hf = AutoModel.from_pretrained(ROOT / "models/TeleOCR", trust_remote_code=True, dtype=torch.float32,
                                   attn_implementation="eager").eval()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(0)
    cases = []
    for name in ["text", "formula", "table", "layout"]:
        d = np.load(ROOT / f"golden/{name}.npz")
        cases.append((name, d["pixel_values"], d["image_grid_thw"], d["image_features"]))
    for grids in ([[1, 6, 10]], [[1, 18, 2]], [[1, 2, 2]], [[1, 10, 6], [1, 4, 22], [1, 16, 16]]):
        grid = np.array(grids, dtype=np.int64)
        n = int(grid.prod(1).sum())
        pv = rng.standard_normal((n, 1176)).astype(np.float32)
        cases.append((f"synthetic{grids}", pv, grid, None))

    ok = True
    for name, pv, grid, golden in cases:
        out = sess.run(["image_features"], {"pixel_values": pv, "image_grid_thw": grid})[0]
        with torch.no_grad():
            ref = torch.cat(list(hf.model.get_image_features(torch.from_numpy(pv), torch.from_numpy(grid))), 0).numpy()
        c = cos_sim(out, ref)
        err = np.abs(out - ref).max() / np.abs(ref).max()
        good = out.shape == ref.shape and c.min() >= args.min_cos
        ok &= good
        extra = f"  vs golden max|d|={np.abs(golden - ref).max():.2e}" if golden is not None else ""
        print(f"{'OK ' if good else 'BAD'} {name:42s} {str(out.shape):14s} min_cos={c.min():.6f} mean_cos={c.mean():.6f} "
              f"rel_max_err={err:.2e}{extra}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
