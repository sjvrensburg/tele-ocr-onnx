"""Convert golden/*.npz into raw little-endian .bin files + meta.json for the Node/WebGPU harness.

Also precomputes the exact fp32 merged `inputs_embeds` (text embeddings with golden image features
spliced in at <|image_pad|> positions) so decoder variants can be checked in isolation.
"""
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
IMAGE_TOKEN = 151655


def main():
    table = safe_open(str(ROOT / "models/TeleOCR/model.safetensors"), "pt").get_tensor("model.embed_tokens.weight").float().numpy()
    out = ROOT / "golden" / "bin"
    out.mkdir(exist_ok=True)
    meta = {}
    for npz in sorted((ROOT / "golden").glob("*.npz")):
        d = dict(np.load(npz))
        ids = d["input_ids"]
        emb = table[ids].copy()
        emb[ids == IMAGE_TOKEN] = d["image_features"]
        d["inputs_embeds"] = emb.astype(np.float32)
        entry = {}
        for k, v in d.items():
            v = np.ascontiguousarray(v)
            (out / f"{npz.stem}.{k}.bin").write_bytes(v.tobytes())
            entry[k] = {"dtype": str(v.dtype), "shape": list(v.shape)}
        meta[npz.stem] = entry
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("wrote", out, list(meta))


if __name__ == "__main__":
    main()
