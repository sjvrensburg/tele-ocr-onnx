"""Export `embed_tokens{suffix}.onnx`: input_ids int64 [B, S] -> inputs_embeds float32 [B, S, 1024].

The output is float32 for every variant because transformers.js splices float32 image features into
it with a raw typed-array copy. Variants:
    ""        fp32 Gather
    "_fp16"   fp16 table + Cast
    "_q4f16"  int4 table (block 32, asymmetric, fp16 scales) via com.microsoft.GatherBlockQuantized
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper, numpy_helper
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "TeleOCR"
sys.path.insert(0, str(Path(__file__).parent))
from onnx_utils import file_size_mb, save_chunked  # noqa: E402

BLOCK = 32


def quantize_int4(w: np.ndarray, block: int = BLOCK):
    """Asymmetric blockwise uint4 along the last axis. Returns packed data, fp16 scales, packed zero points."""
    rows, cols = w.shape
    blocks = w.reshape(rows, cols // block, block)
    lo, hi = blocks.min(-1), blocks.max(-1)
    scale = np.maximum(hi - lo, 1e-8) / 15.0
    scale16 = scale.astype(np.float16)
    scale = scale16.astype(np.float32)
    zp = np.clip(np.round(-lo / scale), 0, 15).astype(np.uint8)
    q = np.clip(np.round(blocks / scale[..., None]) + zp[..., None], 0, 15).astype(np.uint8).reshape(rows, cols)
    pack = lambda a: (a[..., 0::2] | (a[..., 1::2] << 4)).astype(np.uint8)  # element 0 -> low nibble
    return pack(q), scale16, pack(zp)


def build(variant: str) -> onnx.ModelProto:
    f = safe_open(str(MODEL_DIR / "model.safetensors"), "pt")
    table = f.get_tensor("model.embed_tokens.weight").float().numpy()  # [151936, 1024]
    nodes, inits = [], []
    if variant == "":
        inits.append(numpy_helper.from_array(table, "embed_tokens.weight"))
        nodes.append(helper.make_node("Gather", ["embed_tokens.weight", "input_ids"], ["inputs_embeds"], name="/embed/Gather"))
    elif variant == "_fp16":
        inits.append(numpy_helper.from_array(table.astype(np.float16), "embed_tokens.weight"))
        nodes.append(helper.make_node("Gather", ["embed_tokens.weight", "input_ids"], ["/embed/out"], name="/embed/Gather"))
        nodes.append(helper.make_node("Cast", ["/embed/out"], ["inputs_embeds"], name="/embed/Cast", to=TP.FLOAT))
    elif variant == "_q4f16":
        data, scales, zps = quantize_int4(table)
        inits += [numpy_helper.from_array(data, "embed_tokens.qweight"),
                  numpy_helper.from_array(scales, "embed_tokens.scales"),
                  numpy_helper.from_array(zps, "embed_tokens.zero_points")]
        nodes.append(helper.make_node(
            "GatherBlockQuantized", ["embed_tokens.qweight", "input_ids", "embed_tokens.scales", "embed_tokens.zero_points"],
            ["/embed/out"], name="/embed/GatherBlockQuantized", domain="com.microsoft",
            gather_axis=0, quantize_axis=1, block_size=BLOCK, bits=4))
        nodes.append(helper.make_node("Cast", ["/embed/out"], ["inputs_embeds"], name="/embed/Cast", to=TP.FLOAT))
    else:
        raise ValueError(variant)
    graph = helper.make_graph(
        nodes, "teleocr_embed_tokens",
        [helper.make_tensor_value_info("input_ids", TP.INT64, ["batch_size", "sequence_length"])],
        [helper.make_tensor_value_info("inputs_embeds", TP.FLOAT, ["batch_size", "sequence_length", table.shape[1]])],
        inits,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
                             producer_name="tele-ocr-onnx")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "output" / "onnx"))
    ap.add_argument("--suffix", nargs="*", default=["", "_fp16", "_q4f16"])
    args = ap.parse_args()
    for suffix in args.suffix:
        path = Path(args.out) / f"embed_tokens{suffix}.onnx"
        n = save_chunked(build(suffix), path)
        print(f"wrote {path} ({file_size_mb(path):.0f} MB, {n} data chunk(s))")


if __name__ == "__main__":
    main()
