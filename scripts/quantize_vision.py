"""Build vision_encoder_q4f16.onnx: int4 (MatMulNBits, block 32) weights on top of the fp16 graph.

--keep-attn leaves the attention projections (qkv, proj) in fp16 and quantizes only the MLPs/merger.
"""
import argparse
import sys
from pathlib import Path

from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from onnx_utils import file_size_mb, load_full, save_chunked  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "output/onnx/vision_encoder_fp16.onnx"))
    ap.add_argument("--dst", default=str(ROOT / "output/onnx/vision_encoder_q4f16.onnx"))
    ap.add_argument("--keep-attn", action="store_true")
    ap.add_argument("--keep-patch-embed", action="store_true", default=True)
    args = ap.parse_args()

    model = load_full(args.src)
    exclude = []
    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue
        w = node.input[1]
        if (args.keep_patch_embed and w.startswith("visual.patch_embed")) or \
           (args.keep_attn and (".attn.qkv." in w or ".attn.proj." in w)):
            exclude.append(node.name)
    cfg = DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=False, accuracy_level=4)
    q = MatMulNBitsQuantizer(model, nodes_to_exclude=exclude, algo_config=cfg)
    q.process()
    out = q.model.model
    n = save_chunked(out, args.dst)
    nbits = sum(1 for nd in out.graph.node if nd.op_type == "MatMulNBits")
    print(f"wrote {args.dst} ({file_size_mb(args.dst):.0f} MB, {n} chunk(s)); {nbits} MatMulNBits, {len(exclude)} MatMuls kept fp16")


if __name__ == "__main__":
    main()
