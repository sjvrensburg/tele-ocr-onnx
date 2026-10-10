"""Export the TeleOCR language model as `decoder_model_merged{suffix}.onnx` for transformers.js / ORT WebGPU.

TeleOCR's decoder is NOT the Qwen2.5-VL decoder, although config.json says `qwen2_5_vl`:

  * attention is Qwen3-style: per-head RMSNorm on Q and K (`q_norm`, `k_norm`) applied before RoPE,
    no Q/K/V bias, and head_dim=128 decoupled from hidden_size=1024 (q_proj is 1024 -> 2048);
  * positions use Qwen2.5-VL's *sectioned* 3D MRoPE (mrope_section=[16, 24, 24]), not Qwen3-VL's
    interleaved layout.

onnxruntime-genai's builder has each half (Qwen25VLTextModel / Qwen3VLTextModel) but not the
combination, and it would load weights through the *stock* transformers Qwen2.5-VL class, which
cannot represent this checkpoint. `TeleOCRTextModel` below fixes both.
"""
from __future__ import annotations

import argparse
import importlib
import re
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import transformers

ROOT = Path(__file__).resolve().parents[1]
MAX_POSITIONS = 32768
MODEL_DIR = ROOT / "models" / "TeleOCR"
sys.path.insert(0, str(Path(__file__).parent))
from onnx_utils import file_size_mb, load_full, save_chunked  # noqa: E402


def _import_builder():
    """onnxruntime-genai's builder imports transformers>=5 classes it only needs for other model
    families. TeleOCR's remote code targets transformers 4.57, so stub the missing names."""
    for _ in range(20):
        try:
            from onnxruntime_genai.models import builder
            return builder
        except ImportError as e:
            m = re.search(r"cannot import name '(\w+)' from '([\w.]+)'", str(e))
            if not m:
                raise
            name, mod = m.groups()
            setattr(importlib.import_module(mod), name, type(name, (), {}))
            for k in [k for k in sys.modules if k.startswith("onnxruntime_genai.models")]:
                del sys.modules[k]
    raise RuntimeError("could not import onnxruntime_genai builder")


gb = _import_builder()
from onnxruntime_genai.models.builders.qwen import Qwen3VLTextModel  # noqa: E402


class TeleOCRTextModel(Qwen3VLTextModel):
    """Qwen3 attention (QK-norm, fp32 norm->RoPE path) + Qwen2.5-VL sectioned MRoPE."""

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        # Qwen3VLTextModel selects the interleaved layout (1); TeleOCR uses contiguous T/H/W sections.
        self.rope_attrs["mrope_layout"] = 0
        assert self.rope_attrs["op_type"] == "MRotaryEmbedding"
        assert list(self.rope_attrs["mrope_section"]) == [16, 24, 24], self.rope_attrs["mrope_section"]
        assert self.head_size == 128 and self.num_attn_heads == 16 and self.num_kv_heads == 8
        # config says 128k positions; a 128k x 64 fp32 cos+sin cache is 65 MB of download for nothing.
        # MRoPE position ids never exceed the sequence length, so this caps prompt+output at 32k tokens.
        self.rope_attrs["cache_length"] = MAX_POSITIONS

    def load_weights(self, input_path):
        # Use the checkpoint's own modeling code (modeling_naviocr.py) so q_norm/k_norm and the
        # 2048-wide q_proj are materialised; the stock class would drop or reject them.
        model = transformers.AutoModel.from_pretrained(
            str(MODEL_DIR), trust_remote_code=True, torch_dtype=torch.float32
        )
        attn = model.model.language_model.layers[0].self_attn
        assert hasattr(attn, "q_norm") and attn.q_proj.out_features == 2048 and attn.q_proj.bias is None
        return model


def build(precision: str, out_path: Path, ep: str = "webgpu", extra: dict | None = None) -> Path:
    options = {
        "exclude_embeds": "true",  # embeddings live in embed_tokens.onnx (transformers.js layout)
        "prune_lm_head": "true",  # only the last position's logits: [B, 1, V]
        "filename": "model.onnx",
    }
    if extra:
        options.update(extra)
    with tempfile.TemporaryDirectory(dir=ROOT / "build") as tmp:
        tmp = Path(tmp)
        cache = ROOT / "build" / "cache"
        kv = gb.parse_extra_options(
            "TeleOCR", str(MODEL_DIR), str(tmp), precision, ep, str(cache),
            [f"{k}={v}" for k, v in options.items()],
        )
        kv["hf_details"] = gb.get_hf_details("TeleOCR", str(MODEL_DIR), str(cache), kv)
        cfg = kv["hf_details"]["hf_config"]
        assert cfg.architectures[0] == "Qwen2_5_VLForConditionalGeneration"

        # Route the Qwen2.5-VL architecture to our hybrid class for this build only.
        orig = gb.Qwen25VLTextModel
        gb.Qwen25VLTextModel = TeleOCRTextModel
        try:
            gb.create_model("TeleOCR", str(MODEL_DIR), str(tmp), precision, ep, str(cache), **kv)
        finally:
            gb.Qwen25VLTextModel = orig

        model = load_full(tmp / "model.onnx")
        _check_graph_io(model)
        _support_left_padding(model)
        _float32_inputs_embeds(model)
        _static_kv_head_dim(model)
        n = save_chunked(model, out_path)
        print(f"wrote {out_path} ({file_size_mb(out_path):.0f} MB, {n} data chunk(s))")
        return out_path


def _support_left_padding(model):
    """Make GroupQueryAttention honour arbitrary (left-)padding in `attention_mask`.

    The builder feeds GQA `seqlens_k = sum(attention_mask) - 1`, which is only correct when padding is
    on the right: GQA then treats the *first* sum(mask) keys as valid. transformers.js pads batches on
    the left, so instead (like onnx-community's Qwen2.5-VL export) every row is declared full-length and
    padded keys are removed with an additive attention_bias [B, 1, S, T] built from the mask.
    """
    from onnx import TensorProto, helper, numpy_helper
    import numpy as np

    g = model.graph
    io_type = next(i for i in g.input if i.name == "past_key_values.0.key").type.tensor_type.elem_type
    np_dtype = {TensorProto.FLOAT: np.float32, TensorProto.FLOAT16: np.float16}[io_type]
    p = "/model/left_padding"
    consts = {
        f"{p}/one_i64": np.array([1], dtype=np.int64),
        f"{p}/axes_12": np.array([1, 2], dtype=np.int64),
        f"{p}/idx0": np.array([0], dtype=np.int64),
        f"{p}/idx1": np.array([1], dtype=np.int64),
        f"{p}/one": np.array(1, dtype=np_dtype),
        # Finite on purpose: exp(-1e4) is exactly 0, but unlike dtype-min it cannot overflow to -inf
        # in fp16 kernels, which would turn fully-padded rows into NaN and leak via 0*NaN.
        f"{p}/neg": np.array(-1e4, dtype=np_dtype),
    }
    g.initializer.extend(numpy_helper.from_array(v, k) for k, v in consts.items())
    n = lambda op, ins, out, **kw: helper.make_node(op, ins, [f"{p}/{out}"], name=f"{p}/{out}", **kw)
    nodes = [
        n("Shape", ["attention_mask"], "mask_shape"),
        n("Gather", [f"{p}/mask_shape", f"{p}/idx0"], "batch"),  # [1]
        n("Gather", [f"{p}/mask_shape", f"{p}/idx1"], "total"),  # [1]
        n("Sub", [f"{p}/total", f"{p}/one_i64"], "total_m1"),
        n("Expand", [f"{p}/total_m1", f"{p}/batch"], "seqlens_i64"),  # [B]
        n("Cast", [f"{p}/seqlens_i64"], "seqlens_k", to=TensorProto.INT32),
        n("Shape", ["inputs_embeds"], "emb_shape"),
        n("Gather", [f"{p}/emb_shape", f"{p}/idx1"], "seq"),  # [1]
        n("Concat", [f"{p}/batch", f"{p}/one_i64", f"{p}/seq", f"{p}/total"], "bias_shape", axis=0),
        n("Cast", ["attention_mask"], "mask_f", to=io_type),
        n("Sub", [f"{p}/one", f"{p}/mask_f"], "inv_mask"),
        n("Mul", [f"{p}/inv_mask", f"{p}/neg"], "bias_2d"),  # [B, T]
        n("Unsqueeze", [f"{p}/bias_2d", f"{p}/axes_12"], "bias_4d"),  # [B, 1, 1, T]
        n("Expand", [f"{p}/bias_4d", f"{p}/bias_shape"], "attention_bias"),  # [B, 1, S, T]
    ]
    first_gqa = next(i for i, x in enumerate(g.node) if x.op_type == "GroupQueryAttention")
    for i, node in enumerate(nodes):
        g.node.insert(first_gqa + i, node)
    count = 0
    for node in g.node:
        if node.op_type == "GroupQueryAttention":
            ins = list(node.input) + [""] * (11 - len(node.input))
            ins[5] = f"{p}/seqlens_k"
            ins[10] = f"{p}/attention_bias"
            del node.input[:]
            node.input.extend(ins)
            count += 1
    assert count == 28, count


def _float32_inputs_embeds(model):
    """transformers.js splices float32 image features into `inputs_embeds` with a raw typed-array copy,
    so the graph must accept float32 regardless of compute dtype; cast inside the graph instead."""
    from onnx import TensorProto, helper

    g = model.graph
    inp = next(i for i in g.input if i.name == "inputs_embeds")
    elem = inp.type.tensor_type.elem_type
    if elem == TensorProto.FLOAT:
        return
    cast_out = "/model/inputs_embeds/Cast/output_0"
    for node in g.node:
        for j, name in enumerate(node.input):
            if name == "inputs_embeds":
                node.input[j] = cast_out
    g.node.insert(0, helper.make_node("Cast", ["inputs_embeds"], [cast_out], name="/model/inputs_embeds/Cast", to=elem))
    inp.type.tensor_type.elem_type = TensorProto.FLOAT


def _static_kv_head_dim(model):
    """The builder declares the KV head dimension as the symbol 'kv_cache_dim'. transformers.js creates the
    initial empty cache from input metadata and resolves unknown symbols to 0 (-> [B, 8, 0, 0]), which
    GroupQueryAttention rejects. Pin it to the real head size."""
    for v in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        for d in v.type.tensor_type.shape.dim:
            if d.dim_param == "kv_cache_dim":
                d.Clear()
                d.dim_value = 128


def _check_graph_io(model):
    """The builder already uses transformers.js names (inputs are bound by name); sanity-check them."""
    ins = {i.name for i in model.graph.input}
    outs = {o.name for o in model.graph.output}
    assert {"inputs_embeds", "attention_mask", "position_ids", "past_key_values.27.value"} <= ins, ins
    assert {"logits", "present.0.key", "present.27.value"} <= outs, outs


PRECISIONS = {
    # suffix: (builder precision, extra options)
    "": ("fp32", {}),
    "_fp16": ("fp16", {}),
    "_q4f16": ("int4", {"int4_block_size": "32", "int4_accuracy_level": "4"}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "output" / "onnx"))
    ap.add_argument("--suffix", nargs="*", default=list(PRECISIONS), choices=list(PRECISIONS))
    args = ap.parse_args()
    (ROOT / "build").mkdir(exist_ok=True)
    for suffix in args.suffix:
        precision, extra = PRECISIONS[suffix]
        build(precision, Path(args.out) / f"decoder_model_merged{suffix}.onnx", extra=extra)


if __name__ == "__main__":
    main()
