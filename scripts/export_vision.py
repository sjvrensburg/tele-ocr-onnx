"""Hand-built ONNX graph for TeleOCR's vision tower (Qwen2.5-VL ViT + 2x2 patch merger).

Inputs  (transformers.js `Qwen2_5_VLForConditionalGeneration.encode_image` contract):
    pixel_values    float32 [num_patches, 1176]   (Qwen2VLImageProcessor output, processor order)
    image_grid_thw  int64   [num_images, 3]
Output:
    image_features  float32 [num_patches / 4, 1024]

The PyTorch implementation is export-hostile (python loops over grid_thw, .tolist(), per-window
torch.split, and an argsort'ed window permutation). This graph computes the same function differently:

* No token permutation. Attention with explicit RoPE positions is permutation-equivariant, and the
  merger's 2x2 groups are already contiguous in processor order, so we never reorder the sequence.
  Instead every patch gets a (group, slot) address computed arithmetically from image_grid_thw:
    - window layers: group = 112px window (4x4 merged tokens = 64 patches) of its image
    - full layers (7, 15, 23, 31): group = its image (capacity = largest image)
  q/k/v are gathered into padded [groups, capacity, 3*1280] batches, attended with
  com.microsoft.MultiHeadAttention + key_padding_mask, and gathered back to patch order.
* All index math is int32 with ops that have WebGPU kernels (the WebGPU EP has no int64 kernels and
  no int32 Mod/ScatterND); `a % b` is written as `a - (a / b) * b`.
* Images only: each image must have t == 1 (always true for Qwen2VLImageProcessor images; video
  frames would need per-frame grouping).
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
from onnx_utils import clone_saved, file_size_mb, save_chunked  # noqa: E402

HIDDEN, HEADS, HEAD_DIM, DEPTH = 1280, 16, 80, 32
FULLATT = {7, 15, 23, 31}
MERGE = 2  # spatial_merge_size
WIN = 4  # window_size / spatial_merge_size / patch_size = 112 / 2 / 14 merged tokens per window side
WIN_CAP = (WIN * MERGE) ** 2  # 64 patches per window
OUT_HIDDEN = 1024
EPS = 1e-6


class G:
    """Tiny graph builder: unique names, constant caching, typed initializers."""

    def __init__(self, dtype):
        self.dtype = dtype  # TensorProto elem type for activations/weights
        self.np_dtype = {TP.FLOAT: np.float32, TP.FLOAT16: np.float16}[dtype]
        self.nodes, self.inits, self._consts, self._n = [], [], {}, 0

    def op(self, op_type, inputs, n_out=1, domain="", name=None, **attrs):
        self._n += 1
        base = f"/{name or op_type}_{self._n}"
        outs = [f"{base}/out{i}" for i in range(n_out)]
        self.nodes.append(helper.make_node(op_type, inputs, outs, name=base, domain=domain, **attrs))
        return outs[0] if n_out == 1 else outs

    def if_(self, cond, then_fn, else_fn, out_type, name="If"):
        """ONNX If: each fn builds nodes (may read outer-scope tensors) and returns its output name."""
        branches = []
        for tag, fn in (("then", then_fn), ("else", else_fn)):
            saved, self.nodes = self.nodes, []
            out = fn()
            # A branch output must be produced inside the branch.
            self._n += 1
            res = f"/{name}_{tag}_{self._n}/out"
            self.nodes.append(helper.make_node("Identity", [out], [res], name=f"/{name}_{tag}_{self._n}/Identity"))
            branches.append(helper.make_graph(self.nodes, f"{name}_{tag}_{self._n}", [],
                                              [helper.make_tensor_value_info(res, out_type, None)]))
            self.nodes = saved
        return self.op("If", [cond], name=name, then_branch=branches[0], else_branch=branches[1])

    def const(self, value, dtype):
        arr = np.array(value, dtype=dtype)
        key = (arr.dtype.str, arr.shape, arr.tobytes())
        if key not in self._consts:
            name = f"/const/{len(self._consts)}"
            self.inits.append(numpy_helper.from_array(arr, name))
            self._consts[key] = name
        return self._consts[key]

    def i32(self, v):
        return self.const(v, np.int32)

    def i64(self, v):
        return self.const(v, np.int64)

    def weight(self, name, arr, dtype=None):
        self.inits.append(numpy_helper.from_array(np.ascontiguousarray(arr.astype(dtype or self.np_dtype)), name))
        return name

    # int32 helpers --------------------------------------------------------------------------
    def div(self, a, b):
        return self.op("Div", [a, b])

    def mod(self, a, b):  # non-negative operands only
        return self.op("Sub", [a, self.op("Mul", [self.div(a, b), b])])

    def add(self, a, b):
        return self.op("Add", [a, b])

    def sub(self, a, b):
        return self.op("Sub", [a, b])

    def mul(self, a, b):
        return self.op("Mul", [a, b])

    def gather(self, data, idx, axis=0):
        return self.op("Gather", [data, idx], axis=axis)

    def owner(self, idx, ends):
        """For each value in idx [M], the number of entries of ends [I] that are <= it (= its segment)."""
        ge = self.op("GreaterOrEqual", [self.op("Unsqueeze", [idx, self.i64([1])]), self.op("Unsqueeze", [ends, self.i64([0])])])
        return self.op("ReduceSum", [self.op("Cast", [ge], to=TP.INT32), self.i64([1])], keepdims=0)


def build_index_graph(g: G):
    """Patch addressing for window/full attention + RoPE tables. Returns dict of tensor names."""
    grid = g.op("Cast", ["image_grid_thw"], to=TP.INT32)  # [I, 3]  (the only int64 op; runs on CPU)
    T_ = g.gather(grid, g.i32(0), axis=1)
    H = g.gather(grid, g.i32(1), axis=1)
    W = g.gather(grid, g.i32(2), axis=1)
    n = g.mul(g.mul(T_, H), W)  # patches per image [I]
    ends = g.op("CumSum", [n, g.i32(0)])
    starts = g.sub(ends, n)
    Hb, Wb = g.div(H, g.i32(MERGE)), g.div(W, g.i32(MERGE))  # merged-token grid
    two, four, w4 = g.i32(2), g.i32(4), g.i32(WIN)

    N64 = g.gather(g.op("Shape", ["pixel_values"]), g.i64(0))
    N = g.op("Cast", [N64], to=TP.INT32)  # scalar
    p = g.op("Range", [g.i32(0), N, g.i32(1)])  # [N]

    # Per-patch coordinates. Processor order within an image: (bh, bw, mh, mw) with 2x2 merge blocks.
    img = g.owner(p, ends)
    q = g.sub(p, g.gather(starts, img))
    Wb_p = g.gather(Wb, img)
    mw = g.mod(q, two)
    mh = g.mod(g.div(q, two), two)
    m = g.div(q, four)
    bh = g.div(m, Wb_p)
    bw = g.sub(m, g.mul(bh, Wb_p))
    hpos = g.add(g.mul(bh, two), mh)
    wpos = g.add(g.mul(bw, two), mw)

    # Vision RoPE: dim = head_dim/2 = 40 -> 20 inv freqs each for h and w; cos/sin halves [N, 40].
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, HEAD_DIM // 2, 2, dtype=np.float32) / (HEAD_DIM // 2)))
    inv = g.const(inv_freq[None, :], np.float32)
    fh = g.mul(g.op("Unsqueeze", [g.op("Cast", [hpos], to=TP.FLOAT), g.i64([1])]), inv)
    fw = g.mul(g.op("Unsqueeze", [g.op("Cast", [wpos], to=TP.FLOAT), g.i64([1])]), inv)
    ang = g.op("Concat", [fh, fw], axis=1)
    cos, sin = g.op("Cos", [ang]), g.op("Sin", [ang])

    # ---- window groups -------------------------------------------------------------------------
    nWh = g.div(g.add(Hb, g.i32(WIN - 1)), w4)
    nWw = g.div(g.add(Wb, g.i32(WIN - 1)), w4)
    nwin = g.mul(nWh, nWw)
    wends = g.op("CumSum", [nwin, g.i32(0)])
    wstarts = g.sub(wends, nwin)
    n_windows = g.op("ReduceSum", [nwin, g.i64([0])], keepdims=0)

    # patch -> slot
    gwin = g.add(g.add(g.gather(wstarts, img), g.mul(g.div(bh, w4), g.gather(nWw, img))), g.div(bw, w4))
    rank = g.add(g.mul(g.add(g.mul(g.mod(bh, w4), w4), g.mod(bw, w4)), four), g.add(g.mul(mh, two), mw))
    win_slot = g.add(g.mul(gwin, g.i32(WIN_CAP)), rank)

    # slot -> patch (pad slots -> N, a zero row appended to the sequence)
    s = g.op("Range", [g.i32(0), g.mul(n_windows, g.i32(WIN_CAP)), g.i32(1)])
    sw = g.div(s, g.i32(WIN_CAP))
    r = g.sub(s, g.mul(sw, g.i32(WIN_CAP)))
    simg = g.owner(sw, wends)
    lw = g.sub(sw, g.gather(wstarts, simg))
    nWw_s = g.gather(nWw, simg)
    wh = g.div(lw, nWw_s)
    ww = g.sub(lw, g.mul(wh, nWw_s))
    s_bh = g.add(g.mul(wh, w4), g.div(r, g.i32(16)))
    s_bw = g.add(g.mul(ww, w4), g.mod(g.div(r, four), w4))
    s_mh = g.mod(g.div(r, two), two)
    s_mw = g.mod(r, two)
    Wb_s = g.gather(Wb, simg)
    valid = g.op("And", [g.op("Less", [s_bh, g.gather(Hb, simg)]), g.op("Less", [s_bw, Wb_s])])
    s_q = g.add(g.mul(g.add(g.mul(s_bh, Wb_s), s_bw), four), g.add(g.mul(s_mh, two), s_mw))
    win_idx = g.op("Where", [valid, g.add(g.gather(starts, simg), s_q), N])
    win_bias = key_bias(g, valid, g.i64([1, 1, WIN_CAP, WIN_CAP]), WIN_CAP)  # [Gw, 1, 64, 64]

    # ---- full-attention groups (one per image; only used when there are several images) ----------
    I = g.op("Cast", [g.gather(g.op("Shape", ["image_grid_thw"]), g.i64(0))], to=TP.INT32)
    single = g.op("Equal", [I, g.i32(1)])
    cap = g.op("ReduceMax", [n], keepdims=0)
    full_slot = g.add(g.mul(img, cap), q)
    fs = g.op("Range", [g.i32(0), g.mul(I, cap), g.i32(1)])
    fimg = g.div(fs, cap)
    fq = g.sub(fs, g.mul(fimg, cap))
    fvalid = g.op("Less", [fq, g.gather(n, fimg)])
    full_idx = g.op("Where", [fvalid, g.add(g.gather(starts, fimg), fq), N])
    cap64 = g.op("Unsqueeze", [g.op("Cast", [cap], to=TP.INT64), g.i64([0])])

    def multi_bias():
        return key_bias(g, fvalid, g.op("Concat", [g.i64([1, 1]), cap64, cap64], axis=0), -1, cap64)

    # [I, 1, cap, cap]; a 1-element dummy for the (common) single-image case so it is never allocated.
    full_bias = g.if_(single, lambda: g.const(np.zeros((1, 1, 1, 1)), g.np_dtype), multi_bias, g.dtype, name="FullBias")

    return dict(cos=cos, sin=sin, single=single, win=(win_idx, win_bias, win_slot, WIN_CAP),
                full=(full_idx, full_bias, full_slot, cap64))


def key_bias(g: G, valid, expand_shape, cap, cap64=None):
    """Additive attention bias [groups, 1, cap, cap] from per-slot key validity [groups * cap].

    MultiHeadAttention's key_padding_mask input is not implemented by the WebGPU EP, so padding is
    expressed as attention_bias. -1e4 rather than -inf/dtype-min: exp(-1e4) is exactly 0, and it
    cannot overflow to -inf in fp16 kernels.
    """
    neg = g.const(-1e4, g.np_dtype)
    zero = g.const(0.0, g.np_dtype)
    per_key = g.op("Where", [valid, zero, neg])  # [groups * cap]
    shape = g.i64([-1, 1, 1, cap]) if cap > 0 else g.op("Concat", [g.i64([-1, 1, 1]), cap64], axis=0)
    return g.op("Expand", [g.op("Reshape", [per_key, shape]), expand_shape])


def attention(g: G, q, k, v, bias, per_head: bool, fp32: bool = False):
    """softmax(q k^T / sqrt(d) + bias) v for q, k, v [G, C, 1280] -> [G*C, 1280].

    Plain MatMul/Softmax on purpose: com.microsoft.MultiHeadAttention on the ORT 1.30/1.31 WebGPU EP is
    inaccurate (2-30% error vs a float64 reference, even in fp32), while these ops are exact.
    per_head=True computes one head at a time so the [C, C] score buffer is 1/16 the size (full-attention
    layers: 120 MB instead of 1.9 GB in fp32 for a 1036x1036 image), keeping large images under WebGPU
    buffer limits.
    """
    out_type = g.dtype
    if fp32 and g.dtype != TP.FLOAT:  # scores/softmax/PV in fp32 for the fp16 build
        q, k, v = (g.op("Cast", [t], to=TP.FLOAT) for t in (q, k, v))
        bias = None if bias is None else g.op("Cast", [bias], to=TP.FLOAT)
    np_t = np.float32 if fp32 else g.np_dtype
    q = g.op("Mul", [q, g.const(HEAD_DIM ** -0.5, np_t)])  # fold the scale into q
    q4 = g.op("Transpose", [g.op("Reshape", [q, g.i64([0, 0, HEADS, HEAD_DIM])])], perm=[0, 2, 1, 3])  # [G,H,C,D]
    kt = g.op("Transpose", [g.op("Reshape", [k, g.i64([0, 0, HEADS, HEAD_DIM])])], perm=[0, 2, 3, 1])  # [G,H,D,C]
    v4 = g.op("Transpose", [g.op("Reshape", [v, g.i64([0, 0, HEADS, HEAD_DIM])])], perm=[0, 2, 1, 3])  # [G,H,C,D]

    def attend(qh, kh, vh):
        s = g.op("MatMul", [qh, kh])
        if bias is not None:
            s = g.op("Add", [s, bias])  # bias [G, 1, C, C] broadcasts over heads
        return g.op("MatMul", [g.op("Softmax", [s], axis=-1), vh])

    if per_head:
        # 16 heads as 4 x 4 (two levels of 4-way Split/Concat): a single 16-way Split/Concat needs 17
        # storage buffers in one shader, over WebGPU's maxStorageBuffersPerShaderStage (8 by default).
        def split4(t, heads):
            return g.op("Split", [t, g.i64([heads // 4] * 4)], n_out=4, axis=1)
        outer = []
        for grp in zip(*(split4(t, HEADS) for t in (q4, kt, v4))):
            inner = [attend(*p) for p in zip(*(split4(t, HEADS // 4) for t in grp))]
            outer.append(g.op("Concat", inner, axis=1))
        out = g.op("Concat", outer, axis=1)
    else:
        out = attend(q4, kt, v4)
    out = g.op("Reshape", [g.op("Transpose", [out], perm=[0, 2, 1, 3]), g.i64([-1, HIDDEN])])
    return g.op("Cast", [out], to=out_type) if fp32 and out_type != TP.FLOAT else out


def build(dtype: str, residual_fp32: bool, attn_fp32: bool = False) -> onnx.ModelProto:
    elem = {"fp32": TP.FLOAT, "fp16": TP.FLOAT16}[dtype]
    g = G(elem)
    f = safe_open(str(MODEL_DIR / "model.safetensors"), "pt")
    W = lambda k: f.get_tensor(k).float().numpy()

    idx = build_index_graph(g)
    act = elem
    res_t = TP.FLOAT if residual_fp32 else elem  # residual stream / layernorm dtype
    res_np = np.float32 if residual_fp32 else g.np_dtype

    def cast(x, to, frm):
        return x if to == frm else g.op("Cast", [x], to=to)

    # Patch embedding: Conv3d with kernel == stride == the whole patch == MatMul over 3*2*14*14.
    pe = W("visual.patch_embed.proj.weight").reshape(HIDDEN, -1).T  # [1176, 1280]
    x = g.op("MatMul", [cast("pixel_values", act, TP.FLOAT), g.weight("visual.patch_embed.proj.weight", pe)])
    x = cast(x, res_t, act)

    # Zero row appended to the attention operands so padded slots gather zeros.
    pad_row = g.const(np.zeros((1, 3 * HIDDEN)), g.np_dtype)
    rope_pos = g.i64([0])  # RotaryEmbedding: 1-D position_ids = offset, so token i uses cos/sin row i

    skip = None
    for i in range(DEPTH):
        pre = f"visual.blocks.{i}"
        # norm1 (fused with previous residual add where possible)
        w1 = g.weight(f"{pre}.norm1.weight", W(f"{pre}.norm1.weight"), res_np)
        if skip is None:
            h = g.op("SimplifiedLayerNormalization", [x, w1], epsilon=EPS, axis=-1, stash_type=1)
        else:
            h, _, _, x = g.op("SkipSimplifiedLayerNormalization", [x, skip, w1], n_out=4, domain="com.microsoft", epsilon=EPS)
        h = cast(h, act, res_t)

        # attention
        qkv = g.op("MatMul", [h, g.weight(f"{pre}.attn.qkv.weight", W(f"{pre}.attn.qkv.weight").T)])
        qkv = g.op("Add", [qkv, g.weight(f"{pre}.attn.qkv.bias", W(f"{pre}.attn.qkv.bias"))])
        q, k, v = g.op("Split", [qkv, g.i64([HIDDEN] * 3)], n_out=3, axis=-1)
        roped = []
        for t in (q, k):  # RoPE in fp32 like the reference (apply_rotary_pos_emb_vision upcasts)
            t3 = g.op("Unsqueeze", [cast(t, TP.FLOAT, act), g.i64([0])])
            t3 = g.op("RotaryEmbedding", [t3, rope_pos, idx["cos"], idx["sin"]], domain="com.microsoft",
                      num_heads=HEADS, rotary_embedding_dim=HEAD_DIM, interleaved=0)
            roped.append(cast(g.op("Squeeze", [t3, g.i64([0])]), act, TP.FLOAT))
        q, k = roped

        def grouped(addr):
            """Gather into padded groups, attend with key-padding bias, gather back to patch order."""
            gidx, gbias, gslot, cap = addr
            qkv_ = g.op("Concat", [g.op("Concat", [q, k, v], axis=-1), pad_row], axis=0)  # [N+1, 3840]
            shape = (g.i64([-1, cap, 3 * HIDDEN]) if isinstance(cap, int) else
                     g.op("Concat", [g.i64([-1]), cap, g.i64([3 * HIDDEN])], axis=0))
            gq, gk, gv = g.op("Split", [g.op("Reshape", [g.gather(qkv_, gidx), shape]), g.i64([HIDDEN] * 3)], n_out=3, axis=-1)
            att_ = attention(g, gq, gk, gv, gbias, per_head=not isinstance(cap, int), fp32=attn_fp32)
            return g.gather(att_, gslot)  # [N, 1280]

        def whole_sequence():
            """One image: full attention over the whole sequence, no bias, no gathers."""
            q3, k3, v3 = (g.op("Unsqueeze", [t, g.i64([0])]) for t in (q, k, v))
            return attention(g, q3, k3, v3, None, per_head=True, fp32=attn_fp32)

        if i in FULLATT:
            att = g.if_(idx["single"], whole_sequence, lambda: grouped(idx["full"]), act, name=f"FullAttention{i}")
        else:
            att = grouped(idx["win"])
        att = g.op("MatMul", [att, g.weight(f"{pre}.attn.proj.weight", W(f"{pre}.attn.proj.weight").T)])
        att = g.op("Add", [att, g.weight(f"{pre}.attn.proj.bias", W(f"{pre}.attn.proj.bias"))])

        # norm2 fused with the attention residual
        w2 = g.weight(f"{pre}.norm2.weight", W(f"{pre}.norm2.weight"), res_np)
        h, _, _, x = g.op("SkipSimplifiedLayerNormalization", [x, cast(att, res_t, act), w2], n_out=4,
                          domain="com.microsoft", epsilon=EPS)
        h = cast(h, act, res_t)

        # SwiGLU MLP with biases
        gate = g.op("Add", [g.op("MatMul", [h, g.weight(f"{pre}.mlp.gate_proj.weight", W(f"{pre}.mlp.gate_proj.weight").T)]),
                            g.weight(f"{pre}.mlp.gate_proj.bias", W(f"{pre}.mlp.gate_proj.bias"))])
        up = g.op("Add", [g.op("MatMul", [h, g.weight(f"{pre}.mlp.up_proj.weight", W(f"{pre}.mlp.up_proj.weight").T)]),
                          g.weight(f"{pre}.mlp.up_proj.bias", W(f"{pre}.mlp.up_proj.bias"))])
        act_out = g.op("Mul", [g.op("Mul", [gate, g.op("Sigmoid", [gate])]), up])
        down = g.op("Add", [g.op("MatMul", [act_out, g.weight(f"{pre}.mlp.down_proj.weight", W(f"{pre}.mlp.down_proj.weight").T)]),
                            g.weight(f"{pre}.mlp.down_proj.bias", W(f"{pre}.mlp.down_proj.bias"))])
        skip = cast(down, res_t, act)

    # Merger: RMSNorm(1280) -> view [-1, 5120] -> Linear -> GELU(erf) -> Linear(5120 -> 1024)
    wq = g.weight("visual.merger.ln_q.weight", W("visual.merger.ln_q.weight"), res_np)
    h, _, _, _ = g.op("SkipSimplifiedLayerNormalization", [x, skip, wq], n_out=4, domain="com.microsoft", epsilon=EPS)
    h = g.op("Reshape", [cast(h, act, res_t), g.i64([-1, HIDDEN * MERGE * MERGE])])
    h = g.op("Add", [g.op("MatMul", [h, g.weight("visual.merger.mlp.0.weight", W("visual.merger.mlp.0.weight").T)]),
                     g.weight("visual.merger.mlp.0.bias", W("visual.merger.mlp.0.bias"))])
    h = g.op("Gelu", [h])  # opset 20 Gelu, approximate="none" == nn.GELU()
    h = g.op("Add", [g.op("MatMul", [h, g.weight("visual.merger.mlp.2.weight", W("visual.merger.mlp.2.weight").T)]),
                     g.weight("visual.merger.mlp.2.bias", W("visual.merger.mlp.2.bias"))])
    g.nodes.append(helper.make_node("Cast", [h], ["image_features"], name="/output/Cast", to=TP.FLOAT)
                   if act != TP.FLOAT else helper.make_node("Identity", [h], ["image_features"], name="/output/Identity"))

    graph = helper.make_graph(
        g.nodes, "teleocr_vision_encoder",
        [helper.make_tensor_value_info("pixel_values", TP.FLOAT, ["num_patches", 1176]),
         helper.make_tensor_value_info("image_grid_thw", TP.INT64, ["num_images", 3])],
        [helper.make_tensor_value_info("image_features", TP.FLOAT, ["num_features", OUT_HIDDEN])],
        g.inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
                              producer_name="tele-ocr-onnx")
    return model


# suffix: (dtype, residual stream in fp32, attention in fp32). The fp16 build keeps the residual stream and
# attention scores/softmax in fp32: the ViT is precision-sensitive and with these it matches the fp32
# reference at least as well as the authors' own bf16 inference (see README "Accuracy").
VARIANTS = {"": ("fp32", True, True), "_fp16": ("fp16", True, True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "output" / "onnx"))
    ap.add_argument("--suffix", nargs="*", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--residual-fp16", action="store_true", help="experiment: residual stream in fp16 (fp16 variant)")
    ap.add_argument("--attn-fp16", action="store_true", help="experiment: attention scores/softmax in fp16 (fp16 variant)")
    args = ap.parse_args()
    for suffix in args.suffix:
        dtype, res32, attn32 = VARIANTS[suffix]
        res32 &= not args.residual_fp16
        attn32 &= not args.attn_fp16
        model = build(dtype, res32, attn32)
        path = Path(args.out) / f"vision_encoder{suffix}.onnx"
        n = save_chunked(model, path)
        if suffix == "_fp16":
            # No int4 ViT: 4-bit weights measurably break OCR (see README "Accuracy") and are not faster at
            # ViT batch sizes. `vision_encoder_q4f16` carries the fp16 weights so dtype="q4f16" works as a
            # single setting; the bytes are identical, so the Hub deduplicates the storage.
            q4 = Path(args.out) / "vision_encoder_q4f16.onnx"
            clone_saved(path, q4)
            print(f"wrote {q4} (fp16 weights; see comment)")
        print(f"wrote {path} ({file_size_mb(path):.0f} MB, {n} data chunk(s), residual {'fp32' if res32 else dtype})")


if __name__ == "__main__":
    main()
