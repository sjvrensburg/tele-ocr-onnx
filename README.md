# tele-ocr-onnx

Converts [XingChen-AGI/TeleOCR](https://huggingface.co/XingChen-AGI/TeleOCR) to ONNX graphs that run in the browser
on **WebGPU** via transformers.js (`qwen2_5_vl` session layout) / onnxruntime-web. The result in `output/` is a
ready-to-host model folder with its own model card (`docs/MODEL_CARD.md`).

The converted model is published at [stefanj0/TeleOCR-ONNX](https://huggingface.co/stefanj0/TeleOCR-ONNX).

## Quick start

```bash
uv sync                                                   # Python env (CPU torch, transformers 4.57.1, onnxruntime-genai 0.17.1)
hf download XingChen-AGI/TeleOCR --local-dir models/TeleOCR
uv run scripts/export_all.py                              # -> output/  (~10 min, ~40 GB RAM peak, deterministic)

cd web && npm install
python3 serve.py &                                        # static server on http://127.0.0.1:8765
# open http://127.0.0.1:8765/web/demo/index.html          (Chrome/Edge with WebGPU)
```

## What TeleOCR actually is

`config.json` claims `Qwen2_5_VLForConditionalGeneration`, and the remote code (`modeling_naviocr.py`) claims to be
auto-generated from upstream. Diffing it against transformers 4.57.1 shows the real differences:

* **Decoder = Qwen3-style attention on a Qwen2.5-VL body**: `q_norm`/`k_norm` (per-head RMSNorm before RoPE), no
  QKV bias, `head_dim=128` with `hidden_size=1024` (q_proj 1024→2048, o_proj 2048→1024), GQA 16/8.
* **Sectioned MRoPE** (`[16,24,24]`, Qwen2.5-VL style), not Qwen3-VL's interleaved layout.
* `lm_head` is stored separately but equals `embed_tokens` (tied).
* Vision tower is stock Qwen2.5-VL.

## Pipeline

| script | output | how |
|---|---|---|
| `scripts/export_decoder.py` | `decoder_model_merged{,_fp16,_q4f16}.onnx` | onnxruntime-genai builder + `TeleOCRTextModel` (Qwen3VL attention path with `mrope_layout=0`, weights loaded through the checkpoint's own code), then graph fixes (below) |
| `scripts/export_vision.py` | `vision_encoder{,_fp16,_q4f16}.onnx` | hand-built ONNX graph (see below) |
| `scripts/export_embed.py` | `embed_tokens{,_fp16,_q4f16}.onnx` | Gather / GatherBlockQuantized (int4, block 32) |
| `scripts/make_package.py` | configs, tokenizer, model card | adds `transformers.js_config`, explicit preprocessor defaults, chat template in `tokenizer_config.json` |
| `scripts/reference.py` | `golden/*.npz` | PyTorch fp32 reference tensors and greedy outputs |
| `ort/teleocr_ort.py` | shipped as `ort/teleocr_ort.py` | standalone plain-ONNX-Runtime pipeline (no transformers.js / torch) |

### Vision encoder design

The PyTorch window attention is export-hostile (python loops over `grid_thw`, `.tolist()`, per-window `split`,
argsort permutation). The graph instead:

* never permutes tokens. Each patch gets a (group, slot) address computed arithmetically from `image_grid_thw`.
  Window layers gather q/k/v into padded `[windows, 64, ·]` batches with an additive key-padding bias. Full-attention
  layers use an `If`: for a single image, plain attention over the whole sequence (no bias, no gathers); for several
  images, per-image padded groups. Multi-image batches are handled correctly.
* does all index math in int32 (the WebGPU EP has no int64 kernels and no int32 `Mod`, so `a % b` is written as
  `a − (a/b)·b`).
* computes full attention one head at a time (two levels of 4-way Split/Concat, staying under
  `maxStorageBuffersPerShaderStage`), bounding score memory at N²·4 bytes.

### WebGPU pitfalls found (and handled)

| issue | effect | fix |
|---|---|---|
| `com.microsoft.MultiHeadAttention` on the ORT 1.30/1.31 WebGPU EP | 2–30 % error vs a float64 reference, even in fp32 | plain MatMul/Softmax attention in the ViT (exact) |
| MHA `key_padding_mask` | loads, then throws "not implemented for webgpu" at run time | additive attention bias |
| 16-way Split/Concat | 17 storage buffers > per-shader limit | 4×4 two-level split |
| no int64 WebGPU kernels | silent CPU fallbacks | int32 index math; `placement_report.mjs` audits placements |
| builder GQA `seqlens_k = sum(mask)-1` | wrong with left padding (transformers.js pads left) | full-length `seqlens_k` + mask-derived `attention_bias` |
| builder KV head dim is symbolic `kv_cache_dim` | transformers.js builds empty cache `[B,8,0,0]` → GQA error | pinned to 128 |
| transformers.js splices float32 image features into embeddings by raw copy | fp16 embeddings would corrupt | all `inputs_embeds`/`image_features` boundaries float32 |
| `preprocessor_config.json` relies on Python class defaults | transformers.js skipped resize/normalize, used bilinear | defaults written out explicitly |
| transformers.js browser resize ignores filter (canvas) | changed OCR output (table title row dropped) | bit-exact PIL bicubic port (`web/demo/pil_resize.js`) |
| onnx 1.23 writes IR v14, ORT 1.30 max is 13 | load failure | IR 10 |
| fp16 ViT precision | some tokens off | fp32 residual, norms, RoPE and softmax inside the fp16 graph (≥ bf16 fidelity) |
| int4 ViT | table OCR NED 0.12, not faster | `vision_encoder_q4f16` carries fp16 weights |

## Verification

```bash
uv run scripts/verify_decoder.py        # ONNX fp32 decoder vs PyTorch (CPU EP): padded batch + cached decode
uv run scripts/verify_vision.py         # ONNX fp32 vision vs PyTorch (CPU EP): golden + odd grids + multi-image
uv run scripts/verify_ort_pipeline.py   # plain-ORT pipeline: glue identical to transformers + greedy text per dtype (CPU EP)
cd web
node verify_webgpu.mjs --vision fp32 _fp16 --decoder fp32 _fp16 _q4f16   # components on WebGPU vs golden
node e2e_transformers.mjs --dtype vision=fp16,embed=q4f16,decoder=q4f16  # full transformers.js pipeline (Node, WebGPU)
node placement_report.mjs ../output/onnx/*.onnx                          # nodes that fall back to CPU
node browser_test.mjs --dtype q4f16                                      # real Chrome (headless, WebGPU) via the demo page
```

Results are in `results/` (see `final_verification.log`). In Chrome on WebGPU with PIL-exact preprocessing, every
precision reproduces the PyTorch fp32 greedy output exactly for text, formula and layout (512-token prefix). Table
differs by one character at a PyTorch near-tie (p 0.58/0.42), where the ONNX output is the correct reading.
