---
license: apache-2.0
base_model: XingChen-AGI/TeleOCR
library_name: transformers.js
pipeline_tag: image-text-to-text
tags:
  - onnx
  - webgpu
  - ocr
  - document-parsing
language:
  - zh
  - en
---

# TeleOCR — ONNX for WebGPU

Unofficial community ONNX conversion of [XingChen-AGI/TeleOCR](https://huggingface.co/XingChen-AGI/TeleOCR) (1.4B document-parsing VLM)
that runs entirely in the browser with [transformers.js](https://github.com/huggingface/transformers.js) and the
ONNX Runtime **WebGPU** execution provider.

Conversion and verification code: <https://github.com/sjvrensburg/tele-ocr-onnx>.

## Architecture

`config.json` says `qwen2_5_vl`, but TeleOCR is not a stock Qwen2.5-VL:

| part | what it really is |
|---|---|
| vision tower | Qwen2.5-VL ViT (32 blocks, 112 px window attention, full attention in blocks 7/15/23/31, 2×2 merger → 1024) |
| language model | **Qwen3-style attention**: per-head RMSNorm on Q and K (`q_norm`/`k_norm`) before RoPE, no QKV bias, `head_dim=128` decoupled from `hidden_size=1024` (q_proj 1024→2048), 16 Q / 8 KV heads, 28 layers |
| positions | Qwen2.5-VL **sectioned** 3D MRoPE (`mrope_section=[16,24,24]`), not Qwen3-VL's interleaved layout |

Converters that treat it as Qwen2.5-VL (or as Qwen3-VL) produce wrong graphs, and loading the checkpoint through
the stock transformers class silently drops the Q/K norms.

## Files

The graphs use the transformers.js `qwen2_5_vl` layout: `vision_encoder`, `embed_tokens`, `decoder_model_merged`.

| dtype | vision_encoder | embed_tokens | decoder_model_merged | total |
|---|---|---|---|---|
| `q4f16` (recommended) | 1.33 GB (fp16 weights\*) | 90 MB (int4) | 352 MB (int4, fp16 KV) | **1.77 GB** |
| `fp16` | 1.33 GB | 311 MB | 1.21 GB | 2.85 GB |
| `fp32` | 2.65 GB | 622 MB | 2.40 GB | 5.68 GB |

\* `vision_encoder_q4f16` deliberately holds the **fp16** vision weights (byte-identical to `vision_encoder_fp16`).
4-bit weights in the ViT measurably break OCR (one table sample went to NED 0.12) and are not faster at ViT batch
sizes, so no int4 ViT is shipped. The duplicate name means `dtype: "q4f16"` works as a single setting.

In the fp16 vision encoder, the residual stream, RMSNorm, RoPE and attention softmax are computed in fp32.

## Usage (transformers.js ≥ 4.3)

```js
import { AutoProcessor, Qwen2_5_VLForConditionalGeneration, RawImage } from '@huggingface/transformers';
import { resizeBicubicPIL, smartResize } from './js/pil_resize.js';  // shipped in this repo, see below

const model_id = 'stefanj0/TeleOCR-ONNX';
const processor = await AutoProcessor.from_pretrained(model_id);
const model = await Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, { device: 'webgpu', dtype: 'q4f16' });

// rgb: Uint8ClampedArray [h, w, 3] decoded from the image (see "Preprocessing in browsers")
const [h2, w2] = smartResize(h, w, 28, 3136, 12845056);
const image = new RawImage(resizeBicubicPIL(rgb, w, h, 3, w2, h2), w2, h2, 3);

const messages = [
  { role: 'system', content: 'You are a helpful assistant.' },
  { role: 'user', content: [{ type: 'image' }, { type: 'text', text: 'Please output the text content from the image.' }] },
];
const text = processor.apply_chat_template(messages, { add_generation_prompt: true });
const inputs = await processor(text, image);
const out = await model.generate({ ...inputs, max_new_tokens: 4096, do_sample: false });
console.log(processor.batch_decode(out.slice(null, [inputs.input_ids.dims[1], null]), { skip_special_tokens: true })[0]);
```

Prompts from the original model card: text `Please output the text content from the image.` · table
`This is the image of a table. Please output the table in OTSL format.` · formula
`Please write out the expression of the formula in the image using LaTeX format.` · code
`The image contains a code snippet, please output the parsing result.` · layout `Analyze the image layout.` (resize
the page to 1036×1036 first) · distorted layout `\nMulti-point Layout Segmentation Analysis.` · scientific figure
`This is a scientific figure. Please extract the table implied by this figure.`

### Preprocessing in browsers (important for fidelity)

In browsers, transformers.js resizes images with canvas `drawImage`, which ignores the bicubic filter and does not
antialias like PIL. TeleOCR's output changes noticeably with this. One table sample lost its title row, and layout boxes moved.
`js/pil_resize.js` is a **bit-exact** port of Pillow's bicubic resampler and of Qwen2-VL's `smart_resize`. Resize the image
to the `smartResize` size yourself (and to 1036×1036 first for layout prompts). The processor's own resize is then
a no-op. Decode with `createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' })` to match
PIL, which ignores ICC profiles. In Node, transformers.js resizes with sharp, which also differs from PIL. The same
helper fixes that.

`preprocessor_config.json` spells out Qwen2VLImageProcessor's Python defaults (`do_resize`, bicubic `resample`,
`do_normalize`, …). Python ignores this, but transformers.js otherwise skips resizing and normalization.

## Usage with plain ONNX Runtime (Python, no transformers.js)

The graphs are ordinary ONNX Runtime models; any ORT build and execution provider can run them. The fp32, fp16 and
q4f16 variants all run on the default CPU provider. `ort/teleocr_ort.py` is a self-contained pipeline. It covers
preprocessing, the chat prompt, 3D MRoPE positions, feature splicing, and a KV-cached greedy loop with the
checkpoint's repetition penalty. It needs only `onnxruntime`, `tokenizers`, `numpy`, `pillow` and `huggingface_hub`.
It does not use PyTorch or transformers.

```bash
pip install onnxruntime tokenizers numpy pillow huggingface_hub
python teleocr_ort.py page.png --task text                    # downloads only the chosen dtype (default q4f16)
python teleocr_ort.py table.png --task table --dtype fp32
python teleocr_ort.py doc.jpg --task layout --provider cuda   # needs onnxruntime-gpu
```

```python
from teleocr_ort import TeleOCR, download, PROMPTS
from PIL import Image

model = TeleOCR(download("stefanj0/TeleOCR-ONNX", "q4f16"), dtype="q4f16", provider="cpu")
text, stats = model(Image.open("page.png"), PROMPTS["text"])
```

Verified against PyTorch on the CPU provider. Pixel values, prompt ids and position ids are identical to
transformers. Greedy output matches the PyTorch reference (see Accuracy), and the repetition-penalty path matches
`model.generate()` with the checkpoint's `generation_config`. For layout prompts the script resizes the page to
1036×1036 first, as the original model card does.

## Accuracy

Measured against the original PyTorch model in fp32 with greedy decoding, on the samples from the TeleOCR model card. Browser = Chrome 154 on WebGPU,
with bit-exact PIL preprocessing.

| | text | formula | table | layout (512-token prefix) |
|---|---|---|---|---|
| browser fp32 | exact | exact | 1 char (near-tie\*) | exact |
| browser fp16 | exact | exact | 1 char (near-tie\*) | exact |
| browser q4f16 | exact | exact | 1 char (near-tie\*) | exact |

\* PyTorch fp32 itself puts p=0.58 / 0.42 on `外现` / `观` there. All ONNX variants pick `外观检查`, which is the
correct reading of the image.

On the scientific-figure sample, the original model is itself unreliable. PyTorch fp32 reads two of five bar values
wrong, and the ONNX variants differ from it on one or two values at near-ties (sometimes in the correct direction).
Treat chart-to-table output as approximate regardless of runtime.

Component checks (same inputs): the fp32 vision encoder matches PyTorch to cosine 1.000000 (rel. error ≤ 5e-5) for
single images, odd grids, and multi-image batches. The fp32 decoder matches logits to 2.4e-5, including left-padded
batches and cached decoding. The fp16 vision encoder is at least as close to fp32 as the authors' own bf16 PyTorch
inference on every sample.

## Speed

Chrome 154, AMD Radeon 8060S iGPU (Ryzen AI MAX+ 395), Linux/Vulkan. Weights already loaded.

| dtype | time to first token (102 → 1683 visual tokens) | decode |
|---|---|---|
| q4f16 | 0.13 s → 2.1 s | 65–90 tok/s |
| fp16 | 0.15 s → 2.1 s | 29–33 tok/s |
| fp32 | 0.22 s → 3.2 s | 22–26 tok/s |

Plain ONNX Runtime 1.30 on the CPU provider (`ort/teleocr_ort.py`), same machine (Ryzen AI MAX+ 395, 16 cores),
formula sample with 464 visual tokens:

| dtype | time to first token | decode |
|---|---|---|
| q4f16 | 2.0 s | 57 tok/s |
| fp16 | 2.0 s | 26 tok/s |
| fp32 | 1.8 s | 24 tok/s |

## Notes and limitations

* Images only (each image has `t = 1`, always true for Qwen2VLImageProcessor images). Video is not supported.
* Memory grows with image size. Full-attention ViT layers run one head at a time, using N²·4 bytes per score
  buffer for N patches (120 MB at 1036×1036). `max_pixels` keeps the original 12.8 MP. For very large pages, downscale
  first or lower `max_pixels`.
* The RoPE cache covers 32,768 positions (prompt + output).
* `generation_config.json` is the original: `repetition_penalty=1.05` and effectively greedy (`top_k=1`).
* Conversion: the decoder was built with onnxruntime-genai's model builder, using a TeleOCR-specific subclass plus
  graph fixes for left padding, the float32 `inputs_embeds` boundary, and a static KV head dim. The vision encoder
  and embeddings are hand-built ONNX graphs.

## Credits

All credit for the model goes to the TeleOCR authors. Weights are the original Apache-2.0 release, converted without
fine-tuning. For the full document-parsing pipeline, see <https://github.com/caipeng328/TeleOCR>.

```bibtex
@article{teleocr,
  title={TeleOCR: Navigating Document Parsing Across Digital and Camera-Captured Documents},
  author={Cai, Peng and Zou, Zhaofan and Liu, Shifa and Wang, Yikun and Tang, Jiawei and Yang, Kaicheng and Tong, Meng and He, Zhongjiang and Sun, Hao},
  journal={arXiv preprint arXiv:2608.12898},
  year={2026}
}
```
