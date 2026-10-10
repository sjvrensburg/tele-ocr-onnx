"""Run the original TeleOCR PyTorch model (custom remote code, fp32 CPU) and dump golden tensors.

Saved per sample into golden/<name>.npz:
  pixel_values, image_grid_thw, input_ids, position_ids (3,B,S), image_features,
  prefill_logits_last, generated_ids
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "TeleOCR"

SAMPLES = {
    "text": ("assets/text.png", "Please output the text content from the image.", None),
    "formula": ("assets/formula.png", "Please write out the expression of the formula in the image using LaTeX format.", None),
    "table": ("assets/table.png", "This is the image of a table. Please output the table in OTSL format.", None),
    "layout": ("assets/layout.jpg", "Analyze the image layout.", (1036, 1036)),
    "code": ("assets/code.png", "The image contains a code snippet, please output the parsing result.", None),
    "figure": ("assets/scientific_figure.png", "This is a scientific figure. Please extract the table implied by this figure.", None),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="*", default=list(SAMPLES))
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    torch.manual_seed(0)
    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="eager").eval()
    print(type(model), type(model.model.language_model.layers[0].self_attn))

    out_dir = ROOT / "golden"
    out_dir.mkdir(exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for name in args.samples:
        path, prompt, resize = SAMPLES[name]
        image = Image.open(MODEL_DIR / path).convert("RGB")
        if resize:
            image = image.resize(resize, Image.Resampling.BICUBIC)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
        ]
        chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[chat], images=[image], padding=True, return_tensors="pt")

        with torch.no_grad():
            image_features = model.model.get_image_features(inputs.pixel_values, inputs.image_grid_thw)
            image_features = torch.cat(image_features, dim=0)
            position_ids, _ = model.model.get_rope_index(inputs.input_ids, inputs.image_grid_thw, None, inputs.attention_mask)
            prefill = model(**inputs, use_cache=False)
            t0 = time.time()
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, repetition_penalty=1.0)
            dt = time.time() - t0
        new_ids = gen[0, inputs.input_ids.shape[1]:]
        text = processor.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        np.savez(
            out_dir / f"{name}.npz",
            pixel_values=inputs.pixel_values.numpy(),
            image_grid_thw=inputs.image_grid_thw.numpy(),
            input_ids=inputs.input_ids.numpy(),
            attention_mask=inputs.attention_mask.numpy(),
            position_ids=position_ids.numpy(),
            image_features=image_features.numpy(),
            prefill_logits_last=prefill.logits[0, -1].numpy(),
            generated_ids=new_ids.numpy(),
        )
        summary[name] = {"grid_thw": inputs.image_grid_thw.tolist(), "prompt_len": inputs.input_ids.shape[1],
                         "new_tokens": len(new_ids), "seconds": round(dt, 1), "text": text}
        print(f"== {name} grid={inputs.image_grid_thw.tolist()} len={inputs.input_ids.shape[1]} ({dt:.1f}s)\n{text}\n")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
