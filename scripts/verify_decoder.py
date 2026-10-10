"""Check decoder_model_merged*.onnx against the PyTorch TeleOCR language model.

Uses random input embeddings with *different* T/H/W position streams (like an image block) so a
wrong MRoPE layout or missing QK-norm shows up, a left-padded batch of 2, and one cached decode step.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[1]


def make_positions(S, img_at, h, w):
    """Text positions, then an h x w image block (t const, h/w vary), then text again."""
    t_ids, h_ids, w_ids = [], [], []
    p = 0
    for i in range(img_at):
        t_ids.append(p); h_ids.append(p); w_ids.append(p); p += 1
    for r in range(h):
        for c in range(w):
            t_ids.append(p); h_ids.append(p + r); w_ids.append(p + c)
    p = p + max(h, w)
    while len(t_ids) < S:
        t_ids.append(p); h_ids.append(p); w_ids.append(p); p += 1
    return np.array([t_ids, h_ids, w_ids], dtype=np.int64), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "output/onnx/decoder_model_merged.onnx"))
    ap.add_argument("--ep", default="cpu")
    args = ap.parse_args()

    torch.manual_seed(0)
    hf = AutoModel.from_pretrained(ROOT / "models/TeleOCR", trust_remote_code=True, dtype=torch.float32,
                                   attn_implementation="eager").eval()
    lm, head = hf.model.language_model, hf.lm_head

    B, S, pad = 2, 48, 5
    pos1, next_pos = make_positions(S, 6, 4, 6)
    position_ids = np.stack([pos1, pos1], axis=1)  # [3, B, S]
    attention_mask = np.ones((B, S), dtype=np.int64)
    attention_mask[1, :pad] = 0  # left padding on row 1
    # Row 1: shift positions so real tokens start at 0, padded slots get 1 (HF convention).
    real = S - pad
    pos_r1, _ = make_positions(real, 6, 4, 6)
    position_ids[:, 1, pad:] = pos_r1
    position_ids[:, 1, :pad] = 1
    embeds = (torch.randn(B, S, 1024) * 0.02).numpy().astype(np.float32)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(args.model, so, providers=[{"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}[args.ep]])
    in_types = {i.name: i.type for i in sess.get_inputs()}
    kv_dtype = np.float16 if "float16" in in_types["past_key_values.0.key"] else np.float32
    emb_dtype = np.float16 if "float16" in in_types["inputs_embeds"] else np.float32

    feeds = {"inputs_embeds": embeds.astype(emb_dtype), "attention_mask": attention_mask, "position_ids": position_ids}
    for i in range(28):
        for kv in ("key", "value"):
            feeds[f"past_key_values.{i}.{kv}"] = np.zeros((B, 8, 0, 128), dtype=kv_dtype)
    outs = sess.run(None, feeds)
    names = [o.name for o in sess.get_outputs()]
    res = dict(zip(names, outs))
    onnx_logits = res["logits"][:, -1].astype(np.float32)

    with torch.no_grad():
        o = lm(inputs_embeds=torch.from_numpy(embeds), attention_mask=torch.from_numpy(attention_mask),
               position_ids=torch.from_numpy(position_ids), use_cache=True)
        ref_logits = head(o.last_hidden_state)[:, -1].numpy()
        cache = o.past_key_values

    def report(tag, a, b):
        err = np.abs(a - b).max()
        cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
        top = (a.argmax(-1) == b.argmax(-1)).all()
        print(f"{tag:10s} max|diff|={err:.4g}  cos={np.round(cos, 6)}  argmax_match={top}")
        return top

    ok = report("prefill", onnx_logits, ref_logits)

    # KV check against HF cache for layer 0 and 27 (row 0 only; padding rows can differ legitimately).
    for layer in (0, 27):
        k_ref = cache.layers[layer].keys[0].numpy() if hasattr(cache, "layers") else cache.key_cache[layer][0].numpy()
        k_onnx = res[f"present.{layer}.key"][0].astype(np.float32)
        print(f"present.{layer}.key max|diff|={np.abs(k_ref - k_onnx).max():.4g}")

    # One decode step with cache.
    new_emb = (torch.randn(B, 1, 1024) * 0.02).numpy().astype(np.float32)
    step_pos = np.zeros((3, B, 1), dtype=np.int64)
    step_pos[:, 0] = next_pos  # next text position after row 0's prompt
    step_pos[:, 1] = make_positions(real, 6, 4, 6)[1]  # ... and after row 1's (shorter) prompt
    step_mask = np.concatenate([attention_mask, np.ones((B, 1), dtype=np.int64)], axis=1)
    feeds = {"inputs_embeds": new_emb.astype(emb_dtype), "attention_mask": step_mask, "position_ids": step_pos}
    for i in range(28):
        for kv in ("key", "value"):
            feeds[f"past_key_values.{i}.{kv}"] = res[f"present.{i}.{kv}"]
    step = dict(zip(names, sess.run(None, feeds)))
    with torch.no_grad():
        o2 = lm(inputs_embeds=torch.from_numpy(new_emb), attention_mask=torch.from_numpy(step_mask),
                position_ids=torch.from_numpy(step_pos), past_key_values=cache, use_cache=True)
        ref2 = head(o2.last_hidden_state)[:, -1].numpy()
    ok &= report("decode", step["logits"][:, -1].astype(np.float32), ref2)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
