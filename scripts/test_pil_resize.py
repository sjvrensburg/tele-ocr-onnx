"""Check that web/demo/pil_resize.js is bit-exact with Pillow's bicubic resize and matches smart_resize."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "build" / "pil_resize_test"

NODE = """
import fs from 'node:fs';
import { resizeBicubicPIL, smartResize } from '../../web/demo/pil_resize.js';
let bad = 0;
for (const c of JSON.parse(fs.readFileSync('cases.json'))) {
  const src = new Uint8Array(fs.readFileSync(`src_${c.tag}.u8`)), ref = new Uint8Array(fs.readFileSync(`pil_${c.tag}.u8`));
  const out = resizeBicubicPIL(src, c.w, c.h, 3, c.w2, c.h2);
  let diff = 0; for (let i = 0; i < ref.length; i++) diff += ref[i] !== out[i];
  const [sh, sw] = smartResize(c.h, c.w, 28, 3136, 12845056);
  const ok = diff === 0 && sh === c.smart[0] && sw === c.smart[1]; bad += !ok;
  console.log(`${ok ? 'OK ' : 'BAD'} ${c.tag.padEnd(34)} differing pixels: ${diff}, smart_resize ${sh}x${sw}`);
}
process.exit(bad ? 1 : 0);
"""


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    cases = []
    for f, target in [("table.png", None), ("text.png", None), ("formula.png", None), ("code.png", None),
                      ("layout.jpg", (1036, 1036)), ("layout.jpg", None), ("scientific_figure.png", None)]:
        im = Image.open(ROOT / "models/TeleOCR/assets" / f).convert("RGB")
        h2, w2 = smart_resize(im.height, im.width, 28, 3136, 12845056) if target is None else target[::-1]
        tag = f"{f}_{w2}x{h2}"
        np.array(im).tofile(TMP / f"src_{tag}.u8")
        np.array(im.resize((w2, h2), Image.BICUBIC)).tofile(TMP / f"pil_{tag}.u8")
        cases.append(dict(tag=tag, w=im.width, h=im.height, w2=w2, h2=h2,
                          smart=list(smart_resize(im.height, im.width, 28, 3136, 12845056))))
    (TMP / "cases.json").write_text(json.dumps(cases))
    (TMP / "test.mjs").write_text(NODE)
    sys.exit(subprocess.run(["node", "test.mjs"], cwd=TMP).returncode)


if __name__ == "__main__":
    main()
