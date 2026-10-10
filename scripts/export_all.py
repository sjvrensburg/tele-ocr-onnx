"""Build every artifact into output/: decoder, vision encoder, embeddings (fp32 / fp16 / q4f16) + configs.

    uv run scripts/export_all.py            # ~15 min on a CPU-only machine, ~40 GB peak RAM
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ["export_decoder.py"],
    ["export_vision.py"],
    ["export_embed.py"],
    ["make_package.py"],
]

if __name__ == "__main__":
    if not (ROOT / "models/TeleOCR/model.safetensors").exists():
        sys.exit("Download the checkpoint first:  hf download XingChen-AGI/TeleOCR --local-dir models/TeleOCR")
    (ROOT / "build").mkdir(exist_ok=True)
    for step in STEPS:
        print(f"\n### {' '.join(step)}", flush=True)
        subprocess.run([sys.executable, str(ROOT / "scripts" / step[0]), *step[1:]], check=True)
