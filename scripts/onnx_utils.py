"""Shared helpers for writing transformers.js-compatible ONNX files."""
from __future__ import annotations

import os
from pathlib import Path

import onnx
from onnx.external_data_helper import uses_external_data

# onnxruntime 1.30 (node + web) refuses models with IR version > 13; the installed `onnx`
# writes 14 by default. IR 10 is the version used by onnxruntime-genai and is widely supported.
IR_VERSION = 10

# Browsers cannot allocate a single ArrayBuffer >= 2 GiB, so external data is split into
# chunks named <model>.onnx_data, <model>.onnx_data_1, ... (transformers.js convention).
MAX_CHUNK_BYTES = 1_900_000_000
# Tensors smaller than this stay inline in the .onnx protobuf.
INLINE_THRESHOLD = 1024


def save_chunked(model: onnx.ModelProto, path: str | Path) -> int:
    """Save `model` to `path` with external data split into <2 GB chunks.

    Returns the number of external data chunks written (0 if everything fits inline).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.ir_version = IR_VERSION

    # Make sure all tensor data is loaded in memory before we rewrite locations.
    for t in _all_tensors(model):
        if uses_external_data(t):
            raise ValueError(f"tensor {t.name} still references external data; load it first")

    big = [t for t in _all_tensors(model) if len(t.raw_data) > INLINE_THRESHOLD]
    for old in path.parent.glob(path.name + "_data*"):
        old.unlink()

    chunk, offset, files = 0, 0, {}
    for t in big:
        size = len(t.raw_data)
        if offset and offset + size > MAX_CHUNK_BYTES:
            chunk, offset = chunk + 1, 0
        name = f"{path.name}_data" + ("" if chunk == 0 else f"_{chunk}")
        fh = files.get(name) or files.setdefault(name, open(path.parent / name, "wb"))
        # ORT requires each tensor to start at an aligned offset for mmap; pad to 4 KiB.
        pad = (-offset) % 4096
        if pad:
            fh.write(b"\0" * pad)
            offset += pad
        fh.write(t.raw_data)
        del t.external_data[:]
        for k, v in (("location", name), ("offset", str(offset)), ("length", str(size))):
            e = t.external_data.add()
            e.key, e.value = k, v
        t.data_location = onnx.TensorProto.EXTERNAL
        t.ClearField("raw_data")
        offset += size
    for fh in files.values():
        fh.close()

    with open(path, "wb") as f:
        f.write(model.SerializeToString())
    return len(files)


def _all_tensors(model: onnx.ModelProto):
    yield from model.graph.initializer
    for node in _all_nodes(model.graph):
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type:
                yield attr.t


def _all_nodes(graph):
    for node in graph.node:
        yield node
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                yield from _all_nodes(attr.g)


def load_full(path: str | Path) -> onnx.ModelProto:
    """Load a model with all external data pulled into memory (raw_data)."""
    model = onnx.load(str(path), load_external_data=True)
    for t in _all_tensors(model):
        if t.data_location == onnx.TensorProto.EXTERNAL:
            t.data_location = onnx.TensorProto.DEFAULT
            del t.external_data[:]
    return model


def file_size_mb(path: str | Path) -> float:
    path = Path(path)
    total = os.path.getsize(path) + sum(os.path.getsize(p) for p in path.parent.glob(path.name + "_data*"))
    return total / 1e6


def clone_saved(src: str | Path, dst: str | Path) -> None:
    """Copy a saved model + its external data chunks to a new base name, rewriting data locations."""
    import shutil

    src, dst = Path(src), Path(dst)
    model = onnx.load(str(src), load_external_data=False)
    rename = {}
    for chunk in sorted(src.parent.glob(src.name + "_data*")):
        new = dst.name + chunk.name[len(src.name):]
        shutil.copyfile(chunk, dst.parent / new)
        rename[chunk.name] = new
    for t in _all_tensors(model):
        for e in t.external_data:
            if e.key == "location":
                e.value = rename[e.value]
    with open(dst, "wb") as f:
        f.write(model.SerializeToString())
