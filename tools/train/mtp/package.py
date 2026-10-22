"""Splice the trained head into copies of a grafted file: D (the deliverable) and D' (the G8 arm).

There is one job here. The 15 trained head tensors replace the stock ones in a copy of a grafted
GGUF; the 8 2D tensors are written at f16 and the 7 norms at F32, and every other tensor is
copied byte for byte, so the verifier is untouched by construction.

The output is written to a temporary file, asserted, and only then renamed over the destination:
a failed build must never damage the artifact it was derived from.

There is no control artifact. De-quantizing an imatrix Q4_K_S file keeps its rounded weights, so
it is not a clean body and cannot serve as a reference (spec revision 9).
"""

from __future__ import annotations

import argparse
import gc
import json
import tomllib
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter

from tools.train.mtp.graft import _decode, add_source_tensor, copy_metadata
from tools.train.mtp.mtp_model import GGUF_HEAD_NORM_NAMES, GGUF_NAME_MAP, _to_float32
from tools.train.mtp.state import sha256_file

F16 = GGMLQuantizationType.F16
F32 = GGMLQuantizationType.F32
EXPECTED_TENSORS = 441
EXPECTED_BLOCKS = 33
EXPECTED_NEXTN = 4

# `torch.load` defaults to weights_only=True, and a trainer checkpoint carries this marker
# inside the optimizer state. The allowlist is the documented way to keep that default.
torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])


def _arch(reader: GGUFReader) -> str:
    field = reader.get_field("general.architecture")
    return str(_decode(field.parts[field.data[0]], field.types[0]))


def _assert_shape(path: Path) -> None:
    reader = GGUFReader(str(path))
    names = [t.name for t in reader.tensors]
    if len(names) != EXPECTED_TENSORS:
        raise RuntimeError(f"{path}: {len(names)} tensors, expected {EXPECTED_TENSORS}")
    blocks = {int(t.name.split(".")[1]) for t in reader.tensors if t.name.startswith("blk.")}
    if max(blocks) + 1 != EXPECTED_BLOCKS:
        raise RuntimeError(f"{path}: block_count is {max(blocks) + 1}, expected {EXPECTED_BLOCKS}")
    head = [n for n in names if n.startswith("blk.32.")]
    nextn = [n for n in head if n.startswith("blk.32.nextn")]
    if len(nextn) != EXPECTED_NEXTN:
        raise RuntimeError(f"{path}: {len(nextn)} blk.32.nextn tensors, expected {EXPECTED_NEXTN}")
    if len(head) != len(GGUF_NAME_MAP):
        raise RuntimeError(f"{path}: {len(head)} head tensors, expected {len(GGUF_NAME_MAP)}")
    missing = sorted(set(GGUF_NAME_MAP.values()) - set(head))
    if missing:
        raise RuntimeError(f"{path}: missing head tensors {missing}")
    # The norms must stay F32: the head ships unquantized, and a quantized norm shifts every
    # downstream activation by a constant the trainer never saw.
    bad = [
        t.name
        for t in reader.tensors
        if t.name in GGUF_HEAD_NORM_NAMES and t.tensor_type != F32
    ]
    if bad:
        raise RuntimeError(f"{path}: head norms are not F32: {bad}")
    del reader
    gc.collect()


def _write(writer: GGUFWriter, tmp: Path) -> None:
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    del writer
    gc.collect()
    _assert_shape(tmp)


def _gguf_values(name: str, tensor: torch.Tensor) -> np.ndarray:
    """The values a head tensor must have in the GGUF, from the trainer's convention.

    The trainer's checkpoint is HF convention: `Qwen3_5RMSNorm` multiplies by `(1 + w)`. The
    GGUF must hold `w + 1`, because llama.cpp multiplies by the weight itself. Writing the
    checkpoint verbatim puts every one of the 7 norms off by one - which is exactly the bug
    that made the first graft accept 0 of 187 drafts (spec revision 7). The addition happens in
    f32 so a value near 1 does not lose mantissa bits.
    """
    data = tensor.float().numpy()
    if name in GGUF_HEAD_NORM_NAMES:
        data = data + np.float32(1.0)
    return data


def splice(src: Path, head_pt: Path, dst: Path) -> Path:
    """Write the 15 head tensors into a copy of `src`.

    The 8 2D tensors are written at f16 and the 7 norms at F32, matching the graft. Every other
    tensor is copied byte for byte, so the verifier is untouched by construction.
    """
    reader = GGUFReader(str(src))
    payload = torch.load(head_pt, map_location="cpu")
    state = payload.get("head", payload)
    replacements, written = {}, {}
    for key, gguf_name in GGUF_NAME_MAP.items():
        if key not in state:
            raise RuntimeError(f"{head_pt}: checkpoint has no tensor {key}")
        replacements[gguf_name] = state[key]
        written[gguf_name] = _gguf_values(gguf_name, state[key])
    existing = {t.name for t in reader.tensors if t.name.startswith("blk.32.")}
    if set(replacements) != existing:
        raise RuntimeError(
            f"name map mismatch: missing={sorted(existing - set(replacements))} "
            f"extra={sorted(set(replacements) - existing)}"
        )

    source = {t.name: t for t in reader.tensors if t.name in replacements}
    deltas: dict[str, float] = {}
    for name, tensor in source.items():
        before = _to_float32(tensor)
        after = written[name]
        if before.shape != after.shape:
            raise RuntimeError(f"{name}: splice shape {after.shape} != graft shape {before.shape}")
        deltas[name] = float(np.abs(before - after).max())
        del before

    tmp = dst.with_suffix(".gguf.tmp")
    writer = GGUFWriter(str(tmp), _arch(reader))
    copy_metadata(reader, writer)
    for tensor in reader.tensors:
        if tensor.name in replacements:
            data = written[tensor.name]
            numpy_shape = list(data.shape)
            if tensor.name in GGUF_HEAD_NORM_NAMES:
                writer.add_tensor(tensor.name, data, raw_shape=numpy_shape, raw_dtype=F32)
            else:
                writer.add_tensor(
                    tensor.name, data.astype(np.float16), raw_shape=numpy_shape, raw_dtype=F16
                )
        else:
            add_source_tensor(writer, tensor)
    _write(writer, tmp)
    tmp.replace(dst)

    changed = sum(1 for value in deltas.values() if value > 0.0)
    print(json.dumps({"head_delta_max": deltas, "tensors_changed": changed}, indent=2))
    if changed == 0:
        raise RuntimeError(
            f"{head_pt} is identical to the stock head on all 15 tensors: the splice would ship "
            "an untrained head"
        )
    return dst


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--src", type=Path, default=None)
    parser.add_argument("--head", type=Path, default=None)
    parser.add_argument("--dst", type=Path, required=True)
    args = parser.parse_args()
    cfg = tomllib.loads(args.config.read_text(encoding="utf-8"))
    run_dir = Path("runs/mtp-alignment") / cfg["run"]["run_id"]
    src = args.src or (run_dir / "artifacts" / "B-grafted-f16.gguf")
    dst = args.dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    head = args.head or (run_dir / "checkpoints" / "main" / "best.pt")
    if not head.exists():
        raise SystemExit(f"missing head checkpoint: {head}")
    path = splice(src, head, dst)
    print(f"{path}  {path.stat().st_size / 2**30:.2f} GiB  sha256 {sha256_file(path)[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
