"""Graft the MTP head onto a Qwen3.5-4B GGUF that has no block 32.

The head is block 32 of the MTP model: 11 block tensors plus 4 `blk.32.nextn.*` tensors.
llama.cpp detects the MTP path from the tensor names, not from a metadata flag, so the graft
is "copy every tensor, then append block 32".

The head is not quantized: the 8 2D tensors are written at f16 and the 7 norms stay F32.

Usage:
  python -m tools.train.mtp.graft <src.gguf> <dst.gguf> [--head <safetensors shard>]
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

import gguf
import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter, quants
from gguf.quants import quant_shape_to_byte_shape
from safetensors import safe_open

# PyTorch state-dict name -> GGUF tensor name. The head is block 32.
GGUF_NAME_MAP = {
    "pre_fc_norm_embedding.weight": "blk.32.nextn.enorm.weight",
    "pre_fc_norm_hidden.weight": "blk.32.nextn.hnorm.weight",
    "fc.weight": "blk.32.nextn.eh_proj.weight",
    "norm.weight": "blk.32.nextn.shared_head_norm.weight",
    "layers.0.input_layernorm.weight": "blk.32.attn_norm.weight",
    "layers.0.post_attention_layernorm.weight": "blk.32.post_attention_norm.weight",
    "layers.0.self_attn.q_proj.weight": "blk.32.attn_q.weight",
    "layers.0.self_attn.k_proj.weight": "blk.32.attn_k.weight",
    "layers.0.self_attn.v_proj.weight": "blk.32.attn_v.weight",
    "layers.0.self_attn.o_proj.weight": "blk.32.attn_output.weight",
    "layers.0.self_attn.q_norm.weight": "blk.32.attn_q_norm.weight",
    "layers.0.self_attn.k_norm.weight": "blk.32.attn_k_norm.weight",
    "layers.0.mlp.gate_proj.weight": "blk.32.ffn_gate.weight",
    "layers.0.mlp.up_proj.weight": "blk.32.ffn_up.weight",
    "layers.0.mlp.down_proj.weight": "blk.32.ffn_down.weight",
}

# The 7 one-dimensional head tensors. They stay F32.
HEAD_NORM_NAMES = {
    "blk.32.nextn.enorm.weight",
    "blk.32.nextn.hnorm.weight",
    "blk.32.nextn.shared_head_norm.weight",
    "blk.32.attn_norm.weight",
    "blk.32.attn_q_norm.weight",
    "blk.32.attn_k_norm.weight",
    "blk.32.post_attention_norm.weight",
}

HEAD_QUANTS = {
    "F16": GGMLQuantizationType.F16,
}
F32 = GGMLQuantizationType.F32
F16 = GGMLQuantizationType.F16
BF16 = GGMLQuantizationType.BF16
PLAIN_TYPES = (F32, F16, BF16)


def add_source_tensor(writer: GGUFWriter, tensor) -> None:
    """Copy one tensor out of a source GGUF.

    The reader reports shapes in GGUF ne order ([ne0, ne1]); the writer wants numpy order
    ([rows, cols]) and reverses it when it writes the dimensions array. For a quantized
    tensor the writer wants the *byte* shape and converts it back to the logical shape
    itself. The bytes are copied verbatim, so the result is identical to the source.
    """
    shape = [int(x) for x in tensor.shape]
    numpy_shape = list(reversed(shape))
    qtype = tensor.tensor_type
    if qtype in PLAIN_TYPES:
        writer.add_tensor(
            tensor.name,
            np.asarray(tensor.data),
            raw_shape=numpy_shape,
            raw_dtype=qtype,
        )
    else:
        writer.add_tensor(
            tensor.name,
            np.asarray(tensor.data),
            raw_shape=quant_shape_to_byte_shape(numpy_shape, qtype),
            raw_dtype=qtype,
        )


def load_head(shard: Path) -> dict[str, torch.Tensor]:
    """The 15 mtp.* tensors, with the 'mtp.' prefix stripped."""
    head: dict[str, torch.Tensor] = {}
    with safe_open(str(shard), framework="pt") as f:
        for key in f.keys():
            if key.startswith("mtp."):
                head[key[len("mtp."):]] = f.get_tensor(key)
    missing = sorted(set(GGUF_NAME_MAP) - set(head))
    if missing:
        raise RuntimeError(f"shard is missing MTP tensors: {missing}")
    extra = sorted(set(head) - set(GGUF_NAME_MAP))
    if extra:
        raise RuntimeError(f"shard has unexpected MTP tensors: {extra}")
    return head


def _decode(part, vtype) -> object:
    if vtype == GGUFValueType.STRING:
        raw = part.tobytes() if hasattr(part, "tobytes") else bytes(part)
        return raw.decode("utf-8", "replace")
    if hasattr(part, "tolist"):
        value = part.tolist()
        if isinstance(value, list) and len(value) == 1:
            return value[0]
        return value
    return part


def copy_metadata(reader: GGUFReader, writer: GGUFWriter) -> None:
    """Copy every metadata field from the source.

    field.types is [ARRAY, ELEMENT] for an array and [ELEMENT] for a scalar, and
    field.data lists the part indexes that hold the values.
    """
    for key, field in reader.fields.items():
        # The writer owns the GGUF.* bookkeeping keys and the architecture key.
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        types = list(field.types)
        if not types:
            continue
        if len(types) == 2 and types[0] == GGUFValueType.ARRAY:
            vtype, sub_type = types[0], types[1]
            value = [_decode(field.parts[i], sub_type) for i in field.data]
        else:
            vtype, sub_type = types[0], None
            value = _decode(field.parts[field.data[0]], vtype)
        writer.add_key_value(key, value, vtype, sub_type=sub_type)


def graft(src: Path, shard: Path, dst: Path) -> Path:
    reader = GGUFReader(str(src))
    arch_field = reader.get_field("general.architecture")
    arch = str(_decode(arch_field.parts[arch_field.data[0]], GGUFValueType.STRING))

    existing = {t.name for t in reader.tensors}
    if any(n.startswith("blk.32.") for n in existing):
        raise RuntimeError(f"{src} already has a block 32; use the splice path instead")

    head = load_head(shard)
    tmp = dst.with_suffix(".gguf.tmp")
    writer = GGUFWriter(str(tmp), arch)
    copy_metadata(reader, writer)

    for tensor in reader.tensors:
        add_source_tensor(writer, tensor)

    for key, gguf_name in GGUF_NAME_MAP.items():
        # The torch weight is [out, in], which is numpy order already. The writer reverses
        # it, so the file ends up with the GGUF ne order [in, out].
        # The head is not quantized: the 8 2D tensors are f16, the 7 norms stay F32.
        data = head[key].float().numpy()
        numpy_shape = list(head[key].shape)
        if gguf_name in HEAD_NORM_NAMES:
            # llama.cpp multiplies the normalised value by the weight directly, while HF's
            # Qwen3.5 normaliser multiplies by (1 + weight). The official converter stores
            # w + 1 for exactly this reason; a verbatim copy leaves every norm off by one
            # and the head computes a different function than the checkpoint describes.
            writer.add_tensor(gguf_name, data + 1.0, raw_shape=numpy_shape, raw_dtype=F32)
        else:
            writer.add_tensor(
                gguf_name, data.astype(np.float16), raw_shape=numpy_shape, raw_dtype=F16
            )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    check = GGUFReader(str(tmp))
    names = [t.name for t in check.tensors]
    blocks = {int(t.name.split(".")[1]) for t in check.tensors if t.name.startswith("blk.")}
    nextn = [n for n in names if n.startswith("blk.32.nextn")]
    if len(names) != 441:
        raise RuntimeError(f"{tmp}: {len(names)} tensors, expected 441")
    if max(blocks) != 32 or len(blocks) != 33:
        raise RuntimeError(f"{tmp}: blocks {min(blocks)}..{max(blocks)} ({len(blocks)}), expected 0..32 (33)")
    if len(nextn) != 4:
        raise RuntimeError(f"{tmp}: {len(nextn)} blk.32.nextn tensors, expected 4")
    missing = sorted(set(GGUF_NAME_MAP.values()) - set(names))
    if missing:
        raise RuntimeError(f"{tmp}: missing head tensors {missing}")
    # Windows keeps the memmap open; release it before the rename.
    del check
    gc.collect()
    tmp.replace(dst)
    return dst


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument(
        "--head",
        type=Path,
        default=Path("models/qwen35-mtp/hf/model-00003-of-00003.safetensors"),
    )
    args = parser.parse_args()
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    print(graft(args.src, args.head, args.dst))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
