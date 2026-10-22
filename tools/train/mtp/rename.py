"""Rename a GGUF model.

The name is metadata, not identity: llama.cpp loads a model by path and shows
`general.name` in the UI. `general.basename` and `general.finetune` are not read by llama.cpp
but are used by some third-party tools, so they are set too.

Provenance is left alone on purpose. `general.base_model.*`, `general.dataset.*`,
`general.source.*` and `general.license` record where the weights came from, and a rename
must not erase them.

Usage:
  python -m tools.train.mtp.rename <src.gguf> <dst.gguf> [--name Athena-4B-MTP]
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import gguf
import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter

from tools.train.mtp.graft import PLAIN_TYPES, _decode, add_source_tensor

F32 = GGMLQuantizationType.F32


def _copy_metadata(reader: GGUFReader, writer: GGUFWriter, overrides: dict[str, object]) -> None:
    """Copy every metadata field, replacing the ones in overrides.

    field.types is [ARRAY, ELEMENT] for an array and [ELEMENT] for a scalar, and field.data
    lists the part indexes that hold the values.
    """
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        types = list(field.types)
        if not types:
            continue
        if key in overrides:
            value = overrides[key]
            if isinstance(value, list):
                writer.add_key_value(
                    key, value, GGUFValueType.ARRAY, sub_type=GGUFValueType.STRING
                )
            else:
                writer.add_key_value(key, value, GGUFValueType.STRING)
            continue
        if len(types) == 2 and types[0] == GGUFValueType.ARRAY:
            vtype, sub_type = types[0], types[1]
            value = [_decode(field.parts[i], sub_type) for i in field.data]
        else:
            vtype, sub_type = types[0], None
            value = _decode(field.parts[field.data[0]], vtype)
        writer.add_key_value(key, value, vtype, sub_type=sub_type)


def rename(src: Path, dst: Path, name: str, finetune: str = "MTP", tags: list[str] | None = None) -> Path:
    reader = GGUFReader(str(src))
    arch_field = reader.get_field("general.architecture")
    arch = str(_decode(arch_field.parts[arch_field.data[0]], GGUFValueType.STRING))

    old_tags = []
    for key, field in reader.fields.items():
        if key == "general.tags":
            old_tags = [
                _decode(field.parts[i], GGUFValueType.STRING) for i in field.data
            ]
    new_tags = list(old_tags)
    for tag in tags or []:
        if tag not in new_tags:
            new_tags.append(tag)

    overrides: dict[str, object] = {
        "general.name": name,
        "general.basename": name,
        "general.finetune": finetune,
        "general.tags": new_tags,
    }

    tmp = dst.with_suffix(".gguf.tmp")
    writer = GGUFWriter(str(tmp), arch)
    _copy_metadata(reader, writer, overrides)
    for tensor in reader.tensors:
        add_source_tensor(writer, tensor)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    check = GGUFReader(str(tmp))
    names = {t.name for t in check.tensors}
    del check
    gc.collect()
    if len(names) != len(reader.tensors):
        raise RuntimeError(f"{tmp}: tensor count changed")
    tmp.replace(dst)
    return dst


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--name", default="Athena-4B-MTP")
    parser.add_argument("--finetune", default="MTP")
    parser.add_argument("--tag", action="append", default=None)
    args = parser.parse_args()
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    print(rename(args.src, args.dst, args.name, args.finetune, args.tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
