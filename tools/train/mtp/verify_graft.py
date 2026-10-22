"""Verify that a graft did not change the source file.

The graft copies every source tensor and every metadata field, and appends block 32. This
compares the source tensors byte for byte, and the metadata field by field.

Usage:
  python -m tools.train.mtp.verify_graft <source.gguf> <grafted.gguf>
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import gguf
import numpy as np


def tensor_hashes(path: Path) -> dict[str, str]:
    reader = gguf.GGUFReader(str(path))
    out = {}
    for tensor in reader.tensors:
        data = np.asarray(tensor.data)
        out[tensor.name] = hashlib.sha256(data.tobytes()).hexdigest()
    return out


def metadata(path: Path) -> dict[str, str]:
    reader = gguf.GGUFReader(str(path))
    out = {}
    for key, field in reader.fields.items():
        if key.startswith("GGUF."):
            continue
        try:
            parts = [field.parts[i].tobytes() for i in field.data]
        except Exception:
            parts = [b""]
        out[key] = hashlib.sha256(b"|".join(parts)).hexdigest()
    return out


def main() -> int:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    hs, hd = tensor_hashes(src), tensor_hashes(dst)
    ms, md = metadata(src), metadata(dst)

    added = sorted(set(hd) - set(hs))
    missing = sorted(set(hs) - set(hd))
    changed = sorted(n for n in set(hs) & set(hd) if hs[n] != hd[n])

    meta_missing = sorted(set(ms) - set(md))
    meta_changed = sorted(k for k in set(ms) & set(md) if ms[k] != md[k])
    meta_added = sorted(set(md) - set(ms))

    report = {
        "source_tensors": len(hs),
        "grafted_tensors": len(hd),
        "added": len(added),
        "added_names": added,
        "missing": missing,
        "changed": changed,
        "metadata_missing": meta_missing,
        "metadata_changed": meta_changed,
        "metadata_added": meta_added,
    }
    report["verifier_unchanged"] = not missing and not changed
    report["metadata_preserved"] = not meta_missing and not meta_changed
    report["head_added"] = len(added) == 15 and all(n.startswith("blk.32.") for n in added)
    report["pass"] = (
        report["verifier_unchanged"]
        and report["metadata_preserved"]
        and report["head_added"]
    )
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
