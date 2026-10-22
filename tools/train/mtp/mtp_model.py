"""The MTP head: the verifier's block 32, as a PyTorch module.

Built from the *installed* transformers, not from memory. Two details of this version differ
from the obvious reading of the Qwen3.5 source and both are load-bearing:

- `Qwen3_5DecoderLayer.forward` takes `position_embeddings=(cos, sin)`, not `position_ids`, and
  it returns a **tensor**, not a tuple. Slicing `[0]` off the result takes the first batch
  element, not the hidden states.
- `Qwen3_5DecoderLayer` picks its block type from `config.layer_types[layer_idx]`, so the layer
  config must set `layer_types = ["full_attention"]` and `layer_idx = 0`.

The GGUF loader has one trap of its own, documented at `_to_float32`.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
)

import gguf
from gguf.quants import dequantize
from huggingface_hub import snapshot_download

from tools.train.mtp.device import resolve_device  # noqa: F401  (re-exported for callers)


def hf_shards(model_id: str, hf_dir: Path | None = None) -> list[Path]:
    """The shards that hold the `mtp.*` tensors, from a local dir when one is complete.

    The checkpoint has 738 keys over three shards. All 15 mtp.* keys live in
    model-00003-of-00003.safetensors (241 MB); the other two shards hold 9 GB of weights
    the head never reads. Read the index, keep the shards that carry mtp.*, fetch those.
    """
    if hf_dir is not None:
        index_path = hf_dir / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            wanted = {
                name
                for name in index.get("weight_map", {})
                if name.startswith("mtp.")
            }
            shards = sorted({index["weight_map"][name] for name in wanted})
            if wanted and all((hf_dir / shard).exists() for shard in shards):
                return [hf_dir / shard for shard in shards]
    root = Path(snapshot_download(model_id, allow_patterns=["model.safetensors.index.json"]))
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    wanted = {
        name for name, shard in index.get("weight_map", {}).items() if name.startswith("mtp.")
    }
    if not wanted:
        raise RuntimeError(f"{model_id} has no mtp.* tensors")
    shards = sorted({index["weight_map"][name] for name in wanted})
    root = Path(snapshot_download(model_id, allow_patterns=shards))
    return [root / shard for shard in shards]


class MTPHead(nn.Module):
    """The MTP head: the verifier's block 32 plus 4 nextn tensors.

    The layer is a full_attention Qwen3_5 block, so it is built from the text config with
    layer_types = ["full_attention"]. The full-attention block's q projection outputs
    num_attention_heads * head_dim * 2 = 8192: the second half is a gate, not a second query,
    and the attention output is multiplied by sigmoid(gate).

    The forward follows `docs/notes/mtp-forward.md`: hidden is the trunk's **post-output-norm**
    state, the concat is `[norm(embed), norm(hidden)]`, and the block has its own attention.
    """

    def __init__(self, text_config):
        super().__init__()
        cfg = copy.deepcopy(text_config)
        cfg.num_hidden_layers = 1
        cfg.layer_types = ["full_attention"]
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.fc = nn.Linear(2 * cfg.hidden_size, cfg.hidden_size, bias=False)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(cfg, layer_idx=0)])
        self.norm = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        # No weights, only non-persistent buffers, so it does not appear in the state dict.
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(cfg)

    def forward(self, hidden, embed, attention_mask, position_ids):
        x = self.fc(
            torch.cat(
                [self.pre_fc_norm_embedding(embed), self.pre_fc_norm_hidden(hidden)], dim=-1
            )
        )
        # The rotary expands a (1, n) position_ids to its three mrope sections itself.
        position_embeddings = self.rotary_emb(x, position_ids)
        x = self.layers[0](
            x,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return self.norm(x)


def load_mtp_weights(paths: list[Path], head: MTPHead) -> None:
    state = {}
    for path in paths:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("mtp."):
                    state[key[len("mtp."):]] = f.get_tensor(key)
    missing, unexpected = head.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"missing MTP tensors: {missing}")
    if unexpected:
        raise RuntimeError(f"unexpected MTP tensors: {unexpected}")


GGUF_NAME_MAP = {
    # The MTP head IS block 32. It is a full-attention decoder block, identical in shape
    # to blk.31, plus 4 nextn tensors. The MTP model's GGUF has 4 blk.32.nextn.* tensors
    # and 11 blk.32.* tensors; together they are the 15 mtp.* tensors of the checkpoint.
    # Verified against the real file on 2026-10-20.
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

# The 7 one-dimensional head tensors. They stay F32 in every artifact.
GGUF_HEAD_NORM_NAMES = {
    "blk.32.nextn.enorm.weight",
    "blk.32.nextn.hnorm.weight",
    "blk.32.nextn.shared_head_norm.weight",
    "blk.32.attn_norm.weight",
    "blk.32.attn_q_norm.weight",
    "blk.32.attn_k_norm.weight",
    "blk.32.post_attention_norm.weight",
}


def _to_float32(tensor) -> np.ndarray:
    """Read a GGUF tensor as float32, in PyTorch layout.

    **Do not reshape the result.** `dequantize()` returns `(ne1, ne0)` — the same
    `(out_features, in_features)` orientation PyTorch wants — while `tuple(tensor.shape)` is
    the GGUF metadata order `(ne0, ne1)`. Reshaping to `tensor.shape` transposes every
    weight, silently, and for a non-square matrix the result still has the right inverse
    shape. That is a head that trains to a plausible-looking loss while computing something
    else entirely.

    `dequantize()` also only understands the quantized types, so F16/F32/BF16 are converted
    directly; their numpy shape is already the PyTorch one.
    """
    plain = (
        gguf.GGMLQuantizationType.F32,
        gguf.GGMLQuantizationType.F16,
        gguf.GGMLQuantizationType.BF16,
    )
    if tensor.tensor_type in plain:
        return np.array(tensor.data, dtype=np.float32, copy=True)
    return np.array(
        dequantize(tensor.data, tensor.tensor_type), dtype=np.float32, copy=True
    )


def load_head_from_gguf(
    gguf_path: Path, head: MTPHead, strict: bool = True, dtype: torch.dtype = torch.bfloat16
) -> None:
    """Load the 15 head tensors out of a GGUF, de-quantized.

    The fidelity probe uses this so that the PyTorch side and the llama.cpp side run on the
    very same weights. A mismatch here is a false G1a failure.

    Every tensor's shape is checked against the parameter it fills. A transpose is the one
    mistake that would otherwise pass every other check in the pipeline, so it is an error.
    """
    reader = gguf.GGUFReader(str(gguf_path))
    by_gguf_name = {gguf_name: key for key, gguf_name in GGUF_NAME_MAP.items()}
    params = dict(head.named_parameters())
    found: dict[str, torch.Tensor] = {}
    for tensor in reader.tensors:
        key = by_gguf_name.get(tensor.name)
        if key is None:
            continue
        raw = _to_float32(tensor)
        if tensor.name in GGUF_HEAD_NORM_NAMES:
            # The GGUF stores w_gguf = w_hf + 1 for Qwen3.5 norms (llama.cpp multiplies by
            # the weight, HF by 1 + weight). Undo it in f32, then cast: subtracting in bf16
            # would quantise a value near 1 to 8 mantissa bits.
            raw = raw - 1.0
        arr = torch.from_numpy(raw).to(dtype)
        want = tuple(params[key].shape)
        if tuple(arr.shape) != want:
            raise RuntimeError(
                f"{tensor.name} -> {key}: GGUF gives {tuple(arr.shape)}, "
                f"the parameter is {want}. The weight is transposed or the map is wrong."
            )
        found[key] = arr
    missing = sorted(set(GGUF_NAME_MAP) - set(found))
    if missing and strict:
        raise RuntimeError(f"GGUF is missing MTP tensors: {missing}")
    head.load_state_dict(found, strict=False)


def load_lm_head(gguf_path: Path, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """The tied output head: `token_embd.weight`, `(vocab, hidden)`, de-quantized.

    `dtype` matters for the fidelity probe: Q6_K dequantized to bf16 loses three mantissa
    bits, and that error reaches both the input embedding and the output logits.
    """
    reader = gguf.GGUFReader(str(gguf_path))
    for tensor in reader.tensors:
        if tensor.name != "token_embd.weight":
            continue
        return torch.from_numpy(_to_float32(tensor)).to(dtype)
    raise KeyError("token_embd.weight not found")
