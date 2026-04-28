#!/usr/bin/env python3
"""Inspect a DanLM checkpoint without importing torch or platform modules."""

from __future__ import annotations

import argparse
import json
import pickletools
import zipfile
from pathlib import Path


MODEL_KEYS = {
    "state_token_dim",
    "n_attn_blocks",
    "n_query_heads",
    "n_kv_heads",
    "qk_dim",
    "v_dim",
    "ffn_hidden",
    "max_seq_len",
    "seq_buckets",
    "hand_emb_dim",
    "hand_hidden_dim",
    "n_hand_hiddens",
    "q_head_hidden_dim",
    "n_q_head_hiddens",
    "action_chunk_size",
    "qval_buckets",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="res/DanLM-main/ckpts/DanLM_v1/dansformer_v1_best_eval.pt",
        help="Path to DanLM .pt checkpoint",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    return parser.parse_args()


def read_pickle(checkpoint: Path) -> bytes:
    if not zipfile.is_zipfile(checkpoint):
        raise ValueError(f"Checkpoint is not a torch zip archive: {checkpoint}")
    with zipfile.ZipFile(checkpoint) as archive:
        data_name = next(
            name for name in archive.namelist() if name.endswith("data.pkl")
        )
        return archive.read(data_name)


def ops_from_pickle(data: bytes) -> list[tuple[str, object, int]]:
    return [(opcode.name, arg, pos) for opcode, arg, pos in pickletools.genops(data)]


def next_value(ops: list[tuple[str, object, int]], start: int) -> tuple[object, int]:
    i = start
    while i < len(ops) and ops[i][0] in {"BINPUT", "LONG_BINPUT", "MEMOIZE"}:
        i += 1
    if i >= len(ops):
        return None, i
    name, arg, _ = ops[i]
    if name in {"BININT", "BININT1", "BININT2", "LONG1", "LONG4"}:
        return arg, i + 1
    if name == "BINFLOAT":
        return arg, i + 1
    if name == "NEWTRUE":
        return True, i + 1
    if name == "NEWFALSE":
        return False, i + 1
    if name in {"BINUNICODE", "SHORT_BINUNICODE", "UNICODE"}:
        return arg, i + 1
    if name == "MARK":
        values: list[object] = []
        i += 1
        while i < len(ops):
            current, current_arg, _ = ops[i]
            if current in {"TUPLE", "TUPLE1", "TUPLE2", "TUPLE3"}:
                return tuple(values), i + 1
            if current in {"BININT", "BININT1", "BININT2", "LONG1", "LONG4"}:
                values.append(current_arg)
            elif current == "BINFLOAT":
                values.append(current_arg)
            elif current in {"BINUNICODE", "SHORT_BINUNICODE", "UNICODE"}:
                values.append(current_arg)
            i += 1
    return None, i + 1


def extract_config(ops: list[tuple[str, object, int]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for i, (name, arg, _) in enumerate(ops):
        if name not in {"BINUNICODE", "SHORT_BINUNICODE", "UNICODE"}:
            continue
        if arg not in MODEL_KEYS:
            continue
        value, _ = next_value(ops, i + 1)
        result[str(arg)] = value
    return result


def extract_parameter_shapes(ops: list[tuple[str, object, int]]) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for i, (name, arg, _) in enumerate(ops):
        if name not in {"BINUNICODE", "SHORT_BINUNICODE", "UNICODE"}:
            continue
        if not isinstance(arg, str) or "." not in arg:
            continue
        if not (arg.endswith(".weight") or arg.endswith(".bias")):
            continue

        storage_idx = None
        for j in range(i + 1, min(i + 40, len(ops))):
            if ops[j][0] == "BINPERSID":
                storage_idx = j
                break
        if storage_idx is None:
            continue

        dims: list[int] = []
        for j in range(storage_idx + 1, min(storage_idx + 16, len(ops))):
            op_name, op_arg, _ = ops[j]
            if op_name in {"BININT", "BININT1", "BININT2", "LONG1", "LONG4"}:
                dims.append(int(op_arg))
            elif op_name in {"TUPLE", "TUPLE1", "TUPLE2", "TUPLE3"}:
                if len(dims) > 1:
                    shapes[arg] = dims[1:]
                break
    return shapes


def summarize(checkpoint: Path) -> dict[str, object]:
    data = read_pickle(checkpoint)
    ops = ops_from_pickle(data)
    shapes = extract_parameter_shapes(ops)
    config = extract_config(ops)
    token_shape = shapes.get("token_emb.weight", [])
    return {
        "checkpoint": str(checkpoint),
        "pickle_bytes": len(data),
        "config": config,
        "parameter_count": len(shapes),
        "selected_shapes": {
            name: shapes.get(name)
            for name in (
                "token_emb.weight",
                "hand_mlp.0.weight",
                "q_head.0.weight",
                "q_head.6.weight",
            )
        },
        "derived_contract": {
            "vocab_size": token_shape[0] if len(token_shape) == 2 else None,
            "token_ids": "int64[B,T], 0=padding, T<=max_seq_len",
            "seq_lens": "int64[B]",
            "hand": "float32[B,54]",
            "action": "float32[B,N,80]",
            "self_abs": "int64[B], current player seat 0..3",
            "q_values": "float32[B,N], one score per legal action",
        },
    }


def main() -> None:
    args = parse_args()
    summary = summarize(Path(args.checkpoint))
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    print(f"checkpoint={summary['checkpoint']}")
    print(f"pickleBytes={summary['pickle_bytes']}")
    print(f"parameterCount={summary['parameter_count']}")
    print("config:")
    for key, value in summary["config"].items():
        print(f"  {key}={value}")
    print("selectedShapes:")
    for key, value in summary["selected_shapes"].items():
        print(f"  {key}={value}")
    print("derivedContract:")
    for key, value in summary["derived_contract"].items():
        print(f"  {key}={value}")


if __name__ == "__main__":
    main()
