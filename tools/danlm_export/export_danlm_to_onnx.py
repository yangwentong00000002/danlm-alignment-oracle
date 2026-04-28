#!/usr/bin/env python3
"""Export DanLM V1 to the Android ONNX asset used by the app.

The target of this script is always an Android-loadable ONNX file, normally
GuanDan/app/src/main/assets/guandan_ai/danlm_v1.onnx. The conversion host only
needs to provide a runnable DanLM Python runtime; it is not the runtime target.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--danlm-dir",
        default="res/DanLM-main",
        help="Path to the DanLM project root",
    )
    parser.add_argument(
        "--checkpoint",
        default="res/DanLM-main/ckpts/DanLM_v1/dansformer_v1_best_eval.pt",
        help="Path to DanLM V1 .pt checkpoint",
    )
    parser.add_argument(
        "--output",
        default="GuanDan/app/src/main/assets/guandan_ai/danlm_v1.onnx",
        help="Output ONNX path",
    )
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--dummy-seq-len", type=int, default=1280)
    parser.add_argument("--dummy-actions", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    danlm_dir = Path(args.danlm_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()

    if not danlm_dir.exists():
        raise FileNotFoundError(f"DanLM directory not found: {danlm_dir}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    try:
        import torch
        from danlm_v1_portable import load_from_checkpoint
    except Exception as exc:
        raise RuntimeError(
            "Unable to import local DanLM portable exporter dependencies."
        ) from exc

    # The network is reconstructed from checkpoint shapes and docstrings, so
    # conversion does not depend on upstream platform-specific Python modules.
    wrapper = load_from_checkpoint(str(checkpoint))
    token_ids = torch.zeros(1, args.dummy_seq_len, dtype=torch.long)
    seq_lens = torch.ones(1, dtype=torch.long) * args.dummy_seq_len
    hand = torch.zeros(1, 54, dtype=torch.float32)
    action = torch.zeros(1, args.dummy_actions, 80, dtype=torch.float32)
    self_abs = torch.zeros(1, dtype=torch.long)

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (token_ids, seq_lens, hand, action, self_abs),
        str(output),
        input_names=["token_ids", "seq_lens", "hand", "action", "self_abs"],
        output_names=["q_values"],
        dynamic_axes={
            "token_ids": {0: "batch", 1: "seq_len"},
            "seq_lens": {0: "batch"},
            "hand": {0: "batch"},
            "action": {0: "batch", 1: "legal_action_count"},
            "self_abs": {0: "batch"},
            "q_values": {0: "batch", 1: "legal_action_count"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        external_data=False,
    )
    print(f"exported {output}")


if __name__ == "__main__":
    main()
