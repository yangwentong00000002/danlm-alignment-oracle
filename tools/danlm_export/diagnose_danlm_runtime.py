#!/usr/bin/env python3
"""Compare original DanLM runtime, direct model forward, and portable model."""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import sys
import traceback
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--danlm-dir", default="res/DanLM-main")
    parser.add_argument(
        "--checkpoint",
        default="res/DanLM-main/ckpts/DanLM_v1/dansformer_v1_best_eval.pt",
    )
    parser.add_argument(
        "--output",
        default="tools/danlm_export/work/danlm_runtime_diagnostics.json",
    )
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--first-player", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def to_python(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return [to_python(item) for item in value]
    if isinstance(value, list):
        return [to_python(item) for item in value]
    return value


def first_row(value: Any) -> list[float]:
    data = to_python(value)
    while isinstance(data, list) and len(data) == 1 and isinstance(data[0], list):
        data = data[0]
    if not isinstance(data, list):
        raise TypeError(f"not a list-like output: {type(data)!r}")
    return [float(item) for item in data]


def signatures(obj: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in dir(obj):
        if name.startswith("__"):
            continue
        attr = getattr(obj, name, None)
        if not callable(attr):
            continue
        try:
            result[name] = str(inspect.signature(attr))
        except Exception:
            result[name] = "<signature unavailable>"
    return result


def diff_summary(expected: list[float], actual: list[float]) -> dict[str, Any]:
    count = min(len(expected), len(actual))
    diffs = [actual[i] - expected[i] for i in range(count)]
    max_index = max(range(count), key=lambda i: abs(diffs[i])) if count else -1
    return {
        "count": count,
        "max_abs_diff": abs(diffs[max_index]) if max_index >= 0 else None,
        "max_abs_diff_index": max_index,
        "first5_expected": expected[:5],
        "first5_actual": actual[:5],
        "first5_diff": diffs[:5],
        "expected_argmax": max(range(len(expected)), key=expected.__getitem__)
        if expected else None,
        "actual_argmax": max(range(len(actual)), key=actual.__getitem__)
        if actual else None,
    }


def run_direct_model(model: Any, sample: dict[str, Any]) -> list[float]:
    import torch

    inputs = sample["input"]
    with torch.no_grad():
        token_ids = torch.tensor([inputs["token_ids"]], dtype=torch.long)
        seq_lens = torch.tensor([inputs["seq_lens"]], dtype=torch.long)
        hand = torch.tensor([inputs["hand"]], dtype=torch.float32)
        action = torch.tensor([inputs["action"]], dtype=torch.float32)
        self_abs = torch.tensor([inputs["self_abs"]], dtype=torch.long)
        return first_row(model(token_ids, seq_lens, hand, action, self_abs))


def error_payload(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def main() -> None:
    args = parse_args()
    danlm_dir = Path(args.danlm_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    sys.path.insert(0, str(danlm_dir))
    sys.path.insert(0, str(Path("tools/danlm_export").resolve()))

    import torch
    from danlm_v1_portable import load_from_checkpoint as load_portable
    from export_danlm_alignment_sample import (
        call_first_success,
        flatten_batch_vector,
        matrix,
        normalize_tokenize_result,
    )
    from danzero.engine.cards import deal_hands
    from danzero.engine.game import GuanDanRound
    from danzero.eval.agents import create_agent
    from danzero.model.transformer import TransformerConfig, TransformerQNetwork

    hands = deal_hands(seed=args.seed)
    round_obj = GuanDanRound(
        level=args.level,
        hands=hands,
        first_player=args.first_player,
        team_levels=(args.level, args.level),
    )
    agent = create_agent(f"transformer:{checkpoint}", args.device)
    agent.reset(args.first_player, args.level)
    if hasattr(agent, "notify_tribute"):
        agent.notify_tribute([])
    if hasattr(agent, "notify_start"):
        agent.notify_start(round_obj.state.hands[args.first_player].copy())

    obs = round_obj.get_observation()
    if obs is None:
        raise RuntimeError("first observation is unavailable")

    agent_q = [float(value) for value in flatten_batch_vector(
        agent.get_q_values(obs, round_obj), "agent_q_values")]
    tokenized = call_first_success(
        agent,
        "_tokenize",
        [(round_obj,), (round_obj, obs), (obs, round_obj), (obs,), tuple()],
    )
    token_ids, seq_len = normalize_tokenize_result(tokenized)
    legal_plays = matrix(obs.legal_plays, "legal_plays")
    hand = [float(value) for value in flatten_batch_vector(
        round_obj.state.hands[int(obs.player)], "hand")]
    sample = {
        "input": {
            "token_ids": token_ids,
            "seq_lens": seq_len,
            "hand": hand,
            "action": [[float(value) for value in row] for row in legal_plays],
            "self_abs": int(obs.player),
        },
        "output": {
            "q_values": agent_q,
            "selected_index": max(range(len(agent_q)), key=agent_q.__getitem__),
        },
    }

    ckpt = torch.load(checkpoint, weights_only=False, map_location="cpu")
    raw_cfg = ckpt.get("model_config") or ckpt.get("config", {})
    valid = {field.name for field in dataclasses.fields(TransformerConfig)}
    tcfg = TransformerConfig(**{k: v for k, v in raw_cfg.items() if k in valid})
    original_model = TransformerQNetwork(tcfg)
    original_model.load_state_dict(ckpt.get("model_state_dict") or ckpt["model"])
    original_model.eval()

    result: dict[str, Any] = {
        "source": {
            "checkpoint": str(checkpoint),
            "seed": args.seed,
            "level": args.level,
            "first_player": args.first_player,
            "player": int(obs.player),
        },
        "torch_version": torch.__version__,
        "agent_class": type(agent).__name__,
        "agent_signatures": {
            name: value for name, value in signatures(agent).items()
            if "q" in name or "token" in name or "play" in name
        },
        "original_model_class": type(original_model).__name__,
        "original_model_repr": repr(original_model),
        "original_model_signatures": {
            name: value for name, value in signatures(original_model).items()
            if "forward" in name or "encode" in name or "q" in name
        },
        "sample": sample,
    }

    try:
        original_q = run_direct_model(original_model, sample)
        result["original_direct_q_values"] = original_q
        result["agent_vs_original_direct"] = diff_summary(agent_q, original_q)
    except Exception as exc:
        result["original_direct_error"] = error_payload(exc)

    try:
        portable_model = load_portable(str(checkpoint))
        portable_q = run_direct_model(portable_model, sample)
        result["portable_direct_q_values"] = portable_q
        result["agent_vs_portable_direct"] = diff_summary(agent_q, portable_q)
        if "original_direct_q_values" in result:
            result["original_direct_vs_portable_direct"] = diff_summary(
                result["original_direct_q_values"], portable_q)
    except Exception as exc:
        result["portable_direct_error"] = error_payload(exc)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
