#!/usr/bin/env python3
"""Export DanLM V1 Java-alignment samples.

The output JSON captures the exact tensors Java must reproduce before Android
inference can be trusted:

    token_ids, seq_lens, hand[54], action[N,80], self_abs, q_values[N]

Run this in the same conversion environment that exports the Android ONNX
asset. The conversion host must provide a runnable DanLM Python runtime; the
target runtime remains Android.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
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
        default="tools/danlm_export/work/danlm_alignment_samples.json",
    )
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--first-player", type=int, default=0)
    parser.add_argument(
        "--samples",
        type=int,
        default=8,
        help="Number of consecutive greedy play decisions to export",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--introspect",
        action="store_true",
        help="Print TransformerAgent methods before exporting the sample",
    )
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


def flatten_batch_vector(value: Any, name: str) -> list:
    data = to_python(value)
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], list):
        data = data[0]
    if not isinstance(data, list):
        raise TypeError(f"{name} is not a list-like tensor: {type(data)!r}")
    return data


def matrix(value: Any, name: str) -> list[list[float]]:
    data = to_python(value)
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], list):
        data = data[0]
    if not isinstance(data, list) or not all(isinstance(row, list) for row in data):
        raise TypeError(f"{name} is not a matrix-like tensor")
    return data


def call_first_success(agent: Any, method_name: str, candidates: list[tuple]) -> Any:
    method = getattr(agent, method_name, None)
    if method is None:
        raise AttributeError(f"TransformerAgent has no {method_name} method")
    errors: list[str] = []
    for args in candidates:
        try:
            return method(*args)
        except TypeError as exc:
            errors.append(f"{method_name}{args}: {exc}")
    raise RuntimeError(
        "Unable to call DanLM private method. Tried:\n" + "\n".join(errors)
    )


def describe_agent(agent: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in dir(agent):
        if not ("token" in name or "q" in name or "play" in name):
            continue
        attr = getattr(agent, name, None)
        if not callable(attr):
            continue
        try:
            signature = str(inspect.signature(attr))
        except Exception:
            signature = "<signature unavailable>"
        result[name] = signature
    return result


def card_ids_from_count_vector(value: Any) -> list[int]:
    data = flatten_batch_vector(value, "card_count_vector")
    result: list[int] = []
    for card_id, count in enumerate(data[:54]):
        repeats = int(round(float(count)))
        for _ in range(repeats):
            result.append(card_id)
    return result


def card_string(card_id: int, card_int_to_str: Any) -> str:
    if callable(card_int_to_str):
        text = card_int_to_str(int(card_id))
    else:
        text = card_int_to_str[int(card_id)]
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    return str(text)


def card_strings(card_ids: list[int], card_int_to_str: Any) -> list[str]:
    result: list[str] = []
    for card_id in card_ids:
        result.append(card_string(card_id, card_int_to_str))
    return result


def normalize_play_type(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def rank_value(rank_idx: int | None) -> int | None:
    if rank_idx is None:
        return None
    if rank_idx == 13:
        return 16
    if rank_idx == 14:
        return 17
    if 0 <= rank_idx <= 12:
        return 2 + rank_idx
    return None


def describe_play(play: Any, level: int, play_type_of: Any, level_rank_index: Any,
                  order_cards_in_play: Any, card_int_to_str: Any) -> dict[str, Any]:
    row = flatten_batch_vector(play, "play")
    play_type = normalize_play_type(play_type_of(play))
    rank_idx = None
    if len(row) >= 80:
        rank_vector = row[65:80]
        if any(float(value) > 0 for value in rank_vector):
            rank_idx = max(range(len(rank_vector)),
                           key=lambda index: float(rank_vector[index]))
    if play_type.lower() == "pass":
        cards = []
    else:
        try:
            cards = [int(card_id) for card_id in order_cards_in_play(
                row[:54], play_type, rank_idx, level_rank_index(level))]
        except Exception:
            cards = card_ids_from_count_vector(row[:54])
    return {
        "type": play_type,
        "rank_idx": rank_idx,
        "rank_value": rank_value(rank_idx),
        "card_ints": cards,
        "cards": card_strings(cards, card_int_to_str),
        "vector": [float(item) for item in row],
    }


def normalize_tokenize_result(value: Any) -> tuple[list[int], int]:
    data = to_python(value)
    token_ids: Any
    seq_len: Any = None
    if isinstance(data, (tuple, list)) and len(data) >= 2:
        token_ids, seq_len = data[0], data[1]
    else:
        token_ids = data
    token_ids = flatten_batch_vector(token_ids, "token_ids")
    token_ids = [int(value) for value in token_ids]
    if seq_len is None:
        seq_len = sum(1 for value in token_ids if value != 0)
    else:
        seq_len = flatten_batch_vector(seq_len, "seq_lens")
        seq_len = int(seq_len[0] if isinstance(seq_len, list) else seq_len)
    return token_ids, seq_len


def main() -> None:
    args = parse_args()
    danlm_dir = Path(args.danlm_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    sys.path.insert(0, str(danlm_dir))

    try:
        import numpy as np
        from danzero.encoding.tokenizer import order_cards_in_play
        from danzero.engine.actions import play_type_of
        from danzero.engine.cards import CARD_INT_TO_STR, deal_hands, level_rank_index
        from danzero.engine.game import GuanDanRound
        from danzero.eval.agents import create_agent
    except Exception as exc:
        raise RuntimeError(
            "Unable to import DanLM runtime for Android alignment sample "
            "export. Provide runnable DanLM Python modules in the conversion "
            "environment before enabling the Android model."
        ) from exc

    if args.first_player < 0 or args.first_player > 3:
        raise ValueError("--first-player must be in [0, 3]")
    if args.samples <= 0:
        raise ValueError("--samples must be positive")

    agents = {
        player: create_agent(f"transformer:{checkpoint}", args.device)
        for player in range(4)
    }
    if args.introspect:
        print(json.dumps(describe_agent(agents[0]), ensure_ascii=False, indent=2))

    hands = deal_hands(seed=args.seed)
    round_obj = GuanDanRound(
        level=args.level,
        hands=hands,
        first_player=args.first_player,
        team_levels=(args.level, args.level),
    )
    for player, agent in agents.items():
        agent.reset(player, args.level)
        if hasattr(agent, "notify_tribute"):
            agent.notify_tribute([])
        if hasattr(agent, "notify_start"):
            agent.notify_start(round_obj.state.hands[player].copy())

    samples: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    obs = round_obj.get_observation()
    for step in range(args.samples):
        if obs is None or getattr(round_obj, "done", False):
            break
        player = int(obs.player)
        agent = agents[player]

        q_values = flatten_batch_vector(agent.get_q_values(obs, round_obj), "q_values")
        legal_plays = matrix(obs.legal_plays, "legal_plays")
        hand = flatten_batch_vector(round_obj.state.hands[player], "hand")
        tokenized = call_first_success(
            agent,
            "_tokenize",
            [(round_obj,), (round_obj, obs), (obs, round_obj), (obs,), tuple()],
        )
        token_ids, seq_len = normalize_tokenize_result(tokenized)
        selected_index = int(np.asarray(q_values).argmax())
        legal_action_descriptors = [
            describe_play(np.asarray(play, dtype=np.int8), args.level, play_type_of,
                          level_rank_index, order_cards_in_play, CARD_INT_TO_STR)
            for play in legal_plays
        ]
        current_play = np.asarray(legal_plays[selected_index], dtype=np.int8)

        samples.append({
            "schema_version": 2,
            "source": {
                "checkpoint": str(checkpoint),
                "seed": args.seed,
                "level": args.level,
                "first_player": args.first_player,
                "player": player,
                "step": step,
            },
            "input": {
                "token_ids": token_ids,
                "seq_lens": seq_len,
                "hand": [float(value) for value in hand],
                "action": [[float(value) for value in row] for row in legal_plays],
                "self_abs": player,
            },
            "output": {
                "q_values": [float(value) for value in q_values],
                "selected_index": selected_index,
            },
            "hand_card_ints": card_ids_from_count_vector(hand),
            "hand_cards": card_strings(card_ids_from_count_vector(hand), CARD_INT_TO_STR),
            "history": list(history),
            "legal_actions": legal_action_descriptors,
        })

        obs = round_obj.step(selected_index)
        new_trick = obs is not None and bool(getattr(obs, "is_leading", False))
        finished_after = bool(round_obj.state.hands[player].sum() == 0)
        played_descriptor = describe_play(current_play, args.level, play_type_of,
                                          level_rank_index, order_cards_in_play,
                                          CARD_INT_TO_STR)
        history.append({
            "player_abs": player,
            "action": played_descriptor,
            "new_trick_after": new_trick,
            "finished_after": finished_after,
        })
        for other_agent in agents.values():
            other_agent.observe_action(player, current_play.copy(), new_trick)

    if not samples:
        raise RuntimeError("No DanLM alignment samples were produced")

    sample_set = {
        "schema_version": 2,
        "source": {
            "checkpoint": str(checkpoint),
            "seed": args.seed,
            "level": args.level,
            "first_player": args.first_player,
            "samples_requested": args.samples,
            "samples_exported": len(samples),
        },
        "contract": {
            "vocab_size": 91,
            "max_seq_len": 1280,
            "hand_size": 54,
            "action_size": 80,
            "output": "q_values",
        },
        "samples": samples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(sample_set, ensure_ascii=False, indent=2), "utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
