#!/usr/bin/env python3
"""Portable DanLM V1 network used for Android ONNX export.

The upstream resource cache contains compiled Python extension modules for the
runtime, but the checkpoint itself is enough to reconstruct the inference
network. This module mirrors the checkpoint parameter names so the state dict
loads strictly and can be exported without importing upstream platform modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DanLmV1Config:
    state_token_dim: int
    n_attn_blocks: int
    n_query_heads: int
    n_kv_heads: int
    qk_dim: int
    v_dim: int
    ffn_hidden: int
    max_seq_len: int
    hand_emb_dim: int
    hand_hidden_dim: int
    n_hand_hiddens: int
    q_head_hidden_dim: int
    n_q_head_hiddens: int
    vocab_size: int = 91
    hand_size: int = 54
    action_size: int = 80

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, Any]) -> "DanLmV1Config":
        raw = dict(checkpoint["config"])
        return cls(
            state_token_dim=int(raw["state_token_dim"]),
            n_attn_blocks=int(raw["n_attn_blocks"]),
            n_query_heads=int(raw["n_query_heads"]),
            n_kv_heads=int(raw["n_kv_heads"]),
            qk_dim=int(raw["qk_dim"]),
            v_dim=int(raw["v_dim"]),
            ffn_hidden=int(raw["ffn_hidden"]),
            max_seq_len=int(raw["max_seq_len"]),
            hand_emb_dim=int(raw["hand_emb_dim"]),
            hand_hidden_dim=int(raw["hand_hidden_dim"]),
            n_hand_hiddens=int(raw["n_hand_hiddens"]),
            q_head_hidden_dim=int(raw["q_head_hidden_dim"]),
            n_q_head_hiddens=int(raw["n_q_head_hiddens"]),
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed * self.weight


def apply_rope(x: Tensor, rope_cos: Tensor, rope_sin: Tensor) -> Tensor:
    # x: (B, H, T, D), rope tables: (max_seq_len, D // 2)
    seq_len = x.shape[-2]
    cos = rope_cos[:seq_len].unsqueeze(0).unsqueeze(0).to(dtype=x.dtype)
    sin = rope_sin[:seq_len].unsqueeze(0).unsqueeze(0).to(dtype=x.dtype)
    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


class GQAttention(nn.Module):
    def __init__(self, config: DanLmV1Config) -> None:
        super().__init__()
        self.n_query_heads = config.n_query_heads
        self.n_kv_heads = config.n_kv_heads
        self.qk_dim = config.qk_dim
        self.v_dim = config.v_dim
        self.q_proj = nn.Linear(
            config.state_token_dim,
            config.n_query_heads * config.qk_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.state_token_dim,
            config.n_kv_heads * config.qk_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.state_token_dim,
            config.n_kv_heads * config.v_dim,
            bias=False,
        )
        self.out_proj = nn.Linear(
            config.n_query_heads * config.v_dim,
            config.state_token_dim,
            bias=False,
        )
        self.q_norm = RMSNorm(config.qk_dim)
        self.k_norm = RMSNorm(config.qk_dim)

    def forward(self, x: Tensor, rope_cos: Tensor, rope_sin: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_query_heads, self.qk_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.qk_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.v_dim)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        if self.n_query_heads != self.n_kv_heads:
            repeat = self.n_query_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.qk_dim ** -0.5)
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).triu(1)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        context = torch.matmul(probs, v)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch, seq_len, self.n_query_heads * self.v_dim)
        return self.out_proj(context)


class SwiGLUFFN(nn.Module):
    def __init__(self, config: DanLmV1Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.state_token_dim, config.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(config.state_token_dim, config.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(config.ffn_hidden, config.state_token_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: DanLmV1Config) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.state_token_dim)
        self.attn = GQAttention(config)
        self.ffn_norm = RMSNorm(config.state_token_dim)
        self.ffn = SwiGLUFFN(config)

    def forward(self, x: Tensor, rope_cos: Tensor, rope_sin: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x), rope_cos, rope_sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


def make_mlp(input_dim: int, hidden_dim: int, hidden_count: int, output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = input_dim
    for _ in range(hidden_count):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.SiLU())
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class TransformerQNetwork(nn.Module):
    def __init__(self, config: DanLmV1Config) -> None:
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(
            config.vocab_size,
            config.state_token_dim,
            padding_idx=0,
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_attn_blocks)
        )
        self.final_norm = RMSNorm(config.state_token_dim)
        self.hand_mlp = make_mlp(
            config.hand_size + config.action_size,
            config.hand_hidden_dim,
            config.n_hand_hiddens,
            config.hand_emb_dim,
        )
        self.q_head = make_mlp(
            config.state_token_dim + config.hand_emb_dim,
            config.q_head_hidden_dim,
            config.n_q_head_hiddens,
            1,
        )
        self.register_buffer(
            "rope_cos",
            torch.empty(config.max_seq_len, config.qk_dim // 2),
            persistent=True,
        )
        self.register_buffer(
            "rope_sin",
            torch.empty(config.max_seq_len, config.qk_dim // 2),
            persistent=True,
        )

    def _abs_to_rel_players(self, token_ids: Tensor, self_abs: Tensor) -> Tensor:
        player_mask = (token_ids >= 1) & (token_ids <= 4)
        absolute = token_ids - 1
        relative = torch.remainder(absolute - self_abs.unsqueeze(1), 4) + 1
        return torch.where(player_mask, relative, token_ids)

    def encode(self, token_ids: Tensor, seq_lens: Tensor, self_abs: Tensor) -> Tensor:
        token_ids = self._abs_to_rel_players(token_ids, self_abs)
        hidden = self.token_emb(token_ids)
        for block in self.blocks:
            hidden = block(hidden, self.rope_cos, self.rope_sin)
        hidden = self.final_norm(hidden)
        index = torch.clamp(seq_lens, min=1, max=token_ids.shape[1]) - 1
        gather_index = index.view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])
        return hidden.gather(1, gather_index).squeeze(1)

    def q_values_with_context(self, context: Tensor, hand: Tensor, action: Tensor) -> Tensor:
        action_count = action.shape[1]
        hand_expanded = hand.unsqueeze(1).expand(-1, action_count, -1)
        hand_action = torch.cat((hand_expanded, action), dim=-1)
        hand_action_emb = self.hand_mlp(hand_action)
        context_expanded = context.unsqueeze(1).expand(-1, action_count, -1)
        q_input = torch.cat((context_expanded, hand_action_emb), dim=-1)
        return self.q_head(q_input).squeeze(-1)

    def forward(
        self,
        token_ids: Tensor,
        seq_lens: Tensor,
        hand: Tensor,
        action: Tensor,
        self_abs: Tensor,
    ) -> Tensor:
        context = self.encode(token_ids, seq_lens, self_abs)
        return self.q_values_with_context(context, hand, action)


def load_from_checkpoint(checkpoint_path: str | bytes | Any) -> TransformerQNetwork:
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    config = DanLmV1Config.from_checkpoint(checkpoint)
    model = TransformerQNetwork(config)
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("model")
    if state_dict is None:
        raise KeyError("checkpoint does not contain a DanLM model state dict")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model
