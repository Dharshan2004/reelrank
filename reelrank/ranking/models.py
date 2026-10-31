"""Ranking models: logistic regression features helper, DeepFM, MMoE."""

from __future__ import annotations

import torch
import torch.nn as nn


class DeepFM(nn.Module):
    def __init__(self, n_fields: int, emb_dim: int = 16, hidden: tuple[int, ...] = (128, 64)):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(n + 1, emb_dim, padding_idx=0) for n in n_fields]
        )
        self.fm_linear = nn.ModuleList([nn.Embedding(n + 1, 1, padding_idx=0) for n in n_fields])
        deep_in = emb_dim * len(n_fields)
        layers: list[nn.Module] = []
        prev = deep_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.deep = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F] long
        emb = [self.embeddings[i](x[:, i]) for i in range(x.size(1))]
        emb_stack = torch.stack(emb, dim=1)  # [B, F, D]
        # FM second order
        sum_square = emb_stack.sum(dim=1).pow(2)
        square_sum = emb_stack.pow(2).sum(dim=1)
        fm = 0.5 * (sum_square - square_sum).sum(dim=1, keepdim=True)
        linear = sum(self.fm_linear[i](x[:, i]) for i in range(x.size(1)))
        deep_in = emb_stack.reshape(x.size(0), -1)
        deep = self.deep(deep_in)
        return (linear + fm + deep).squeeze(-1)


class MMoE(nn.Module):
    def __init__(
        self,
        n_fields: list[int],
        n_tasks: int = 3,
        n_experts: int = 4,
        emb_dim: int = 16,
        expert_hidden: int = 128,
        tower_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.n_tasks = n_tasks
        self.embeddings = nn.ModuleList(
            [nn.Embedding(n + 1, emb_dim, padding_idx=0) for n in n_fields]
        )
        in_dim = emb_dim * len(n_fields)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(in_dim, expert_hidden), nn.ReLU(), nn.Linear(expert_hidden, expert_hidden), nn.ReLU())
                for _ in range(n_experts)
            ]
        )
        self.gates = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_dim, n_experts), nn.Softmax(dim=-1)) for _ in range(n_tasks)]
        )
        self.towers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(expert_hidden, tower_hidden), nn.ReLU(), nn.Linear(tower_hidden, 1))
                for _ in range(n_tasks)
            ]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        emb = [self.embeddings[i](x[:, i]) for i in range(x.size(1))]
        return torch.cat(emb, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.encode(x)
        expert_out = torch.stack([e(shared) for e in self.experts], dim=1)  # [B, E, H]
        outs = []
        for t in range(self.n_tasks):
            gate = self.gates[t](shared).unsqueeze(-1)  # [B, E, 1]
            mixed = (expert_out * gate).sum(dim=1)
            outs.append(self.towers[t](mixed).squeeze(-1))
        return torch.stack(outs, dim=-1)  # [B, T]
