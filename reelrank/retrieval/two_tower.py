"""Two-tower retrieval model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTower(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_videos: int,
        n_user_cats: int,
        n_video_cats: int,
        user_num_dim: int,
        emb_dim: int = 64,
        hidden: int = 128,
        history_n: int = 20,
        temperature: float = 0.05,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.history_n = history_n
        self.user_id_emb = nn.Embedding(n_users + 1, emb_dim, padding_idx=0)
        self.video_id_emb = nn.Embedding(n_videos + 1, emb_dim, padding_idx=0)
        self.user_cat_emb = nn.Embedding(max(n_user_cats, 2), emb_dim // 2, padding_idx=0)
        self.video_cat_emb = nn.Embedding(max(n_video_cats, 2), emb_dim // 2, padding_idx=0)

        user_in = emb_dim + emb_dim // 2 + user_num_dim + emb_dim
        item_in = emb_dim + emb_dim // 2 + 2  # duration + popularity (log1p)

        self.user_mlp = nn.Sequential(
            nn.Linear(user_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb_dim),
        )
        self.item_mlp = nn.Sequential(
            nn.Linear(item_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb_dim),
        )

    def encode_user(
        self,
        user_idx: torch.Tensor,
        user_cat: torch.Tensor,
        user_num: torch.Tensor,
        hist_idx: torch.Tensor,
    ) -> torch.Tensor:
        uid = self.user_id_emb(user_idx)
        ucat = self.user_cat_emb(user_cat).mean(dim=1)
        hist = self.video_id_emb(hist_idx)
        mask = (hist_idx > 0).float().unsqueeze(-1)
        hist_pooled = (hist * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        x = torch.cat([uid, ucat, user_num, hist_pooled], dim=-1)
        return F.normalize(self.user_mlp(x), dim=-1)

    def encode_item(
        self,
        video_idx: torch.Tensor,
        video_cat: torch.Tensor,
        video_num: torch.Tensor,
    ) -> torch.Tensor:
        vid = self.video_id_emb(video_idx)
        vcat = self.video_cat_emb(video_cat).mean(dim=1)
        x = torch.cat([vid, vcat, video_num], dim=-1)
        return F.normalize(self.item_mlp(x), dim=-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        u = self.encode_user(batch["user_idx"], batch["user_cat"], batch["user_num"], batch["hist_idx"])
        v = self.encode_item(batch["video_idx"], batch["video_cat"], batch["video_num"])
        return u, v


def sampled_softmax_loss(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    logq: torch.Tensor | None = None,
    temperature: float = 0.05,
    candidate_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """In-batch softmax; mask repeated positive items used as negatives.

    logq contains the log sampling probability of each candidate column. It
    must match the positive-event distribution sampled by the data loader.
    Other distinct items remain negatives, even if another event liked them.
    """
    logits = (user_emb @ item_emb.T) / temperature
    if logq is not None:
        logits = logits - logq.unsqueeze(0)
    labels = torch.arange(user_emb.size(0), device=user_emb.device)
    if candidate_ids is not None:
        candidate_ids = candidate_ids.to(logits.device)
        if candidate_ids.ndim != 1 or len(candidate_ids) != len(labels):
            raise ValueError("candidate_ids must contain one ID per paired example")
        accidental = candidate_ids[:, None] == candidate_ids[None, :]
        accidental.fill_diagonal_(False)
        logits = logits.masked_fill(accidental, float("-inf"))
    return F.cross_entropy(logits, labels)
