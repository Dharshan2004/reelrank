"""Caption blocklist / heuristic rules for content safety."""

from __future__ import annotations

import re

# Patterns covering spam/ads, solicitation, and explicit content cues (CN + EN).
BLOCK_PATTERNS = [
    r"加微信",
    r"加我微信",
    r"威信",
    r"vx\s*[:：]?",
    r"v信",
    r"微信\s*[:：]?\s*\w+",
    r"加qq",
    r"qq\s*[:：]\s*\d+",
    r"刷单",
    r"代刷",
    r"兼职日结",
    r"私信领取",
    r"免费领取",
    r"点击下方小黄车",
    r"小黄车",
    r"优惠券",
    r"返利",
    r"赌博",
    r"博彩",
    r"色情",
    r"约炮",
    r"裸聊",
    r"成人",
    r"黄片",
    r"crack\b",
    r"crypto\s*signal",
    r"whatsapp\s*\+?\d",
    r"telegram\s*@?\w+",
]

COMPILED = [re.compile(p, re.IGNORECASE) for p in BLOCK_PATTERNS]


def rule_hit(text: str) -> bool:
    if not text:
        return False
    return any(p.search(text) for p in COMPILED)


def rule_score(text: str) -> float:
    """Soft score in [0,1] from number of pattern hits."""
    if not text:
        return 0.0
    hits = sum(1 for p in COMPILED if p.search(text))
    if hits <= 0:
        return 0.0
    return float(min(1.0, 0.35 + 0.2 * hits))
