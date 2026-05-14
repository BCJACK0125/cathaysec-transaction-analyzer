"""Health metrics for realized and unrealized PnL."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def calculate_health(realized: Iterable[dict], unrealized: Iterable[float]) -> dict:
    pnls = [float(item["pnl"]) for item in realized]

    total = sum(pnls) + sum(float(x) for x in unrealized)

    win = [p for p in pnls if p > 0]
    loss = [p for p in pnls if p < 0]

    win_rate = len(win) / len(pnls) if pnls else 0.0

    avg_win = float(np.mean(win)) if win else 0.0
    avg_loss = abs(float(np.mean(loss))) if loss else 1.0

    profit_factor = avg_win / avg_loss if avg_loss else 0.0

    score = min(100.0, max(0.0, 50.0 + profit_factor * 10.0))

    return {
        "total": total,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "score": score,
    }
