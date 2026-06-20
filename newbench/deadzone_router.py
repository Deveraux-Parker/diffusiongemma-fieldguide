#!/usr/bin/env python3
"""Utilities for routing exact facts around a vLLM chunked-prefill trap.

The empirical rule from the local mechanism decomposition:

    A fact is high risk when it is in the 1024-token block immediately before
    the question block, and the fact's block index is even.

Correction: this rule describes the old/effective 2048-token prefill compile
range behavior. Reruns with `--max-num-batched-tokens 8192` and compile range
`[8192]` made the bad cells pass. Keep this utility for regression testing or
for old server configs, not as evidence of an intrinsic model limitation.

This module does not tokenize by itself. Callers pass measured token positions:
`n` = tokens before the fact, `q` = total prompt tokens through the question.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterable


BLOCK_SIZE = 1024
CORE_MIN_DISTANCE = 650
CORE_MAX_DISTANCE = 2200


@dataclass(frozen=True)
class PositionReport:
    name: str
    n: int
    q: int
    distance: int
    needle_block: int
    question_block: int
    block_diff: int
    needle_block_parity: int
    in_core_band: bool
    dead_zone: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def classify_position(
    name: str,
    n: int,
    q: int,
    block_size: int = BLOCK_SIZE,
    core_min_distance: int = CORE_MIN_DISTANCE,
    core_max_distance: int = CORE_MAX_DISTANCE,
) -> PositionReport:
    """Classify one fact position relative to the final question."""
    distance = q - n
    needle_block = n // block_size
    question_block = q // block_size
    block_diff = question_block - needle_block
    parity = needle_block % 2
    in_core_band = core_min_distance <= distance <= core_max_distance
    dead_zone = block_diff == 1 and parity == 0 and in_core_band
    if dead_zone:
        reason = "diff==1, even needle block, core distance band"
    elif block_diff == 0:
        reason = "same block as question / recency-safe"
    elif block_diff >= 2:
        reason = "two or more blocks before question / generally safe"
    elif parity == 1:
        reason = "diff==1 but odd needle block"
    elif not in_core_band:
        reason = "outside core distance band"
    else:
        reason = "not classified as trap"
    return PositionReport(
        name=name,
        n=n,
        q=q,
        distance=distance,
        needle_block=needle_block,
        question_block=question_block,
        block_diff=block_diff,
        needle_block_parity=parity,
        in_core_band=in_core_band,
        dead_zone=dead_zone,
        reason=reason,
    )


def choose_padding(
    candidates: Iterable[int],
    evaluator: Callable[[int], list[PositionReport]],
) -> dict[str, object]:
    """Choose the padding value with the fewest dead-zone facts.

    `evaluator(pad)` should return reports for the prompt assembled with that
    many approximate tokens of start padding. Ties prefer lower padding, then
    fewer diff==1 facts, then greater distance to the nearest bad condition.
    """
    best: dict[str, object] | None = None
    for pad in candidates:
        reports = evaluator(pad)
        dead_count = sum(1 for report in reports if report.dead_zone)
        diff1_count = sum(1 for report in reports if report.block_diff == 1)
        min_distance_to_core_edge = min(
            (
                min(
                    abs(report.distance - CORE_MIN_DISTANCE),
                    abs(report.distance - CORE_MAX_DISTANCE),
                )
                for report in reports
            ),
            default=0,
        )
        score = (dead_count, diff1_count, -min_distance_to_core_edge, pad)
        row = {
            "pad": pad,
            "score": score,
            "dead_count": dead_count,
            "diff1_count": diff1_count,
            "reports": reports,
        }
        if best is None or score < best["score"]:
            best = row
    if best is None:
        raise ValueError("no padding candidates supplied")
    return best


def reports_to_dicts(reports: Iterable[PositionReport]) -> list[dict[str, object]]:
    return [report.to_dict() for report in reports]
