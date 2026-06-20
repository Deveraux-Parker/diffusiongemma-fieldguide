#!/usr/bin/env python3
"""Mitigation strategy benchmark for the attention valley.

This tests whether duplicating a target inside the dead zone helps, and compares
that to moving/duplicating one copy into a known safer zone.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from kv_index_benchmark import request_json


DEFAULT_STRATEGIES = {
    "baseline_deadzone": [1000],
    "dup_adjacent_x2": [1000, 1000],
    "dup_adjacent_x3": [1000, 1000, 1000],
    "dup_spread_deadzone": [850, 1050, 1250],
    "dup_recency": [1000, 150],
    "dup_primacy": [1000, 5300],
    "dup_primacy_recency": [1000, 5300, 150],
}

COLORS = ["MAGENTA", "CYAN", "AMBER", "INDIGO", "CRIMSON", "JADE", "SLATE", "OCHRE"]


def iso_now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def stamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")


def stable_int(*parts: object, mod: int = 100000) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:12], 16) % mod


def code_for(strategy: str, rep: int) -> str:
    return f"{COLORS[stable_int(strategy, rep, 'color', mod=len(COLORS))]}-{1000 + stable_int(strategy, rep, 'code', mod=9000)}"


def deterministic_float(seed: int, low: float, high: float) -> float:
    # Low-cost deterministic pseudo-random float without depending on random hash state.
    value = stable_int("float", seed, mod=1_000_000) / 1_000_000.0
    return low + (high - low) * value


def nominal_line(i: int, salt: str) -> str:
    seed = stable_int(salt, i, mod=10_000_000)
    return (
        f"[t={i:05d}] sensor_temp={deterministic_float(seed, 18, 26):.2f}C "
        f"pressure={deterministic_float(seed + 1, 99, 103):.1f}kPa "
        f"flow={deterministic_float(seed + 2, 8, 16):.1f}L/min "
        f"vibration={deterministic_float(seed + 3, 0.1, 0.9):.2f}mm "
        "status=nominal"
    )


def needle_line(i: int, code: str) -> str:
    return (
        f"[t={i:05d}] ANOMALY sensor_temp=87.3C pressure=142.7kPa "
        f"flow=0.0L/min vibration=4.88mm status=CRITICAL incident_code={code}"
    )


def post_json(base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = request_json("POST", f"{base_url}{path}", payload, timeout=600)
    if not response.get("ok"):
        raise RuntimeError(response.get("error") or response)
    return response.get("data") or {}


def token_count(base_url: str, model: str, text: str) -> int:
    return int(post_json(base_url, "/tokenize", {"model": model, "prompt": text})["count"])


def chat(base_url: str, model: str, user: str, max_tokens: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    start = time.perf_counter()
    data = post_json(base_url, "/v1/chat/completions", payload)
    elapsed = time.perf_counter() - start
    choice = (data.get("choices") or [{}])[0]
    usage = data.get("usage") or {}
    return {
        "content": ((choice.get("message") or {}).get("content") or ""),
        "finish_reason": choice.get("finish_reason"),
        "usage": usage,
        "elapsed_sec": elapsed,
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
    }


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    m, n = len(a), len(b)
    d = list(range(n + 1))
    for i in range(1, m + 1):
        prev = d[0]
        d[0] = i
        for j in range(1, n + 1):
            old = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = old
    return d[n]


def found_code(output: str, code: str) -> bool:
    out = (output or "").upper()
    expected = code.upper()
    if expected in out:
        return True
    tokens = (
        out.replace("=", " ")
        .replace(",", " ")
        .replace(".", " ")
        .replace(":", " ")
        .split()
    )
    return any(levenshtein(token, expected) <= 1 for token in tokens)


def classify_miss(output: str, code: str) -> str:
    stripped = (output or "").strip()
    if not stripped:
        return "empty"
    out = stripped.upper()
    if "87" in out or "142" in out or "4.88" in out:
        return "needle_neighbor_value"
    if code.split("-", 1)[0] in out or code.split("-", 1)[1][:2] in out:
        return "code_fragment"
    if any(char.isdigit() for char in out):
        return "wrong_number"
    return "other"


def idx_for_distance(distance: int, nlines: int, tokens_per_line: float) -> int:
    return max(1, min(nlines - 1, nlines - round(distance / tokens_per_line)))


def build_prompt(
    base_url: str,
    model: str,
    strategy: str,
    requested_distances: list[int],
    code: str,
    fill_tokens: int,
    rep: int,
) -> dict[str, Any]:
    sample = "\n".join(nominal_line(i, "calibration") for i in range(1, 21))
    tokens_per_line = token_count(base_url, model, sample) / 20.0
    nlines = int(fill_tokens / tokens_per_line)
    filler = [nominal_line(i, f"{strategy}-{rep}") for i in range(1, nlines + 1)]
    placements = []
    occupied = set()
    for distance in requested_distances:
        idx = idx_for_distance(distance, len(filler), tokens_per_line)
        while idx in occupied:
            idx += 1
        occupied.add(idx)
        placements.append(idx)

    for copy_idx, idx in enumerate(sorted(placements, reverse=True), start=1):
        filler.insert(idx, needle_line(900000 + idx + copy_idx, code))

    question = (
        "\n\nOne or more lines have status=CRITICAL with an incident_code. "
        "In one short sentence, state that incident_code."
    )
    user = "Telemetry log:\n" + "\n".join(filler) + question
    actual_distances = []
    for idx in sorted(placements):
        suffix = "\n".join(filler[idx + 1 :]) + question
        actual_distances.append(token_count(base_url, model, suffix))
    return {
        "user": user,
        "tokens_per_line": tokens_per_line,
        "nlines": nlines,
        "placements": sorted(placements),
        "actual_distances": actual_distances,
        "prompt_tokens_by_tokenize": token_count(base_url, model, user),
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_strategy: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_strategy.setdefault(row["strategy"], []).append(row)
    out = []
    for strategy, group in by_strategy.items():
        hits = [1.0 if row["hit"] else 0.0 for row in group]
        out.append(
            {
                "strategy": strategy,
                "copies": statistics.mean(len(row["actual_distances"]) for row in group),
                "hit_rate": statistics.mean(hits),
                "hits": int(sum(hits)),
                "n": len(group),
                "prompt_tokens_mean": statistics.mean(row["prompt_tokens"] for row in group),
                "elapsed_sec_mean": statistics.mean(row["elapsed_sec"] for row in group),
                "actual_distances_sample": " / ".join(
                    str(round(value)) for value in group[0]["actual_distances"]
                ),
                "miss_types": json.dumps(
                    {
                        miss_type: sum(
                            1
                            for row in group
                            if not row["hit"] and row["miss_type"] == miss_type
                        )
                        for miss_type in sorted({row["miss_type"] for row in group})
                    },
                    sort_keys=True,
                ),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(output_dir: Path, metadata: dict[str, Any], aggregate_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Mitigation Strategy Benchmark",
        "",
        f"- Started: {metadata['started_at']}",
        f"- Model: `{metadata['model']}`",
        f"- Fill target: `{metadata['args']['fill_tokens']}` prompt tokens",
        f"- Reps: `{metadata['args']['reps']}`",
        "",
        "## Results",
        "",
        "| strategy | copies | hits | hit rate | actual copy distances | miss types |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in aggregate_rows:
        lines.append(
            "| {strategy} | {copies:.1f} | {hits}/{n} | {hit_rate:.3f} | {actual_distances_sample} | `{miss_types}` |".format(
                **row
            )
        )
    lines += [
        "",
        "## Interpretation",
        "",
        (
            "Back-to-back duplication inside the dead zone tests salience. "
            "A recency or primacy copy tests relocation into a safer attention region. "
            "If adjacent copies fail while relocated copies pass, duplication works "
            "only when at least one copy leaves the valley."
        ),
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="dg-awq")
    parser.add_argument("--out-dir", default="NEWBENCH")
    parser.add_argument("--fill-tokens", type=int, default=5500)
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.base_url = args.base_url.rstrip("/")
    output_dir = Path(args.out_dir) / f"mitigation-strategies-{stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "started_at": iso_now(),
        "base_url": args.base_url,
        "model": args.model,
        "args": {
            "fill_tokens": args.fill_tokens,
            "reps": args.reps,
            "max_tokens": args.max_tokens,
            "strategies": DEFAULT_STRATEGIES,
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    rows = []
    total = len(DEFAULT_STRATEGIES) * args.reps
    with (output_dir / "raw.jsonl").open("w", encoding="utf-8") as raw:
        index = 0
        for strategy, distances in DEFAULT_STRATEGIES.items():
            for rep in range(1, args.reps + 1):
                index += 1
                code = code_for(strategy, rep)
                prompt = build_prompt(
                    args.base_url,
                    args.model,
                    strategy,
                    distances,
                    code,
                    args.fill_tokens,
                    rep,
                )
                result = chat(args.base_url, args.model, prompt["user"], args.max_tokens)
                hit = found_code(result["content"], code)
                row = {
                    "strategy": strategy,
                    "rep": rep,
                    "requested_distances": distances,
                    "actual_distances": prompt["actual_distances"],
                    "placements": prompt["placements"],
                    "code": code,
                    "output": result["content"],
                    "hit": hit,
                    "miss_type": "hit" if hit else classify_miss(result["content"], code),
                    "prompt_tokens": result["prompt_tokens"],
                    "prompt_tokens_by_tokenize": prompt["prompt_tokens_by_tokenize"],
                    "completion_tokens": result["completion_tokens"],
                    "elapsed_sec": round(result["elapsed_sec"], 4),
                }
                rows.append(row)
                raw.write(json.dumps(row, ensure_ascii=True) + "\n")
                raw.flush()
                print(
                    f"[{index:02d}/{total}] {strategy:22s} rep={rep} "
                    f"hit={int(hit)} dists={prompt['actual_distances']} "
                    f"out={result['content'][:70]!r}"
                )

    trial_rows = [
        {
            **{k: v for k, v in row.items() if k not in {"requested_distances", "actual_distances", "placements"}},
            "requested_distances": " / ".join(str(v) for v in row["requested_distances"]),
            "actual_distances": " / ".join(str(v) for v in row["actual_distances"]),
            "placements": " / ".join(str(v) for v in row["placements"]),
        }
        for row in rows
    ]
    write_csv(output_dir / "trials.csv", trial_rows)
    agg = aggregate(rows)
    write_csv(output_dir / "aggregate.csv", agg)
    write_summary(output_dir, metadata, agg)
    print(f"\nWrote mitigation strategy benchmark to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
