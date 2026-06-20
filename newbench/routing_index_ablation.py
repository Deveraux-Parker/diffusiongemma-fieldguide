#!/usr/bin/env python3
"""Ablate pointer indexes vs exact-value indexes for the attention valley.

Question: if the real target record is stuck in the dead zone, can a safe-zone
pointer make the model route back to it, or must the actual answer values be
duplicated in a safe zone?
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from kv_index_benchmark import extract_json_object, request_json


STRATEGIES = {
    "baseline_deadzone": {"top": None, "tail": None},
    "tail_pointer": {"top": None, "tail": "pointer"},
    "tail_exact_index": {"top": None, "tail": "exact_index"},
    "tail_summary_duplicate": {"top": None, "tail": "summary"},
    "tail_full_duplicate": {"top": None, "tail": "full"},
    "top_pointer": {"top": "pointer", "tail": None},
    "top_exact_index": {"top": "exact_index", "tail": None},
    "top_summary_duplicate": {"top": "summary", "tail": None},
    "top_full_duplicate": {"top": "full", "tail": None},
}

CHANNELS = ["temp", "flow", "pressure", "vibration", "voltage", "humidity"]
REGIMES = ["steady", "ramp_up", "cooldown", "drift", "spike", "recovery"]
RISKS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def iso_now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def stamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")


def stable_int(*parts: object, mod: int = 100000) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:12], 16) % mod


def deterministic_float(seed: int, low: float, high: float) -> float:
    value = stable_int("float", seed, mod=1_000_000) / 1_000_000.0
    return low + (high - low) * value


def fields_for(strategy: str, rep: int) -> dict[str, str]:
    salt = stable_int(strategy, rep, "fields", mod=90000)
    first = 800 + stable_int(strategy, rep, "first", mod=700)
    delta = stable_int(strategy, rep, "delta", mod=181) - 90
    if abs(delta) < 12:
        delta += 31
    return {
        "window_id": f"W{stable_int(strategy, rep, 'window', mod=10000):04d}",
        "incident_code": f"RI-{stable_int(strategy, rep, 'code', mod=100000):05d}",
        "checksum": f"RS-{stable_int(strategy, rep, 'checksum', mod=100000):05d}",
        "channel": CHANNELS[(salt + rep) % len(CHANNELS)],
        "regime": REGIMES[(salt * 3 + rep) % len(REGIMES)],
        "risk": RISKS[(salt * 5 + rep) % len(RISKS)],
        "first_value": str(first),
        "last_value": str(first + delta),
        "delta": str(delta),
        "trend": "rising" if delta > 0 else "falling",
    }


def nominal_line(i: int, salt: str) -> str:
    seed = stable_int(salt, i, mod=10_000_000)
    return (
        f"WINDOW W{stable_int(salt, i, 'win', mod=10000):04d}\n"
        "START_SUMMARY: "
        f"window_id=W{stable_int(salt, i, 'win', mod=10000):04d}; marker=routine; "
        f"channel={CHANNELS[stable_int(salt, i, 'ch', mod=len(CHANNELS))]}; "
        f"regime={REGIMES[stable_int(salt, i, 'reg', mod=len(REGIMES))]}; "
        "risk=LOW; incident_code=none; checksum=none; semantic_unit=front_loaded.\n"
        f"SAMPLES: m00={deterministic_float(seed, 800, 1500):.0f}, "
        f"m06={deterministic_float(seed + 1, 800, 1500):.0f}, "
        f"m12={deterministic_float(seed + 2, 800, 1500):.0f}.\n"
        "DETAILS: routine telemetry payload with no target answer.\n"
        "END_WINDOW"
    )


def summary_line(fields: dict[str, str]) -> str:
    return (
        "START_SUMMARY: "
        f"window_id={fields['window_id']}; marker=TARGET; "
        f"channel={fields['channel']}; regime={fields['regime']}; "
        f"risk={fields['risk']}; incident_code={fields['incident_code']}; "
        f"checksum={fields['checksum']}; first_value={fields['first_value']}; "
        f"last_value={fields['last_value']}; delta={fields['delta']}; "
        f"trend={fields['trend']}; semantic_unit=front_loaded."
    )


def full_record(fields: dict[str, str]) -> str:
    return (
        f"WINDOW {fields['window_id']}\n"
        f"{summary_line(fields)}\n"
        f"SAMPLES: m00={fields['first_value']}, m06={fields['last_value']}, "
        f"m12={fields['delta']}.\n"
        "DETAILS: TARGET anomaly payload; exact fields are in START_SUMMARY.\n"
        "END_WINDOW"
    )


def safe_item(kind: str | None, fields: dict[str, str], zone: str) -> str:
    if kind is None:
        return ""
    if kind == "pointer":
        return (
            f"{zone.upper()}_POINTER: target_window_id={fields['window_id']}; "
            "the target's incident_code and checksum must be read from the "
            "original TARGET START_SUMMARY record in the telemetry packet."
        )
    if kind == "exact_index":
        return (
            f"{zone.upper()}_TARGET_INDEX: window_id={fields['window_id']}; "
            f"incident_code={fields['incident_code']}; checksum={fields['checksum']}; "
            "copied_from=TARGET START_SUMMARY."
        )
    if kind == "summary":
        return f"{zone.upper()}_DUPLICATED_SUMMARY: {summary_line(fields)}"
    if kind == "full":
        return f"{zone.upper()}_DUPLICATED_RECORD:\n{full_record(fields)}"
    raise ValueError(f"unknown safe item kind: {kind}")


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
        "elapsed_sec": elapsed,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "usage": usage,
    }


def idx_for_distance(distance: int, nlines: int, tokens_per_record: float) -> int:
    return max(1, min(nlines - 1, nlines - round(distance / tokens_per_record)))


def build_prompt(
    base_url: str,
    model: str,
    strategy: str,
    fill_tokens: int,
    target_distance: int,
    rep: int,
) -> dict[str, Any]:
    config = STRATEGIES[strategy]
    fields = fields_for(strategy, rep)
    sample = "\n\n".join(nominal_line(i, "calibration") for i in range(1, 11))
    tokens_per_record = token_count(base_url, model, sample) / 10.0
    nrecords = int(fill_tokens / tokens_per_record)
    filler = [nominal_line(i, f"{strategy}-{rep}") for i in range(1, nrecords + 1)]
    target_idx = idx_for_distance(target_distance, len(filler), tokens_per_record)
    filler.insert(target_idx, full_record(fields))

    header_parts = [
        "BEGIN TELEMETRY PACKET. Exactly one record has marker=TARGET. "
        "Routine records have incident_code=none and checksum=none.",
    ]
    top = safe_item(config["top"], fields, "top")
    if top:
        header_parts.append(top)

    tail_parts = []
    tail = safe_item(config["tail"], fields, "tail")
    if tail:
        tail_parts.append(tail)
    tail_parts.append(
        "Question: Return JSON only with keys window_id, incident_code, checksum "
        "for marker=TARGET. Copy exact values."
    )

    user = "\n\n".join([*header_parts, *filler, *tail_parts])
    target_suffix = "\n\n".join(filler[target_idx + 1 :] + tail_parts)
    return {
        "user": user,
        "fields": fields,
        "target_idx": target_idx,
        "target_distance_actual": token_count(base_url, model, target_suffix),
        "prompt_tokens_by_tokenize": token_count(base_url, model, user),
        "top_kind": config["top"],
        "tail_kind": config["tail"],
    }


def score_output(content: str, fields: dict[str, str]) -> dict[str, Any]:
    parsed, parse_error = extract_json_object(content)
    parsed_obj = parsed if isinstance(parsed, dict) else {}
    expected = {
        "window_id": fields["window_id"],
        "incident_code": fields["incident_code"],
        "checksum": fields["checksum"],
    }
    exact_by_field = {
        key: str(parsed_obj.get(key)) == value or value in content
        for key, value in expected.items()
    }
    text = content.upper()
    if not content.strip():
        miss_type = "empty"
    elif expected["incident_code"].upper() in text:
        miss_type = "partial_with_code"
    elif expected["window_id"].upper() in text:
        miss_type = "window_only"
    elif "NONE" in text:
        miss_type = "routine_none"
    elif any(char.isdigit() for char in text):
        miss_type = "wrong_number"
    else:
        miss_type = "other"
    return {
        "field_score": sum(exact_by_field.values()) / len(exact_by_field),
        "all_exact": all(exact_by_field.values()),
        "json_parse_ok": parsed is not None,
        "parse_error": parse_error,
        "exact_by_field": exact_by_field,
        "miss_type": "hit" if all(exact_by_field.values()) else miss_type,
        "expected": expected,
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for strategy in STRATEGIES:
        group = [row for row in rows if row["strategy"] == strategy]
        scores = [row["scoring"]["field_score"] for row in group]
        exacts = [1.0 if row["scoring"]["all_exact"] else 0.0 for row in group]
        miss_types = {
            miss_type: sum(
                1
                for row in group
                if not row["scoring"]["all_exact"]
                and row["scoring"]["miss_type"] == miss_type
            )
            for miss_type in sorted({row["scoring"]["miss_type"] for row in group})
        }
        out.append(
            {
                "strategy": strategy,
                "top_kind": STRATEGIES[strategy]["top"],
                "tail_kind": STRATEGIES[strategy]["tail"],
                "safe_zone_contains_values": STRATEGIES[strategy]["top"]
                in {"exact_index", "summary", "full"}
                or STRATEGIES[strategy]["tail"] in {"exact_index", "summary", "full"},
                "pointer_only": STRATEGIES[strategy]["top"] == "pointer"
                or STRATEGIES[strategy]["tail"] == "pointer",
                "exact_rate": statistics.mean(exacts),
                "field_score_mean": statistics.mean(scores),
                "field_score_min": min(scores),
                "hits": int(sum(exacts)),
                "n": len(group),
                "target_distance_mean": statistics.mean(
                    row["target_distance_actual"] for row in group
                ),
                "prompt_tokens_mean": statistics.mean(row["prompt_tokens"] for row in group),
                "elapsed_sec_mean": statistics.mean(row["elapsed_sec"] for row in group),
                "miss_types": json.dumps(miss_types, sort_keys=True),
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
        "# Routing Index Ablation",
        "",
        f"- Started: {metadata['started_at']}",
        f"- Fill target: `{metadata['args']['fill_tokens']}` prompt tokens",
        f"- Target distance requested: `{metadata['args']['target_distance']}` tokens",
        f"- Reps: `{metadata['args']['reps']}`",
        "",
        "## Results",
        "",
        "| strategy | top | tail | values in safe zone | pointer only | hits | exact rate | field score | miss types |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in aggregate_rows:
        lines.append(
            "| {strategy} | {top_kind} | {tail_kind} | {safe_zone_contains_values} | {pointer_only} | {hits}/{n} | {exact_rate:.3f} | {field_score_mean:.3f} | `{miss_types}` |".format(
                **row
            )
        )
    lines += [
        "",
        "## Interpretation",
        "",
        (
            "Pointer-only rows test whether a safe-zone pointer can route the model "
            "back into the dead-zone source. Exact indexes and duplicated summaries "
            "test whether putting the actual answer values in a safe zone is enough."
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
    parser.add_argument("--target-distance", type=int, default=1000)
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=140)
    return parser.parse_args()


def iso_now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def stamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")


def main() -> int:
    args = parse_args()
    args.base_url = args.base_url.rstrip("/")
    output_dir = Path(args.out_dir) / f"routing-index-ablation-{stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "started_at": iso_now(),
        "base_url": args.base_url,
        "model": args.model,
        "args": {
            "fill_tokens": args.fill_tokens,
            "target_distance": args.target_distance,
            "reps": args.reps,
            "max_tokens": args.max_tokens,
            "strategies": STRATEGIES,
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    rows = []
    total = len(STRATEGIES) * args.reps
    with (output_dir / "raw.jsonl").open("w", encoding="utf-8") as raw:
        index = 0
        for strategy in STRATEGIES:
            for rep in range(1, args.reps + 1):
                index += 1
                prompt = build_prompt(
                    args.base_url,
                    args.model,
                    strategy,
                    args.fill_tokens,
                    args.target_distance,
                    rep,
                )
                result = chat(args.base_url, args.model, prompt["user"], args.max_tokens)
                scoring = score_output(result["content"], prompt["fields"])
                row = {
                    "strategy": strategy,
                    "rep": rep,
                    "top_kind": prompt["top_kind"],
                    "tail_kind": prompt["tail_kind"],
                    "target_distance_actual": prompt["target_distance_actual"],
                    "target_idx": prompt["target_idx"],
                    "prompt_tokens": result["prompt_tokens"],
                    "prompt_tokens_by_tokenize": prompt["prompt_tokens_by_tokenize"],
                    "completion_tokens": result["completion_tokens"],
                    "elapsed_sec": round(result["elapsed_sec"], 4),
                    "output": result["content"],
                    "scoring": scoring,
                }
                rows.append(row)
                raw.write(json.dumps(row, ensure_ascii=True) + "\n")
                raw.flush()
                print(
                    f"[{index:02d}/{total}] {strategy:24s} rep={rep} "
                    f"hit={int(scoring['all_exact'])} field={scoring['field_score']:.2f} "
                    f"miss={scoring['miss_type']} out={result['content'][:80]!r}"
                )

    trial_rows = []
    for row in rows:
        trial_rows.append(
            {
                "strategy": row["strategy"],
                "rep": row["rep"],
                "top_kind": row["top_kind"],
                "tail_kind": row["tail_kind"],
                "target_distance_actual": row["target_distance_actual"],
                "prompt_tokens": row["prompt_tokens"],
                "completion_tokens": row["completion_tokens"],
                "elapsed_sec": row["elapsed_sec"],
                "field_score": row["scoring"]["field_score"],
                "all_exact": row["scoring"]["all_exact"],
                "json_parse_ok": row["scoring"]["json_parse_ok"],
                "miss_type": row["scoring"]["miss_type"],
                "output": row["output"],
            }
        )
    write_csv(output_dir / "trials.csv", trial_rows)
    aggregate_rows = aggregate(rows)
    write_csv(output_dir / "aggregate.csv", aggregate_rows)
    write_summary(output_dir, metadata, aggregate_rows)
    print(f"\nWrote routing index ablation to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
