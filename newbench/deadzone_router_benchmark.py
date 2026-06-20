#!/usr/bin/env python3
"""Live benchmark for the 1024-block dead-zone router.

This tests whether the computed router can mitigate the known retrieval trap:

1. raw_bad: a record near D~1000 in an even 1024-token block.
2. ballast_router: add start padding chosen by the classifier to move the fact
   out of the trap while keeping distance roughly constant.
3. tail_registry: duplicate the answer-bearing value in a safe registry before
   the question.
4. combo_router: ballast plus tail registry.

It also tests a multi-record prompt where only dead-zone records are copied into
a tail registry.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import statistics
import time
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from deadzone_router import (
    BLOCK_SIZE,
    choose_padding,
    classify_position,
    reports_to_dicts,
)


BASE_URL = "http://127.0.0.1:8001"
MODEL = "dg-awq"
HEADER = "OPERATIONS LOG\n\n"
QUESTION_SINGLE = "\n\nState the secret token from the RECORD line, in one short sentence."
QUESTION_MULTI = (
    "\n\nThere are four RECORD lines, each with a secret token. Return JSON only "
    'with this shape: {"secret_tokens":["TOKEN-1111","TOKEN-2222"]}.'
)
LOREM = (
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
    "incididunt ut labore et dolore magna aliqua enim ad minim veniam quis nostrud "
    "exercitation ullamco laboris nisi aliquip ex ea commodo"
).split()
COLORS = ["MAGENTA", "CYAN", "AMBER", "INDIGO", "CRIMSON", "JADE", "SLATE", "OCHRE"]


def stamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")


def iso_now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout: int = 300) -> dict[str, Any]:
    req = urllib.request.Request(
        base_url + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return {"elapsed_sec": time.perf_counter() - started, "data": json.loads(resp.read())}


def tok(base_url: str, model: str, text: str) -> int:
    return int(post_json(base_url, "/tokenize", {"model": model, "prompt": text})["data"]["count"])


def chat(base_url: str, model: str, prompt: str, max_tokens: int = 96) -> dict[str, Any]:
    result = post_json(
        base_url,
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=600,
    )
    data = result["data"]
    choice = data["choices"][0]
    usage = data.get("usage") or {}
    return {
        "output": choice.get("message", {}).get("content") or "",
        "elapsed_sec": result["elapsed_sec"],
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "finish_reason": choice.get("finish_reason"),
    }


def lev(a: str, b: str) -> int:
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


def exact_hit(output: str, code: str) -> bool:
    return code.upper() in (output or "").upper()


def lenient_hit(output: str, code: str) -> bool:
    upper = (output or "").upper()
    code_upper = code.upper()
    if code_upper in upper:
        return True
    for token in re.split(r"[\s,.:;\"'`{}\[\]()]+", upper):
        if lev(token, code_upper) <= 1:
            return True
    return False


def lorem(rng: random.Random, n: int = 12) -> str:
    return " ".join(rng.choice(LOREM) for _ in range(n)).capitalize() + "."


def record_line(name: str, code: str) -> str:
    return f"RECORD {name}: secret token = {code}."


def rand_code(rng: random.Random) -> str:
    return f"{rng.choice(COLORS)}-{rng.randint(1000, 9999)}"


def registry_for(records: list[dict[str, str]]) -> str:
    if not records:
        return ""
    rows = [f"- {item['name']} secret_token={item['code']}" for item in records]
    return (
        "\n\nSAFE_EXACT_REGISTRY:\n"
        + "\n".join(rows)
        + "\nUse SAFE_EXACT_REGISTRY as authoritative for exact secret tokens."
    )


class PromptFactory:
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url
        self.model = model
        self.tpl = tok(base_url, model, "\n".join(lorem(random.Random(i)) for i in range(20))) / 20.0

    def token_count(self, text: str) -> int:
        return tok(self.base_url, self.model, text)

    def build_single(
        self,
        seed: int,
        pad_tokens: int = 0,
        tail_registry: bool = False,
        pre_tokens: int = 4500,
        post_tokens: int = 1000,
    ) -> dict[str, Any]:
        rng = random.Random(seed)
        code = rand_code(rng)
        pad_lines = round(pad_tokens / self.tpl)
        pre_lines = round(pre_tokens / self.tpl)
        post_lines = round(post_tokens / self.tpl)
        body = (
            [lorem(rng) for _ in range(pad_lines)]
            + [lorem(rng) for _ in range(pre_lines)]
            + [record_line("TARGET", code)]
            + [lorem(rng) for _ in range(post_lines)]
        )
        record_index = pad_lines + pre_lines
        registry = registry_for([{"name": "TARGET", "code": code}]) if tail_registry else ""
        prompt = HEADER + "\n".join(body) + registry + QUESTION_SINGLE
        q = self.token_count(prompt)
        n_record = self.token_count(HEADER + "\n".join(body[:record_index]))
        reports = [classify_position("source_record", n_record, q)]
        if tail_registry:
            prefix = HEADER + "\n".join(body)
            n_registry = self.token_count(prefix)
            reports.append(classify_position("tail_registry", n_registry, q))
        return {
            "prompt": prompt,
            "code": code,
            "reports": reports,
            "pad_tokens": pad_tokens,
            "tail_registry": tail_registry,
        }

    def choose_single_pad(self, seed: int) -> int:
        candidates = range(0, 2305, 128)

        def evaluate(pad: int) -> list[Any]:
            return self.build_single(seed, pad_tokens=pad, tail_registry=False)["reports"]

        return int(choose_padding(candidates, evaluate)["pad"])

    def build_multi(
        self,
        seed: int,
        pad_tokens: int = 0,
        registry_mode: str = "none",
        registry_names: list[str] | None = None,
        q_target: int = 5600,
        distances: list[int] | None = None,
    ) -> dict[str, Any]:
        distances = distances or [320, 1020, 1780, 3000]
        rng = random.Random(seed)
        nlines = max(20, round((q_target - len(QUESTION_MULTI)) / self.tpl))
        pad_lines = round(pad_tokens / self.tpl)
        lines = [lorem(rng) for _ in range(pad_lines + nlines)]
        records: list[dict[str, str]] = []
        inserted: set[int] = set()
        for i, distance in enumerate(distances, start=1):
            code = rand_code(rng)
            name = f"R{i}"
            idx = pad_lines + max(1, min(nlines - 1, nlines - round(distance / self.tpl)))
            while idx in inserted and idx < len(lines) - 1:
                idx += 1
            inserted.add(idx)
            lines.insert(idx, record_line(name, code))
            records.append({"name": name, "code": code, "line_index": str(idx), "requested_distance": str(distance)})

        base_prompt = HEADER + "\n".join(lines) + QUESTION_MULTI
        q_base = self.token_count(base_prompt)
        source_reports = []
        for item in records:
            idx = int(item["line_index"])
            n_record = self.token_count(HEADER + "\n".join(lines[:idx]))
            source_reports.append(classify_position(item["name"], n_record, q_base))

        dead_records = [
            item
            for item, report in zip(records, source_reports)
            if report.dead_zone
        ]
        if registry_names is not None:
            registry_name_set = set(registry_names)
            registry_records = [item for item in records if item["name"] in registry_name_set]
        elif registry_mode == "dead_only":
            registry_records = dead_records
        elif registry_mode == "all":
            registry_records = records
        else:
            registry_records = []

        registry = registry_for(registry_records)
        prompt = HEADER + "\n".join(lines) + registry + QUESTION_MULTI
        q = self.token_count(prompt)
        reports = []
        for item in records:
            idx = int(item["line_index"])
            n_record = self.token_count(HEADER + "\n".join(lines[:idx]))
            reports.append(classify_position(item["name"], n_record, q))
        if registry_records:
            n_registry = self.token_count(HEADER + "\n".join(lines))
            reports.append(classify_position("tail_registry", n_registry, q))

        return {
            "prompt": prompt,
            "records": records,
            "dead_records_from_base": dead_records,
            "reports": reports,
            "pad_tokens": pad_tokens,
            "registry_mode": registry_mode,
        }

    def choose_multi_pad(self, seed: int) -> int:
        candidates = range(0, 2305, 128)

        def evaluate(pad: int) -> list[Any]:
            return self.build_multi(seed, pad_tokens=pad, registry_mode="none")["reports"]

        return int(choose_padding(candidates, evaluate)["pad"])


def write_row(raw: Any, rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    rows.append(row)
    raw.write(json.dumps(row) + "\n")
    raw.flush()
    print(
        f"{row['suite']:8s} {row['strategy']:22s} trial={row['trial']:02d} "
        f"exact={row['exact_score']:.3f} lenient={row['lenient_score']:.3f} "
        f"dead={row['dead_count']} pad={row['pad_tokens']} prompt={row['prompt_tokens']} "
        f"time={row['elapsed_sec']:.2f}s",
        flush=True,
    )


def run_single(args: argparse.Namespace, factory: PromptFactory, raw: Any, rows: list[dict[str, Any]]) -> None:
    for trial in range(args.reps):
        seed = 910000 + trial
        safe_pad = factory.choose_single_pad(seed)
        variants = {
            "raw_bad": {"pad": 0, "registry": False},
            "ballast_router": {"pad": safe_pad, "registry": False},
            "tail_registry": {"pad": 0, "registry": True},
            "combo_router": {"pad": safe_pad, "registry": True},
        }
        for strategy, opts in variants.items():
            built = factory.build_single(seed, pad_tokens=opts["pad"], tail_registry=opts["registry"])
            result = chat(args.base_url, args.model, built["prompt"], 96)
            exact = exact_hit(result["output"], built["code"])
            lenient = lenient_hit(result["output"], built["code"])
            reports = built["reports"]
            write_row(
                raw,
                rows,
                {
                    "suite": "single",
                    "strategy": strategy,
                    "trial": trial,
                    "seed": seed,
                    "expected": [built["code"]],
                    "output": result["output"],
                    "exact_score": float(exact),
                    "lenient_score": float(lenient),
                    "dead_count": sum(1 for report in reports if report.dead_zone),
                    "pad_tokens": built["pad_tokens"],
                    "chosen_pad": safe_pad,
                    "reports": reports_to_dicts(reports),
                    **result,
                },
            )


def run_multi(args: argparse.Namespace, factory: PromptFactory, raw: Any, rows: list[dict[str, Any]]) -> None:
    for trial in range(args.multi_reps):
        seed = 920000 + trial
        safe_pad = factory.choose_multi_pad(seed)
        base = factory.build_multi(seed, pad_tokens=0, registry_mode="none")
        original_dead_names = [
            report.name for report in base["reports"] if report.dead_zone
        ]
        variants = {
            "raw_mixed": {"pad": 0, "registry": "none", "registry_names": None},
            "ballast_router": {"pad": safe_pad, "registry": "none", "registry_names": None},
            "dead_only_registry": {"pad": 0, "registry": "dead_only", "registry_names": None},
            "all_registry": {"pad": 0, "registry": "all", "registry_names": None},
            "combo_original_dead_registry": {
                "pad": safe_pad,
                "registry": "none",
                "registry_names": original_dead_names,
            },
        }
        for strategy, opts in variants.items():
            built = factory.build_multi(
                seed,
                pad_tokens=opts["pad"],
                registry_mode=opts["registry"],
                registry_names=opts["registry_names"],
            )
            result = chat(args.base_url, args.model, built["prompt"], 180)
            codes = [item["code"] for item in built["records"]]
            exact_hits = [exact_hit(result["output"], code) for code in codes]
            lenient_hits = [lenient_hit(result["output"], code) for code in codes]
            reports = built["reports"]
            write_row(
                raw,
                rows,
                {
                    "suite": "multi",
                    "strategy": strategy,
                    "trial": trial,
                    "seed": seed,
                    "expected": codes,
                    "output": result["output"],
                    "exact_score": sum(exact_hits) / len(exact_hits),
                    "lenient_score": sum(lenient_hits) / len(lenient_hits),
                    "exact_hits": exact_hits,
                    "lenient_hits": lenient_hits,
                    "dead_count": sum(1 for report in reports if report.dead_zone and report.name != "tail_registry"),
                    "base_dead_records": built["dead_records_from_base"],
                    "pad_tokens": built["pad_tokens"],
                    "chosen_pad": safe_pad,
                    "reports": reports_to_dicts(reports),
                    **result,
                },
            )


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["suite"], row["strategy"])].append(row)
    out = []
    for (suite, strategy), group in sorted(grouped.items()):
        exact = [float(row["exact_score"]) for row in group]
        lenient = [float(row["lenient_score"]) for row in group]
        out.append(
            {
                "suite": suite,
                "strategy": strategy,
                "n": len(group),
                "mean_exact": statistics.mean(exact),
                "min_exact": min(exact),
                "mean_lenient": statistics.mean(lenient),
                "pass_rate_exact_1": sum(score == 1.0 for score in exact) / len(exact),
                "mean_dead_count": statistics.mean(int(row["dead_count"]) for row in group),
                "mean_pad_tokens": statistics.mean(int(row["pad_tokens"]) for row in group),
                "mean_prompt_tokens": statistics.mean(int(row["prompt_tokens"]) for row in group),
                "mean_elapsed_sec": statistics.mean(float(row["elapsed_sec"]) for row in group),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        vals = []
        for field in fields:
            val = row.get(field, "")
            if isinstance(val, float):
                val = f"{val:.3f}"
            vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_summary(outdir: Path, args: argparse.Namespace, factory: PromptFactory, rows: list[dict[str, Any]], agg: list[dict[str, Any]]) -> None:
    lines = [
        "# Dead-Zone Router Benchmark",
        "",
        f"- Started: {args.started}",
        f"- Model: `{args.model}`",
        f"- Block size: `{BLOCK_SIZE}`",
        f"- Calibrated filler tokens/line: `{factory.tpl:.2f}`",
        f"- Single reps: `{args.reps}`",
        f"- Multi reps: `{args.multi_reps}`",
        "",
        "## Results",
        "",
        md_table(
            agg,
            [
                "suite",
                "strategy",
                "n",
                "mean_exact",
                "min_exact",
                "mean_lenient",
                "pass_rate_exact_1",
                "mean_dead_count",
                "mean_pad_tokens",
                "mean_prompt_tokens",
            ],
        ),
        "",
        "## Interpretation",
        "",
        "- `raw_bad` deliberately places the target in `diff==1` with an even needle block.",
        "- `ballast_router` adds computed start padding until the classifier predicts no dead-zone source record.",
        "- `tail_registry` duplicates exact answer-bearing fields immediately before the question.",
        "- `dead_only_registry` copies only records classified as dead in the base prompt.",
        "- `combo_original_dead_registry` adds ballast and still copies records that were dead before padding.",
        "",
        "Use `raw.jsonl` for full prompts' position reports and model outputs.",
        "",
    ]
    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--reps", type=int, default=12)
    parser.add_argument("--multi-reps", type=int, default=8)
    parser.add_argument("--out-root", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.started = iso_now()
    factory = PromptFactory(args.base_url, args.model)
    outdir = args.out_root / f"deadzone-router-{stamp()}"
    outdir.mkdir(parents=True, exist_ok=False)
    print(f"Writing results to {outdir}", flush=True)
    print(f"Calibrated filler tokens/line ~= {factory.tpl:.2f}", flush=True)

    rows: list[dict[str, Any]] = []
    with (outdir / "raw.jsonl").open("w", encoding="utf-8") as raw:
        run_single(args, factory, raw, rows)
        run_multi(args, factory, raw, rows)

    agg = aggregate(rows)
    write_csv(
        outdir / "trials.csv",
        rows,
        [
            "suite",
            "strategy",
            "trial",
            "exact_score",
            "lenient_score",
            "dead_count",
            "pad_tokens",
            "chosen_pad",
            "prompt_tokens",
            "completion_tokens",
            "elapsed_sec",
            "finish_reason",
        ],
    )
    write_csv(
        outdir / "aggregate.csv",
        agg,
        [
            "suite",
            "strategy",
            "n",
            "mean_exact",
            "min_exact",
            "mean_lenient",
            "pass_rate_exact_1",
            "mean_dead_count",
            "mean_pad_tokens",
            "mean_prompt_tokens",
            "mean_elapsed_sec",
        ],
    )
    (outdir / "metadata.json").write_text(
        json.dumps(
            {
                "started": args.started,
                "base_url": args.base_url,
                "model": args.model,
                "reps": args.reps,
                "multi_reps": args.multi_reps,
                "block_size": BLOCK_SIZE,
                "tokens_per_line": factory.tpl,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_summary(outdir, args, factory, rows, agg)
    print(f"Saved summary to {outdir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
