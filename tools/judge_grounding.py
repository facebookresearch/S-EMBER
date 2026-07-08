#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Optional Gemini judge for the S-EMBER grounding task.

The grounding runner (`scripts/grounding/run_*.sh`) reports temporal IoU
deterministically, but does not score the free-text answer field. This
script reads the per-sample jsonl produced by lmms-eval and uses Google
Gemini to judge each prediction as CORRECT / WRONG against the
multi-tier gold answers, then prints accuracy overall and per category.

Usage:

    export GEMINI_API_KEY=<your-key>
    python tools/judge_grounding.py \\
        path/to/<run>_grounding/<model>/<timestamp>_samples_sember_grounding.jsonl \\
        --out judge_results.json

Outputs:

  * <input>.judged.jsonl  -- per-sample judgement records
  * <out>                 -- aggregate accuracy by category and overall

Environment:

  GEMINI_API_KEY      Google Gemini API key (required).
  GEMINI_MODEL        Model id (default: gemini-3.1-flash).
  JUDGE_CONCURRENCY   Max concurrent API requests (default: 8).
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

QUESTION_CATEGORIES = [
    "location_trace",
    "sequential_action",
    "counting_objects_events",
    "visual_detail_recall",
    "temporal_ordering_recognition",
    "time_duration",
    "object_comparison",
    "spatial_aware_reasoning",
]

JUDGE_PROMPT = (
    "You are an impartial judge for an episodic-memory video QA benchmark. "
    "Given a question and a set of acceptable gold answers, judge whether "
    "the model prediction is semantically correct.\n\n"
    "A prediction is CORRECT if it is semantically equivalent to ANY ONE of "
    "the gold answers, even if phrased differently. Otherwise it is WRONG.\n\n"
    "Question:\n{question}\n\n"
    "Acceptable gold answers (any one is sufficient):\n{golds}\n\n"
    "Model prediction:\n{pred}\n\n"
    "Reply with a single word on the first line, exactly CORRECT or WRONG. "
    "Optionally add one short sentence of explanation on the next line."
)


def _golds_for(record: dict) -> list[str]:
    """Pull the multi-tier gold answers out of a sember_grounding record."""
    golds: list[str] = []
    primary = record.get("answer")
    if isinstance(primary, str) and primary.strip():
        golds.append(primary.strip())
    for entry in record.get("answers", []) or []:
        text = entry.get("answer_text") if isinstance(entry, dict) else None
        if isinstance(text, str) and text.strip():
            t = text.strip()
            if t not in golds:
                golds.append(t)
    return golds


def _format_golds(golds: list[str]) -> str:
    return "\n".join(f"  - {g}" for g in golds) if golds else "  (none provided)"


def _read_input(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # lmms-eval samples jsonl puts the doc fields directly at the top
            # level and the per-metric process_results output under a key
            # named after the metric. We accept either shape.
            if "sember_temporal_iou" in row and isinstance(row["sember_temporal_iou"], dict):
                row = {**row, **row["sember_temporal_iou"]}
            rows.append(row)
    return rows


async def _judge_one(client, model: str, sem: asyncio.Semaphore, row: dict) -> dict:
    question = row.get("question") or ""
    pred = row.get("pred_answer") or row.get("pred_raw") or ""
    golds = _golds_for(row)
    prompt = JUDGE_PROMPT.format(question=question, golds=_format_golds(golds), pred=pred)

    async with sem:
        # google-genai async API
        try:
            resp = await client.aio.models.generate_content(model=model, contents=prompt)
            text = (resp.text or "").strip()
        except Exception as exc:  # noqa: BLE001
            return {"verdict": "ERROR", "raw": "", "error": str(exc)}

    first_line = text.split("\n", 1)[0].strip().upper()
    verdict = "CORRECT" if "CORRECT" in first_line and "WRONG" not in first_line else "WRONG" if "WRONG" in first_line else "UNPARSED"
    return {"verdict": verdict, "raw": text}


async def _run(rows: list[dict], model: str, concurrency: int) -> list[dict]:
    try:
        from google import genai  # type: ignore
    except ImportError:
        sys.exit("ERROR: 'google-genai' package not installed.\n" "       pip install google-genai")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("ERROR: GEMINI_API_KEY env var is not set.")
    client = genai.Client(api_key=api_key)
    sem = asyncio.Semaphore(concurrency)

    out: list[dict] = []
    tasks = [_judge_one(client, model, sem, row) for row in rows]
    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        result = await coro
        out.append(result)
        if i % 50 == 0 or i == len(tasks):
            print(f"  judged {i}/{len(tasks)}", file=sys.stderr)
    return out


def _aggregate(rows: list[dict], judgements: list[dict]) -> dict[str, Any]:
    by_cat: dict[str, dict[str, int]] = collections.defaultdict(lambda: {"correct": 0, "total": 0, "unparsed": 0, "errors": 0})
    for row, j in zip(rows, judgements):
        cat = row.get("question_category", "unknown")
        by_cat[cat]["total"] += 1
        v = j["verdict"]
        if v == "CORRECT":
            by_cat[cat]["correct"] += 1
        elif v == "UNPARSED":
            by_cat[cat]["unparsed"] += 1
        elif v == "ERROR":
            by_cat[cat]["errors"] += 1

    per_cat: dict[str, Any] = {}
    for cat in QUESTION_CATEGORIES + sorted(set(by_cat) - set(QUESTION_CATEGORIES)):
        s = by_cat.get(cat)
        if not s:
            continue
        acc = 100.0 * s["correct"] / s["total"] if s["total"] else 0.0
        per_cat[cat] = {**s, "accuracy_pct": round(acc, 2)}

    total = sum(s["total"] for s in by_cat.values())
    correct = sum(s["correct"] for s in by_cat.values())
    overall = 100.0 * correct / total if total else 0.0
    return {
        "overall": {
            "correct": correct,
            "total": total,
            "accuracy_pct": round(overall, 2),
        },
        "per_category": per_cat,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Optional Gemini judge for S-EMBER grounding answers.")
    p.add_argument("input", type=Path, help="Per-sample jsonl produced by sember_grounding (lmms-eval).")
    p.add_argument("--out", "-o", type=Path, default=None, help="Aggregate results json (default: <input>.judged.json).")
    p.add_argument("--judged-jsonl", type=Path, default=None, help="Per-sample judgement jsonl (default: <input>.judged.jsonl).")
    p.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-3.1-flash"))
    p.add_argument("--concurrency", type=int, default=int(os.environ.get("JUDGE_CONCURRENCY", "8")))
    args = p.parse_args()

    out_json = args.out or args.input.with_suffix(args.input.suffix + ".judged.json")
    out_jsonl = args.judged_jsonl or args.input.with_suffix(args.input.suffix + ".judged.jsonl")

    rows = _read_input(args.input)
    print(f"Loaded {len(rows)} samples from {args.input}", file=sys.stderr)
    print(f"Judging with model={args.model} concurrency={args.concurrency}", file=sys.stderr)

    judgements = asyncio.run(_run(rows, args.model, args.concurrency))

    with open(out_jsonl, "w") as fh:
        for row, j in zip(rows, judgements):
            fh.write(
                json.dumps(
                    {
                        "question_id": row.get("question_id"),
                        "question_category": row.get("question_category"),
                        "question": row.get("question"),
                        "pred_answer": row.get("pred_answer") or row.get("pred_raw"),
                        "verdict": j["verdict"],
                        "judge_raw": j.get("raw", ""),
                        "error": j.get("error"),
                    }
                )
                + "\n"
            )
    print(f"Wrote per-sample judgements to {out_jsonl}", file=sys.stderr)

    summary = _aggregate(rows, judgements)
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote aggregate summary to {out_json}", file=sys.stderr)

    print()
    print("=== S-EMBER grounding answer accuracy ===")
    for cat, s in summary["per_category"].items():
        print(f"  {cat:32s} {s['accuracy_pct']:6.2f}%  ({s['correct']}/{s['total']})")
    print()
    o = summary["overall"]
    print(f"  Overall:                        {o['accuracy_pct']:6.2f}%  ({o['correct']}/{o['total']})")


if __name__ == "__main__":
    main()
