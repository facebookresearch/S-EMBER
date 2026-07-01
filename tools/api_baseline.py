#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Unified closed-source API baselines for S-EMBER (Gemini / GPT-4o / o3).

Evaluation modes against closed-source backends:

    Modes:
      * mcq        - 5-way multiple choice; rule-based letter accuracy.
      * grounding  - free-text answer + temporal interval; reports temporal IoU.
                     (Run tools/judge_grounding.py for answer correctness.)
      * pure_llm   - text-only (no video); paired with --mode {mcq|grounding}.

    Backends (--model):
      * gemini     - Google Gemini File API; uploads the trimmed mp4 to Gemini
                     and asks one multimodal question.
      * gpt4o      - Azure OpenAI GPT-4o; uniformly samples N frames as base64.
      * o3         - Azure OpenAI o3 via the Responses API; same frame encoding.

Examples:

    # MCQ with Gemini, upload 720p video
    GEMINI_API_KEY=... python tools/api_baseline.py mcq \
        --model gemini --resolution 720

    # MCQ with GPT-4o, 50 uniformly-sampled frames
    AZURE_OPENAI_ENDPOINT=https://<your-host> AZURE_OPENAI_API_KEY=... \
        python tools/api_baseline.py mcq --model gpt4o --max-frames 50

    # Pure-LLM floor (no video)
    AZURE_OPENAI_ENDPOINT=https://<your-host> AZURE_OPENAI_API_KEY=... \
        python tools/api_baseline.py pure_llm --model gpt4o --mode mcq

Environment:
    GEMINI_API_KEY            for --model gemini
    GEMINI_MODEL              override Gemini model id (default gemini-3.1-flash)
    AZURE_OPENAI_API_KEY      for --model gpt4o / o3
    AZURE_OPENAI_ENDPOINT     Azure resource endpoint
    AZURE_OPENAI_API_VERSION  default 2024-12-01-preview
    AZURE_OPENAI_GPT4O_MODEL  default gpt-4o
    AZURE_OPENAI_O3_MODEL     default o3
    SEMBER_VIDEO_DIR          directory of <video_id>.mp4 files
    API_CONCURRENCY           max in-flight API requests (default 5)
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHOICE_LABELS = ["A", "B", "C", "D", "E"]

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

DEFAULT_MAX_FRAMES = 50
MAX_TOTAL_IMAGE_SIZE_MB = 45
PER_SAMPLE_TIMEOUT = 600

PURE_LLM_SYSTEM_PROMPT = (
    "You are watching a first-person egocentric video recorded by someone "
    "wearing a camera during their daily activities. The video captures "
    "everyday tasks such as cooking, cleaning, shopping, working, or moving "
    "around. You have just finished watching the video up to the specified "
    "time. Even though you cannot see the actual video, use your best "
    "judgment, common sense, and world knowledge to provide a reasonable "
    "answer to the question. Answer as if you have seen the video. Do not "
    "say you cannot see the video or refuse to answer; always provide your "
    "best guess."
)

# ---------------------------------------------------------------------------
# ffmpeg trim + downscale (with on-disk caching)
# ---------------------------------------------------------------------------


def _find_ffmpeg() -> str:
    p = shutil.which("ffmpeg")
    if p:
        return p
    conda_bin = os.path.join(os.path.dirname(sys.executable), "ffmpeg")
    if os.path.isfile(conda_bin) and os.access(conda_bin, os.X_OK):
        return conda_bin
    raise RuntimeError("ffmpeg is required but not found on PATH or in the active env")


def trim_video_ffmpeg(
    video_path: str,
    video_end: float,
    temp_dir: str,
    resolution: Optional[int] = None,
) -> str:
    """Trim a video to ``[0, video_end]`` (sec) and optionally downscale.

    Cached by hash(path, end, resolution).
    """
    key = f"{video_path}_{video_end}_{resolution}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    out_path = os.path.join(temp_dir, f"trimmed_{h}.mp4")
    if os.path.exists(out_path):
        return out_path
    ffmpeg = _find_ffmpeg()
    cmd = [ffmpeg, "-y", "-i", video_path, "-t", str(video_end)]
    if resolution is not None:
        cmd += [
            "-vf",
            f"scale=-2:{resolution}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "28",
        ]
    else:
        cmd += ["-c", "copy"]
    cmd += ["-an", "-loglevel", "error", out_path]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


# ---------------------------------------------------------------------------
# Frame extraction (used by GPT-4o / o3 backends)
# ---------------------------------------------------------------------------


def extract_frames_as_base64(
    video_path: str,
    max_frames: int,
    max_size_per_image_mb: int = 20,
) -> list[str]:
    import base64

    import numpy as np
    from decord import VideoReader, cpu
    from PIL import Image

    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    if total == 0:
        return []
    n = min(max_frames, total)
    indices = np.linspace(0, total - 1, n, dtype=int).tolist()
    if total - 1 not in indices:
        indices.append(total - 1)
    frames = vr.get_batch(indices).asnumpy()

    cap_per = max_size_per_image_mb * 1024 * 1024
    cap_total = MAX_TOTAL_IMAGE_SIZE_MB * 1024 * 1024
    out: list[str] = []
    used = 0
    for frame in frames:
        img = Image.fromarray(frame)
        while True:
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=85)
            data = buf.getvalue()
            if len(data) <= cap_per or img.size[0] <= 100 or img.size[1] <= 100:
                break
            img = img.resize((int(img.size[0] * 0.75), int(img.size[1] * 0.75)))
        if used + len(data) > cap_total:
            break
        out.append(base64.b64encode(data).decode("ascii"))
        used += len(data)
    return out


# ---------------------------------------------------------------------------
# Prompt builders + parsers
# ---------------------------------------------------------------------------


def build_mcq_prompt(question: str, options: list[str]) -> str:
    labeled = []
    for i, opt in enumerate(options):
        labeled.append(opt if re.match(r"^[A-E]\.\s", opt) else f"{CHOICE_LABELS[i]}. {opt}")
    options_text = "\n".join(f"  {opt}" for opt in labeled)
    label_str = "/".join(CHOICE_LABELS[: len(options)])
    return (
        "After reviewing the video, answer the following multiple-choice question.\n"
        "\n"
        f"Question: {question}\n"
        "\n"
        f"{options_text}\n"
        "\n"
        f"IMPORTANT: Respond with ONLY a single letter ({label_str}). "
        "Do NOT include any explanation, reasoning, or additional text. "
        "Just the letter.\n"
        "\n"
        "Answer:"
    )


def build_grounding_prompt(question: str) -> str:
    return (
        "After reviewing the video, provide the best answer to the following question. "
        "Answer in 1-2 sentences.\n"
        "Also provide the time interval (in seconds) where the answer evidence "
        "appears in the video.\n\n"
        "Use this exact format:\n"
        "Answer: <your answer>\n"
        "Time: [<start_seconds>, <end_seconds>]\n\n"
        f"{question}"
    )


def build_pure_llm_user_prompt(
    mode: str,
    question: str,
    question_time: float,
    duration: float,
    question_category: str,
    options: Optional[list[str]] = None,
) -> str:
    head = f"Video duration so far: {int(question_time)} seconds " f"(total video length: {int(duration)} seconds).\n" f"Question category: {question_category}\n\n" f"Question: {question}\n\n"
    if mode == "mcq":
        labeled = []
        for i, opt in enumerate(options or []):
            labeled.append(opt if re.match(r"^[A-E]\.\s", opt) else f"{CHOICE_LABELS[i]}. {opt}")
        opts_text = "\n".join(f"  {opt}" for opt in labeled)
        labels = "/".join(CHOICE_LABELS[: len(options or [])])
        return head + opts_text + f"\n\nIMPORTANT: Respond with ONLY a single letter ({labels}). " "Just the letter."
    return head + "Provide the best answer to this question based on what someone would " "typically experience in a first-person daily activity video. " "Answer in 1-2 sentences."


def parse_mcq_choice(text: str, num_options: int = 5) -> Optional[str]:
    if not text:
        return None
    t = text.strip().upper()
    valid = CHOICE_LABELS[:num_options]
    if t in valid:
        return t
    m = re.search(r"\b([A-E])\b", t)
    return m.group(1) if m and m.group(1) in valid else None


def parse_grounding_response(pred: str):
    answer_text = pred.strip()
    pred_start = pred_end = None
    am = re.search(r"Answer:\s*(.+?)(?:\n|Time:)", pred, re.DOTALL)
    if am:
        answer_text = am.group(1).strip()
    tm = re.search(r"Time:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", pred)
    if tm:
        try:
            pred_start = float(tm.group(1))
            pred_end = float(tm.group(2))
        except ValueError:
            pass
    return answer_text, pred_start, pred_end


def temporal_iou(ps, pe, gs, ge) -> float:
    if any(v is None for v in (ps, pe, gs, ge)):
        return 0.0
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class Backend:
    name: str = "?"

    def __init__(self, model: str, semaphore: asyncio.Semaphore):
        self.model = model
        self.sem = semaphore

    async def call_with_video(self, trimmed_video_path: str, prompt: str, max_frames: int) -> str:
        raise NotImplementedError

    async def call_text_only(self, prompt: str) -> str:
        raise NotImplementedError


class GeminiBackend(Backend):
    """Google Gemini File API: upload the trimmed mp4, ask one question."""

    name = "gemini"

    def __init__(self, model: str, semaphore: asyncio.Semaphore):
        super().__init__(model, semaphore)
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except ImportError:
            sys.exit("ERROR: install google-genai  ->  pip install google-genai")
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            sys.exit("ERROR: GEMINI_API_KEY env var is not set.")
        self._types = types
        self.client = genai.Client(api_key=api_key)

    def _upload_video_blocking(self, path: str):
        import time as _time

        f = self.client.files.upload(file=path)
        while f.state == "PROCESSING":
            _time.sleep(3)
            f = self.client.files.get(name=f.name)
        if f.state == "FAILED":
            raise RuntimeError(f"Gemini File API failed for {path}")
        return f

    async def call_with_video(self, trimmed_video_path: str, prompt: str, max_frames: int) -> str:
        f = await asyncio.to_thread(self._upload_video_blocking, trimmed_video_path)
        try:
            async with self.sem:
                resp = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=[f, prompt],
                    config=self._types.GenerateContentConfig(
                        temperature=0,
                        max_output_tokens=2048 if "Time:" in prompt else 512,
                    ),
                )
            return resp.text or ""
        finally:
            try:
                self.client.files.delete(name=f.name)
            except Exception:
                pass

    async def call_text_only(self, prompt: str) -> str:
        async with self.sem:
            resp = await self.client.aio.models.generate_content(
                model=self.model,
                contents=[prompt],
                config=self._types.GenerateContentConfig(temperature=0, max_output_tokens=512),
            )
        return resp.text or ""


class _AzureBaseBackend(Backend):
    def __init__(self, model: str, semaphore: asyncio.Semaphore):
        super().__init__(model, semaphore)
        try:
            from openai import AzureOpenAI  # type: ignore
        except ImportError:
            sys.exit("ERROR: install openai  ->  pip install openai")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        if not endpoint or not api_key:
            sys.exit("ERROR: AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY env vars are required.")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        self.client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)


class AzureGPT4oBackend(_AzureBaseBackend):
    name = "gpt4o"

    async def call_with_video(self, trimmed_video_path: str, prompt: str, max_frames: int) -> str:
        frames = await asyncio.to_thread(extract_frames_as_base64, trimmed_video_path, max_frames)
        content: list[dict] = [{"type": "text", "text": prompt}]
        for b64 in frames:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "auto"},
                }
            )
        async with self.sem:
            resp = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[{"role": "user", "content": content}],
                temperature=0,
                max_tokens=2048 if "Time:" in prompt else 512,
            )
        return resp.choices[0].message.content or ""

    async def call_text_only(self, prompt: str) -> str:
        async with self.sem:
            resp = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=512,
            )
        return resp.choices[0].message.content or ""


class AzureO3Backend(_AzureBaseBackend):
    name = "o3"

    async def call_with_video(self, trimmed_video_path: str, prompt: str, max_frames: int) -> str:
        frames = await asyncio.to_thread(extract_frames_as_base64, trimmed_video_path, max_frames)
        content: list[dict] = [{"type": "input_text", "text": prompt}]
        for b64 in frames:
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{b64}",
                }
            )
        async with self.sem:
            resp = await asyncio.to_thread(
                self.client.responses.create,
                model=self.model,
                input=[{"role": "user", "content": content}],
                max_output_tokens=2048 if "Time:" in prompt else 512,
            )
        return resp.output_text or ""

    async def call_text_only(self, prompt: str) -> str:
        async with self.sem:
            resp = await asyncio.to_thread(
                self.client.responses.create,
                model=self.model,
                input=[
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": prompt}],
                    }
                ],
                max_output_tokens=512,
            )
        return resp.output_text or ""


def make_backend(name: str, model: Optional[str], sem: asyncio.Semaphore) -> Backend:
    if name == "gemini":
        return GeminiBackend(model or os.environ.get("GEMINI_MODEL", "gemini-3.1-flash"), sem)
    if name == "gpt4o":
        return AzureGPT4oBackend(model or os.environ.get("AZURE_OPENAI_GPT4O_MODEL", "gpt-4o"), sem)
    if name == "o3":
        return AzureO3Backend(model or os.environ.get("AZURE_OPENAI_O3_MODEL", "o3"), sem)
    raise SystemExit(f"Unknown --model {name!r}; expected one of: gemini, gpt4o, o3")


# ---------------------------------------------------------------------------
# Per-sample processors
# ---------------------------------------------------------------------------


async def _process_video_sample(
    backend: Backend,
    sample: dict,
    video_dir: str,
    temp_dir: str,
    mode: str,
    resolution: Optional[int],
    max_frames: int,
) -> dict:
    video_id = sample["video_id"]
    question = sample["question"]
    question_time = sample.get("question_time")
    options = sample.get("options", [])

    result = {
        "question_id": sample.get("question_id"),
        "question_category": sample.get("question_category"),
        "question": question,
        "video_id": video_id,
        "question_time": question_time,
        "options": options,
        "correct_letter": sample.get("correct_letter"),
        "correct_index": sample.get("correct_index"),
        "answer": sample.get("answer"),
        "answers": sample.get("answers", []),
        "answer_start_time": sample.get("answer_start_time"),
        "answer_end_time": sample.get("answer_end_time"),
        "duration": sample.get("duration"),
        "pred_raw": None,
        "pred_letter": None,
        "pred_answer": None,
        "pred_start_time": None,
        "pred_end_time": None,
        "is_correct": None,
        "iou": None,
        "status": None,
    }

    video_path = os.path.join(video_dir, f"{video_id}.mp4")
    if not os.path.exists(video_path):
        result["status"] = f"FAILURE: video not found: {video_path}"
        return result

    try:
        trimmed = await asyncio.to_thread(
            trim_video_ffmpeg,
            video_path,
            float(question_time) if question_time is not None else 0.0,
            temp_dir,
            resolution,
        )
        prompt = build_mcq_prompt(question, options) if mode == "mcq" else build_grounding_prompt(question)
        raw = await asyncio.wait_for(
            backend.call_with_video(trimmed, prompt, max_frames),
            timeout=PER_SAMPLE_TIMEOUT,
        )
        result["pred_raw"] = raw

        if mode == "mcq":
            letter = parse_mcq_choice(raw, len(options))
            result["pred_letter"] = letter
            result["is_correct"] = (letter == result["correct_letter"]) if letter else False
        else:
            ans, ps, pe = parse_grounding_response(raw)
            result["pred_answer"] = ans
            result["pred_start_time"] = ps
            result["pred_end_time"] = pe
            result["iou"] = temporal_iou(
                ps,
                pe,
                sample.get("answer_start_time"),
                sample.get("answer_end_time"),
            )
        result["status"] = "SUCCESS"
    except Exception as exc:  # noqa: BLE001
        result["status"] = f"FAILURE: {type(exc).__name__}: {exc}"
    return result


async def _process_pure_llm_sample(backend: Backend, sample: dict, mode: str) -> dict:
    question = sample["question"]
    question_time = sample.get("question_time", 0) or 0
    duration = sample.get("duration", question_time) or question_time
    options = sample.get("options", [])
    result = {
        "question_id": sample.get("question_id"),
        "question_category": sample.get("question_category"),
        "question": question,
        "video_id": sample.get("video_id"),
        "question_time": question_time,
        "options": options,
        "correct_letter": sample.get("correct_letter"),
        "correct_index": sample.get("correct_index"),
        "answer": sample.get("answer"),
        "duration": duration,
        "pred_raw": None,
        "pred_letter": None,
        "pred_answer": None,
        "is_correct": None,
        "status": None,
    }
    user_prompt = build_pure_llm_user_prompt(
        mode,
        question,
        float(question_time),
        float(duration),
        sample.get("question_category", "unknown"),
        options,
    )
    full_prompt = PURE_LLM_SYSTEM_PROMPT + "\n\n" + user_prompt
    try:
        raw = await asyncio.wait_for(backend.call_text_only(full_prompt), timeout=PER_SAMPLE_TIMEOUT)
        result["pred_raw"] = raw
        if mode == "mcq":
            letter = parse_mcq_choice(raw, len(options))
            result["pred_letter"] = letter
            result["is_correct"] = (letter == result["correct_letter"]) if letter else False
        else:
            result["pred_answer"] = raw.strip()
        result["status"] = "SUCCESS"
    except Exception as exc:  # noqa: BLE001
        result["status"] = f"FAILURE: {type(exc).__name__}: {exc}"
    return result


# ---------------------------------------------------------------------------
# I/O + summary
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _print_mcq_summary(rows: list[dict]) -> None:
    by_cat: dict[str, dict[str, int]] = collections.defaultdict(lambda: {"correct": 0, "total": 0, "parsed": 0, "failed": 0})
    for r in rows:
        cat = r.get("question_category", "unknown")
        if r.get("status") and not r["status"].startswith("SUCCESS"):
            by_cat[cat]["failed"] += 1
        by_cat[cat]["total"] += 1
        if r.get("pred_letter") is not None:
            by_cat[cat]["parsed"] += 1
        if r.get("is_correct"):
            by_cat[cat]["correct"] += 1
    print("\n=== MCQ accuracy ===")
    for cat in QUESTION_CATEGORIES + sorted(set(by_cat) - set(QUESTION_CATEGORIES)):
        s = by_cat.get(cat)
        if not s:
            continue
        acc = 100.0 * s["correct"] / s["total"] if s["total"] else 0.0
        print(f"  {cat:32s} {acc:6.2f}%  ({s['correct']}/{s['total']}, " f"parsed={s['parsed']}, failed={s['failed']})")
    total = sum(s["total"] for s in by_cat.values())
    correct = sum(s["correct"] for s in by_cat.values())
    overall = 100.0 * correct / total if total else 0.0
    print(f"  Overall:                         {overall:6.2f}%  ({correct}/{total})")


def _print_grounding_summary(rows: list[dict]) -> None:
    by_cat: dict[str, dict[str, float]] = collections.defaultdict(lambda: {"sum_iou": 0.0, "count": 0, "parseable": 0, "failed": 0})
    for r in rows:
        cat = r.get("question_category", "unknown")
        if r.get("status") and not r["status"].startswith("SUCCESS"):
            by_cat[cat]["failed"] += 1
        by_cat[cat]["count"] += 1
        if r.get("pred_start_time") is not None and r.get("pred_end_time") is not None:
            by_cat[cat]["parseable"] += 1
            by_cat[cat]["sum_iou"] += float(r.get("iou") or 0.0)
    print("\n=== Temporal IoU ===")
    for cat in QUESTION_CATEGORIES + sorted(set(by_cat) - set(QUESTION_CATEGORIES)):
        s = by_cat.get(cat)
        if not s:
            continue
        miou = 100.0 * s["sum_iou"] / s["count"] if s["count"] else 0.0
        prate = 100.0 * s["parseable"] / s["count"] if s["count"] else 0.0
        print(f"  {cat:32s} mIoU={miou:6.2f}%  " f"(parseable={int(s['parseable'])}/{int(s['count'])}={prate:.0f}%, " f"failed={int(s['failed'])})")
    total = sum(s["count"] for s in by_cat.values())
    parse = sum(s["parseable"] for s in by_cat.values())
    siou = sum(s["sum_iou"] for s in by_cat.values())
    overall = 100.0 * siou / total if total else 0.0
    prate = 100.0 * parse / total if total else 0.0
    print(f"  Overall:                         mIoU={overall:6.2f}%  " f"(parseable={parse}/{total}={prate:.0f}%)")


# ---------------------------------------------------------------------------
# Orchestrators
# ---------------------------------------------------------------------------


async def _run_video_eval(args, backend: Backend) -> None:
    samples = _read_jsonl(args.jsonl)
    if args.max_samples:
        samples = samples[: args.max_samples]
    print(
        f"[{args.cmd}] {len(samples)} samples; backend={backend.name} " f"model={backend.model}; video_dir={args.video_dir}",
        file=sys.stderr,
    )
    with tempfile.TemporaryDirectory(prefix="sember_api_") as temp_dir:
        tasks = [
            _process_video_sample(
                backend,
                s,
                args.video_dir,
                temp_dir,
                args.mode,
                args.resolution,
                args.max_frames,
            )
            for s in samples
        ]
        results: list[dict] = []
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            results.append(await coro)
            if i % 25 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} done", file=sys.stderr)
    _write_jsonl(args.output, results)
    print(f"\nWrote per-sample results to {args.output}", file=sys.stderr)
    if args.mode == "mcq":
        _print_mcq_summary(results)
    else:
        _print_grounding_summary(results)


async def _run_pure_llm_eval(args, backend: Backend) -> None:
    samples = _read_jsonl(args.jsonl)
    if args.max_samples:
        samples = samples[: args.max_samples]
    print(
        f"[pure_llm] {len(samples)} samples; backend={backend.name} " f"model={backend.model}; mode={args.mode}",
        file=sys.stderr,
    )
    tasks = [_process_pure_llm_sample(backend, s, args.mode) for s in samples]
    results: list[dict] = []
    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        results.append(await coro)
        if i % 25 == 0 or i == len(tasks):
            print(f"  {i}/{len(tasks)} done", file=sys.stderr)
    _write_jsonl(args.output, results)
    print(f"\nWrote per-sample results to {args.output}", file=sys.stderr)
    if args.mode == "mcq":
        _print_mcq_summary(results)
    else:
        ok = sum(1 for r in results if r.get("status") == "SUCCESS")
        print(f"\n=== pure_llm grounding ===  produced answers for {ok}/{len(results)} samples")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--model", choices=["gemini", "gpt4o", "o3"], required=True)
        sp.add_argument(
            "--model-id",
            default=None,
            help="Override model id (default: env-var or backend default).",
        )
        sp.add_argument(
            "--video-dir",
            default=os.environ.get("SEMBER_VIDEO_DIR", "data/videos"),
        )
        sp.add_argument("--max-samples", type=int, default=None)
        sp.add_argument(
            "--concurrency",
            type=int,
            default=int(os.environ.get("API_CONCURRENCY", "5")),
        )

    p_mcq = sub.add_parser("mcq", help="MCQ video QA over data/sember_mcq.jsonl")
    p_mcq.set_defaults(mode="mcq")
    add_common(p_mcq)
    p_mcq.add_argument("--jsonl", type=Path, default=Path("data/sember_mcq.jsonl"))
    p_mcq.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Per-sample results jsonl (default: output/api/<model>_mcq.jsonl).",
    )
    p_mcq.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Downscale to this height (px) before sending to the API.",
    )
    p_mcq.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help="Frames per video for gpt4o/o3 backends (gemini ignores this).",
    )

    p_gnd = sub.add_parser("grounding", help="Grounding video QA over data/sember_grounding.jsonl")
    p_gnd.set_defaults(mode="grounding")
    add_common(p_gnd)
    p_gnd.add_argument("--jsonl", type=Path, default=Path("data/sember_grounding.jsonl"))
    p_gnd.add_argument("--output", type=Path, default=None)
    p_gnd.add_argument("--resolution", type=int, default=None)
    p_gnd.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)

    p_pure = sub.add_parser("pure_llm", help="No-vision text-only baseline (LLM floor)")
    add_common(p_pure)
    p_pure.add_argument("--mode", choices=["mcq", "grounding"], required=True)
    p_pure.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Defaults to data/sember_mcq.jsonl for --mode=mcq, " "data/sember_grounding.jsonl for --mode=grounding.",
    )
    p_pure.add_argument("--output", type=Path, default=None)

    args = p.parse_args()

    if args.cmd == "pure_llm" and args.jsonl is None:
        args.jsonl = Path("data/sember_mcq.jsonl") if args.mode == "mcq" else Path("data/sember_grounding.jsonl")
    if args.output is None:
        outdir = Path("output/api")
        suffix = f"{args.cmd}_{args.mode}" if args.cmd == "pure_llm" else args.cmd
        args.output = outdir / f"{args.model}_{suffix}.jsonl"
    return args


def main() -> None:
    args = _parse_args()
    if not args.jsonl.exists():
        sys.exit(f"ERROR: input jsonl not found: {args.jsonl}")
    if args.cmd != "pure_llm" and not Path(args.video_dir).is_dir():
        sys.exit(f"ERROR: --video-dir not found: {args.video_dir}\n" "       Set SEMBER_VIDEO_DIR or pass --video-dir <path>.")
    sem = asyncio.Semaphore(args.concurrency)
    backend = make_backend(args.model, args.model_id, sem)
    if args.cmd == "pure_llm":
        asyncio.run(_run_pure_llm_eval(args, backend))
    else:
        asyncio.run(_run_video_eval(args, backend))


if __name__ == "__main__":
    main()
