#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Filter sember_mcq.jsonl to only the questions whose video is on disk.

Smoke-test flow:
    1. Download ``sember_mcq.jsonl`` and 1+ mp4s from the Hugging Face
       dataset into ``data/`` and ``data/videos/``.
    2. ``scripts/run_mcq_smoke.sh`` calls this script before launching the model.

Behavior:
    * On first run, snapshots the user-provided JSONL as
      ``data/sember_mcq.full.jsonl`` so the original is preserved.
    * Reads from ``--master`` (default: ``data/sember_mcq.full.jsonl``) and
      writes the filtered subset to ``--out`` (default:
      ``data/sember_mcq.jsonl``, which is the path the YAML loads).
    * Drops any question whose ``video_id`` does not have a corresponding
      ``<video_id>.mp4`` under ``--video-dir``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--video-dir", default="data/videos", help="Directory holding <video_id>.mp4 files.")
    p.add_argument("--master", default="data/sember_mcq.full.jsonl", help="Master (un-filtered) JSONL.")
    p.add_argument("--fallback", default="data/sember_mcq.jsonl", help="Used as the master on first run, then snapshotted.")
    p.add_argument("--out", default="data/sember_mcq.jsonl", help="Filtered JSONL (the path the YAML loads).")
    args = p.parse_args()

    video_dir = Path(args.video_dir)
    if not video_dir.is_dir():
        print(f"ERROR: video dir does not exist: {video_dir}", file=sys.stderr)
        return 1

    available = {p.stem for p in video_dir.glob("*.mp4")}
    if not available:
        print(f"ERROR: no .mp4 files found in {video_dir}.", file=sys.stderr)
        print("       Download at least one mp4 from the Hugging Face dataset and place it " f"in {video_dir}, then re-run.", file=sys.stderr)
        return 1

    master = Path(args.master)
    out = Path(args.out)
    fallback = Path(args.fallback)

    if master.exists():
        src = master
    elif fallback.exists():
        # First run: snapshot the user-provided JSONL as the master so we
        # can re-filter on subsequent runs without losing the original.
        master.write_text(fallback.read_text())
        print(f"  snapshotted: {fallback} -> {master}")
        src = master
    else:
        print(f"ERROR: neither {master} nor {fallback} exists.", file=sys.stderr)
        print("       Download sember_mcq.jsonl from the Hugging Face dataset and place it " f"in {fallback.parent}/, then re-run.", file=sys.stderr)
        return 1

    kept: list[str] = []
    seen_videos: set[str] = set()
    skipped_videos: set[str] = set()
    with src.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            vid = doc.get("video_id", "")
            seen_videos.add(vid)
            if vid in available:
                kept.append(line)
            else:
                skipped_videos.add(vid)

    if not kept:
        print("ERROR: none of the videos on disk match any question in the JSONL.", file=sys.stderr)
        print(f"       videos on disk:   {sorted(available)}", file=sys.stderr)
        print(f"       videos referenced in JSONL ({len(seen_videos)} total): " f"{sorted(list(seen_videos))[:5]}...", file=sys.stderr)
        return 1

    with out.open("w") as f:
        for line in kept:
            f.write(line + "\n")

    print(f"  master jsonl:           {src}")
    print(f"  videos on disk:         {len(available)}")
    print(f"  videos referenced:      {len(seen_videos)}")
    print(f"  videos missing locally: {len(skipped_videos)}")
    print(f"  questions kept:         {len(kept)} / {len(kept) + sum(1 for _ in skipped_videos)}+")
    print(f"  written to:             {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
