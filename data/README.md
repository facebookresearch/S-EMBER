# S-EMBER Data

This directory is the expected local location for the S-EMBER benchmark data.
The public dataset is hosted on Hugging Face:

```text
https://huggingface.co/datasets/xidwang/S-EMBER/tree/main
```

The Hugging Face dataset layout is:

```text
xidwang/S-EMBER/
├── sember_mcq.jsonl
├── sember_grounding.jsonl
└── videos/
    ├── <video_id>.mp4
    └── ...
```

The finalized release contains **9,448 QA pairs** over **3,141 videos** and **388 hours** of video.
Both JSONL files reference the same set of video IDs. Each row keeps
`video_id` as the canonical evaluator ID and includes `video` as the
Hugging Face-relative media path `videos/<video_id>.mp4`.

Full benchmark reproduction requires the full JSONL files and all referenced videos.

## Download

From the repository root:

```bash
mkdir -p data/videos
pip install -U huggingface_hub
```

Download the finalized JSONL files:

```bash
huggingface-cli download xidwang/S-EMBER sember_mcq.jsonl \
    --repo-type dataset --local-dir data
huggingface-cli download xidwang/S-EMBER sember_grounding.jsonl \
    --repo-type dataset --local-dir data
```

Download all videos:

```bash
huggingface-cli download xidwang/S-EMBER --repo-type dataset \
    --include 'videos/*.mp4' --local-dir data
```

This produces the expected local layout:

```text
data/
├── sember_mcq.jsonl
├── sember_grounding.jsonl
└── videos/
    ├── <video_id>.mp4
    └── ...
```

For a small smoke test, you can instead download only a few videos from
`https://huggingface.co/datasets/xidwang/S-EMBER/tree/main/videos` and
place them under `data/videos/` with their original filenames.

If you downloaded files into another directory, the convenience helper can
move them into place:

```bash
bash scripts/data/place_data.sh ~/Downloads
```

## Question Categories

The 8 question categories are:

```text
location_trace
sequential_action
counting_objects_events
visual_detail_recall
temporal_ordering_recognition
time_duration
object_comparison
spatial_aware_reasoning
```

A typical video contributes a handful of questions across multiple
categories.

## Streaming Evaluation Contract

Each question carries a `question_time` in seconds. The evaluation
respects Grounded Streaming Episodic Retrieval (GSER): the model is shown
only the video segment `[0, question_time]`. Frames after `question_time`
are never sampled, so the model cannot use future context. This is
enforced inside the patched model wrappers in
`lmms_eval/models/simple/internvl3.py` and
`lmms_eval/models/simple/qwen3_vl.py`.

## MCQ JSONL Schema

Each line in `sember_mcq.jsonl` is a single JSON object:

```jsonc
{
  "question_id": "string",
  "video_id": "abc_start_..._end_...",          // canonical evaluator ID
  "video": "videos/abc_start_..._end_....mp4",  // Hugging Face media path
  "video_category_broad": "...",
  "video_category": "...",
  "question": "How long was I holding the clay kit before I put it back?",
  "question_time": 435.0,                        // seconds; trim video to [0, question_time]
  "question_category": "time_duration",
  "duration": 600.256,
  "options": [
    "A. ...",
    "B. ...",
    "C. ...",
    "D. ...",
    "E. ..."
  ],
  "correct_index": 2,
  "correct_letter": "C",
  "correct_option_source": "GT1",
  "ground_truths": ["..."],
  "answer_start_time": 377.0,
  "answer_end_time": 397.0
}
```

Required at evaluation time: `video_id`, `question_time`,
`question_category`, `question`, `options`, and either `correct_letter` or
`correct_index`. The `video` field is for Hugging Face visualization and is
ignored by the local evaluator.

## Grounding JSONL Schema

Each line in `sember_grounding.jsonl` is a single JSON object:

```jsonc
{
  "question_id": "string",
  "video_id": "abc_start_..._end_...",          // canonical evaluator ID
  "video": "videos/abc_start_..._end_....mp4",  // Hugging Face media path
  "video_category_broad": "...",
  "video_category": "...",
  "question": "How much time did I spend using the paper cutter?",
  "question_time": 300.0,
  "question_category": "time_duration",
  "memory_recency": 154.0,
  "duration": 599.99,
  "answer": "1 minute and 27 seconds.",
  "answer_start_time": 146.0,
  "answer_end_time": 233.0,
  "answer_range": 87.0,
  "answers": [
    {
      "answer_text": "1 minute and 27 seconds.",
      "start_ts": 146.0,
      "end_ts": 233.0,
      "answer_rater_group": "Group A: optimize for efficiency"
    }
  ]
}
```

Required at evaluation time: `video_id`, `question_time`,
`question_category`, `question`, `answer_start_time`, and
`answer_end_time`. `answer` and `answers` are used for answer-text
inspection and optional judged correctness. The `video` field is for
Hugging Face visualization and is ignored by the local evaluator.

## Troubleshooting

- `ERROR: S-EMBER data not found.` from `run_mcq_smoke.sh`: `data/sember_mcq.jsonl`
  or `data/videos/*.mp4` is missing. Download the MCQ JSONL and at least one
  matching video.
- `ERROR: none of the videos on disk match any question in the JSONL.`: the
  mp4 filenames in `data/videos/` do not match `video_id` values in the JSONL.
  Keep the original Hugging Face filenames exactly.
- `Video path .../<id>.mp4 does not exist, skipping sample...`: either the
  filtering step was skipped or `SEMBER_VIDEO_DIR` points to the wrong video
  directory.
- `CUDA out of memory` at 128 frames on a 24 GB card: set `NUM_PROCESSES=1`
  for a single-GPU smoke test or downscale with `NUM_FRAME=16 TOTAL_MAX_NUM=16`.
