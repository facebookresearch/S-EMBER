"""Task helpers for the S-EMBER MCQ benchmark.

Environment variables
---------------------
SEMBER_VIDEO_DIR
    Absolute path to the directory containing ``<video_id>.mp4`` files.
    Required for evaluation.
"""

import os
import re
from collections import defaultdict

from loguru import logger as eval_logger

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

CHOICE_LABELS = ["A", "B", "C", "D", "E"]

VIDEO_DIR = os.environ.get("SEMBER_VIDEO_DIR", "")

# Cache to avoid repeated filesystem checks and log spam for missing videos.
_missing_videos_logged = set()


def sember_doc_to_visual(doc):
    """Resolve a sample's video path; trim to question_time if provided.

    Returns a list with a single dict the model wrapper understands:
        ``[{"video_path": <abs_path>, "video_end": <seconds_or_none>}]``

    Returns ``None`` when the video file is missing on disk so the model
    wrapper can short-circuit and emit an empty prediction.
    """
    if not VIDEO_DIR:
        raise RuntimeError("SEMBER_VIDEO_DIR is not set. Export it to the directory holding " "the benchmark mp4 files before running evaluation.")

    video_path = doc["video_id"] + ".mp4"
    full_path = os.path.join(VIDEO_DIR, video_path)
    if not os.path.exists(full_path):
        if video_path not in _missing_videos_logged:
            eval_logger.warning(f"Video path {full_path} does not exist, skipping sample...")
            _missing_videos_logged.add(video_path)
        return None

    # Trim the visible video to the moment the question was asked so the model
    # cannot peek at frames after question_time.
    question_time = doc.get("question_time", None)
    return [{"video_path": full_path, "video_end": question_time}]


def sember_doc_to_text_mcq(doc, lmms_eval_specific_kwargs=None):
    """Build the MCQ prompt from a question and its options."""
    question = doc["question"]
    options = doc["options"]

    labeled = []
    for i, opt in enumerate(options):
        if re.match(r"^[A-E]\.\s", opt):
            labeled.append(opt)
        else:
            labeled.append(f"{CHOICE_LABELS[i]}. {opt}")
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


def _parse_mcq_choice(response_text, num_options=5):
    """Extract the chosen letter from the model response, or None if unparseable."""
    if not response_text:
        return None
    text = response_text.strip().upper()
    valid_labels = CHOICE_LABELS[:num_options]
    if text in valid_labels:
        return text
    match = re.search(r"\b([A-E])\b", text)
    if match and match.group(1) in valid_labels:
        return match.group(1)
    return None


def sember_process_results_mcq(doc, results):
    """Compare predicted letter against the ground-truth letter."""
    pred = results[0]
    options = doc.get("options", [])
    correct_letter = doc.get("correct_letter") or CHOICE_LABELS[doc.get("correct_index", 0)]

    pred_letter = _parse_mcq_choice(pred, len(options))
    is_correct = pred_letter == correct_letter if pred_letter is not None else False

    return {
        "sember_mcq_accuracy": {
            "question_id": doc.get("question_id", "missing"),
            "question_category": doc.get("question_category", "missing"),
            "question": doc.get("question", ""),
            "video_id": doc.get("video_id", ""),
            "question_time": doc.get("question_time"),
            "options": options,
            "correct_index": doc.get("correct_index"),
            "correct_letter": correct_letter,
            "pred_raw": pred,
            "pred_letter": pred_letter,
            "is_correct": is_correct,
        }
    }


def sember_aggregate_results_mcq(results):
    """Aggregate accuracy overall and per question category."""
    category2stats = defaultdict(lambda: {"correct": 0, "total": 0, "parsed": 0})

    for result in results:
        cat = result.get("question_category", "unknown")
        category2stats[cat]["total"] += 1
        if result.get("pred_letter") is not None:
            category2stats[cat]["parsed"] += 1
        if result.get("is_correct"):
            category2stats[cat]["correct"] += 1

    eval_logger.info("MCQ Accuracy by category:")
    for cat in QUESTION_CATEGORIES:
        s = category2stats.get(cat, {"correct": 0, "total": 0, "parsed": 0})
        if s["total"] > 0:
            acc = 100.0 * s["correct"] / s["total"]
            eval_logger.info(f"  {cat}: {acc:.1f}% ({s['correct']}/{s['total']}, parsed={s['parsed']})")

    total_correct = sum(v["correct"] for v in category2stats.values())
    total_count = sum(v["total"] for v in category2stats.values())
    total_parsed = sum(v["parsed"] for v in category2stats.values())
    overall = 100.0 * total_correct / total_count if total_count > 0 else 0.0

    eval_logger.info(f"Overall MCQ accuracy: {overall:.1f}% ({total_correct}/{total_count}, " f"parsed={total_parsed}/{total_count})")

    pred_dist = defaultdict(int)
    for r in results:
        if r.get("pred_letter"):
            pred_dist[r["pred_letter"]] += 1
    eval_logger.info("Predicted choice distribution: " + ", ".join(f"{l}={pred_dist.get(l, 0)}" for l in CHOICE_LABELS))

    return overall


# ---------------------------------------------------------------------------
# Grounding variant: free-text answer + temporal interval [t_start, t_end]
# ---------------------------------------------------------------------------


def sember_doc_to_text_with_grounding(doc, lmms_eval_specific_kwargs=None):
    """Build the grounding prompt: ask for an answer and a time interval."""
    answer_prompt = (
        "After reviewing the video, provide the best answer to the following question. "
        "Answer in 1-2 sentences.\n"
        "Also provide the time interval (in seconds) where the answer evidence appears in the video.\n\n"
        "Use this exact format:\n"
        "Answer: <your answer>\n"
        "Time: [<start_seconds>, <end_seconds>]"
    )
    return answer_prompt + "\n\n" + doc["question"]


def _parse_grounding_response(pred):
    """Parse model response into (answer_text, pred_start, pred_end).

    Expected format::

        Answer: <text>
        Time: [<start>, <end>]

    If parsing fails, the raw prediction is returned as the answer and the
    times are ``None`` (counted as un-parseable in the aggregator).
    """
    answer_text = pred.strip()
    pred_start = None
    pred_end = None

    answer_match = re.search(r"Answer:\s*(.+?)(?:\n|Time:)", pred, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()

    time_match = re.search(r"Time:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", pred)
    if time_match:
        try:
            pred_start = float(time_match.group(1))
            pred_end = float(time_match.group(2))
        except ValueError:
            pass

    return answer_text, pred_start, pred_end


def sember_process_results_grounding(doc, results):
    """Extract pred answer + pred temporal interval from model output."""
    pred = results[0]
    pred_answer, pred_start, pred_end = _parse_grounding_response(pred)

    return {
        "sember_temporal_iou": {
            "question_id": doc.get("question_id", "missing"),
            "question_category": doc.get("question_category", "missing"),
            "video_id": doc.get("video_id", ""),
            "question": doc.get("question", ""),
            "question_time": doc.get("question_time"),
            "pred_raw": pred,
            "pred_answer": pred_answer,
            "pred_start_time": pred_start,
            "pred_end_time": pred_end,
            "answer": doc.get("answer", ""),
            "answers": doc.get("answers", []),
            "answer_start_time": doc.get("answer_start_time"),
            "answer_end_time": doc.get("answer_end_time"),
            "memory_recency": doc.get("memory_recency"),
            "answer_range": doc.get("answer_range"),
            "duration": doc.get("duration"),
            "video_category": doc.get("video_category"),
        }
    }


def _temporal_iou(pred_start, pred_end, gt_start, gt_end):
    """Compute temporal Intersection-over-Union between two intervals.

    Returns 0.0 for un-parseable predictions or missing GT, and 0.0 for
    intervals with no overlap (rather than a negative value).
    """
    if any(v is None for v in (pred_start, pred_end, gt_start, gt_end)):
        return 0.0
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    inter = max(0.0, inter_end - inter_start)
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    if union <= 0:
        return 0.0
    return inter / union


def sember_aggregate_temporal_iou(results):
    """Aggregate mean temporal IoU overall and per question category."""
    category2iou = defaultdict(lambda: {"total_iou": 0.0, "count": 0, "parseable": 0})

    for result in results:
        cat = result.get("question_category", "unknown")
        gt_start = result.get("answer_start_time")
        gt_end = result.get("answer_end_time")
        pred_start = result.get("pred_start_time")
        pred_end = result.get("pred_end_time")

        category2iou[cat]["count"] += 1
        if pred_start is not None and pred_end is not None:
            category2iou[cat]["parseable"] += 1
            category2iou[cat]["total_iou"] += _temporal_iou(pred_start, pred_end, gt_start, gt_end)

    eval_logger.info("Temporal IoU by category:")
    for cat in QUESTION_CATEGORIES:
        s = category2iou.get(cat, {"total_iou": 0.0, "count": 0, "parseable": 0})
        if s["count"] > 0:
            mean_iou = 100.0 * s["total_iou"] / s["count"]
            parse_rate = 100.0 * s["parseable"] / s["count"]
            eval_logger.info(f"  {cat}: mIoU={mean_iou:.1f}% " f"(parseable={s['parseable']}/{s['count']}={parse_rate:.0f}%)")

    total_iou = sum(v["total_iou"] for v in category2iou.values())
    total_count = sum(v["count"] for v in category2iou.values())
    total_parse = sum(v["parseable"] for v in category2iou.values())
    overall = 100.0 * total_iou / total_count if total_count > 0 else 0.0
    parse_rate = 100.0 * total_parse / total_count if total_count > 0 else 0.0
    eval_logger.info(f"Overall temporal IoU: {overall:.1f}% " f"(parseable={total_parse}/{total_count}={parse_rate:.0f}%)")
    return overall
