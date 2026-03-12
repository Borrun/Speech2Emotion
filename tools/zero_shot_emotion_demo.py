#!/usr/bin/env python3
"""Zero-shot emotion classification for one-text-per-line input files."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from transformers import pipeline

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


DEFAULT_LABELS = ["happy", "fear", "angry", "sad", "clam", "confused"]
DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
DEFAULT_TEMPLATE = "This text expresses {} emotion."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zero-shot emotion classifier")
    p.add_argument(
        "--input",
        default="annotater/train_texts_one_per_line.txt",
        help="Input txt file, one utterance per line",
    )
    p.add_argument(
        "--output_csv",
        default="wavs/emotion_outputs/zero_shot_emotion_results.csv",
        help="Output CSV path",
    )
    p.add_argument(
        "--output_json",
        default="wavs/emotion_outputs/zero_shot_emotion_results.json",
        help="Output JSON path",
    )
    p.add_argument(
        "--summary_json",
        default="wavs/emotion_outputs/zero_shot_emotion_summary.json",
        help="Output summary JSON path",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model name",
    )
    p.add_argument(
        "--labels",
        nargs="+",
        default=DEFAULT_LABELS,
        help="Candidate emotion labels",
    )
    p.add_argument(
        "--hypothesis_template",
        default=DEFAULT_TEMPLATE,
        help="Hypothesis template for NLI inference",
    )
    p.add_argument("--batch_size", type=int, default=16, help="Batch size")
    p.add_argument(
        "--multi_label",
        action="store_true",
        help="Enable independent multi-label scoring",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device",
    )
    p.add_argument(
        "--show_progress",
        action="store_true",
        help="Show tqdm progress bar",
    )
    return p.parse_args()


def resolve_device(device_arg: str) -> int:
    if device_arg == "cpu":
        return -1
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return 0
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return "mps"
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return -1


def read_texts(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def chunked(items: List[str], n: int) -> List[List[str]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


def classify_texts(args: argparse.Namespace, texts: List[str]) -> List[Dict]:
    clf = pipeline(
        "zero-shot-classification",
        model=args.model,
        device=resolve_device(args.device),
    )
    batches = chunked(texts, args.batch_size)
    iterator = batches
    if args.show_progress and tqdm is not None:
        iterator = tqdm(batches, desc="Zero-shot inference", unit="batch")

    rows: List[Dict] = []
    for batch in iterator:
        out = clf(
            batch,
            candidate_labels=args.labels,
            hypothesis_template=args.hypothesis_template,
            multi_label=args.multi_label,
        )
        if isinstance(out, dict):
            out = [out]
        for text, pred in zip(batch, out):
            labels = pred.get("labels", [])
            scores = pred.get("scores", [])
            top_label = labels[0] if labels else ""
            top_score = float(scores[0]) if scores else 0.0
            rows.append(
                {
                    "text": text,
                    "emotion": top_label,
                    "score": top_score,
                    "labels": labels,
                    "scores": [float(s) for s in scores],
                }
            )
    return rows


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["text", "emotion", "score"])
        for r in rows:
            wr.writerow([r["text"], r["emotion"], f"{r['score']:.6f}"])


def write_json(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def build_summary(rows: List[Dict], labels: List[str]) -> Dict:
    count = Counter(r["emotion"] for r in rows)
    score_acc = defaultdict(float)
    for r in rows:
        score_acc[r["emotion"]] += float(r["score"])

    stats = []
    for label in labels:
        c = int(count.get(label, 0))
        avg = (score_acc[label] / c) if c > 0 else 0.0
        stats.append({"emotion": label, "count": c, "avg_top_score": round(avg, 6)})

    return {
        "num_texts": len(rows),
        "labels": labels,
        "distribution": stats,
    }


def write_summary(path: Path, summary: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def print_summary(summary: Dict) -> None:
    print(f"Processed texts: {summary['num_texts']}")
    print("Emotion distribution:")
    for item in summary["distribution"]:
        print(
            f"  - {item['emotion']:>8}: {item['count']:>5} "
            f"(avg score={item['avg_top_score']:.3f})"
        )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    texts = read_texts(input_path)
    if not texts:
        raise RuntimeError("Input text file is empty after stripping blank lines.")

    rows = classify_texts(args, texts)
    summary = build_summary(rows, args.labels)

    write_csv(Path(args.output_csv), rows)
    write_json(Path(args.output_json), rows)
    write_summary(Path(args.summary_json), summary)
    print_summary(summary)
    print(f"Saved CSV: {args.output_csv}")
    print(f"Saved JSON: {args.output_json}")
    print(f"Saved summary: {args.summary_json}")


if __name__ == "__main__":
    main()
