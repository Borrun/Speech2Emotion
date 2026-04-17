#!/usr/bin/env python3
"""
Render eye socket PNGs based on annotater/app.py ALLOWED_TYPES.

Defaults:
  - data source: annotater/data/*.txt
  - output dir: outputs/eye_socket_allowed_types
  - base types:
      happy/sad/angry/fear/calm
    exported as one image per level 0..5
  - confused types:
      happy_confused/sad_confused/angry_confused/fear_confused/calm_confused
    exported as one image per type
  - canvas x: [-1200, 1200]
  - canvas y: [-600, 600]

The template txt format follows the same block parsing used by test_server.py.
Only the outer eye socket groups (group 0 and group 1) are drawn.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


Point = Tuple[float, float]
Group = List[Point]
FrameGroups = List[Group]

BASE_TYPES = ["happy", "sad", "angry", "fear", "calm"]
CONFUSED_TYPES = [
    "happy_confused",
    "sad_confused",
    "angry_confused",
    "fear_confused",
    "calm_confused",
]
DEFAULT_TYPES = BASE_TYPES + CONFUSED_TYPES
DEFAULT_LEVEL_TO_FRAME = {0: 5, 1: 18, 2: 38, 3: 68, 4: 98, 5: 130}
DEFAULT_SOCKET_GROUPS = (0, 1)
DEFAULT_CANVAS_X = (-1200.0, 1200.0)
DEFAULT_CANVAS_Y = (-600.0, 600.0)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Render eye socket curves for each emotion type and level."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "annotater" / "data",
        help="Directory containing emotion template txt files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "outputs" / "eye_socket_allowed_types",
        help="Directory to save rendered PNG files.",
    )
    parser.add_argument(
        "--types",
        nargs="*",
        default=None,
        help="Optional emotion types to render.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output image DPI.",
    )
    return parser.parse_args()


def discover_types(requested: Sequence[str] | None) -> List[str]:
    if requested:
        return [item.strip() for item in requested if item.strip()]
    return list(DEFAULT_TYPES)


def load_emotion_template(txt_path: Path) -> Dict[int, FrameGroups]:
    blocks: Dict[int, FrameGroups] = {}
    current: Dict[Tuple[int, str], List[Group]] = {}

    with txt_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) == 65:
                try:
                    block_index = int(float(parts[0]))
                except ValueError:
                    continue
                shape_id = "0"
                num_parts = parts[1:]
            elif len(parts) >= 66:
                try:
                    block_index = int(float(parts[0]))
                except ValueError:
                    continue
                shape_id = parts[1]
                num_parts = parts[2:66]
            else:
                continue

            try:
                numbers = [float(item) for item in num_parts]
            except ValueError:
                continue

            if len(numbers) != 64:
                continue

            points = [(numbers[i], numbers[i + 1]) for i in range(0, 64, 2)]
            key = (block_index, shape_id)
            current.setdefault(key, []).append(points)

            if len(current[key]) == 4 and block_index not in blocks:
                blocks[block_index] = current[key][:4]

    return blocks


def nearest_frame(frames: Iterable[int], target_frame: int) -> int:
    frame_list = list(frames)
    if not frame_list:
        raise ValueError("no frames available")
    return min(frame_list, key=lambda value: abs(value - target_frame))


def setup_axis(ax: plt.Axes) -> None:
    ax.set_xlim(*DEFAULT_CANVAS_X)
    ax.set_ylim(*DEFAULT_CANVAS_Y)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def draw_groups(ax: plt.Axes, groups: FrameGroups, group_ids: Sequence[int]) -> None:
    for group_id in group_ids:
        if group_id >= len(groups):
            continue
        points = groups[group_id]
        if not points:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        ax.plot(xs, ys, color="black", linewidth=2.4, solid_capstyle="round")


def resolve_template_path(data_dir: Path, emotion_type: str) -> Path:
    return data_dir / f"{emotion_type}.txt"


def confused_frame(frames: Iterable[int]) -> int:
    frame_list = sorted(set(frames))
    if not frame_list:
        raise ValueError("no frames available")
    # Confused templates in annotater/data are short; use the last block as the
    # representative shape when only one image is needed.
    return frame_list[-1]


def render_level_image(
    emotion_type: str,
    level: int,
    frame: int,
    groups: FrameGroups,
    out_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=dpi)
    setup_axis(ax)
    draw_groups(ax, groups, DEFAULT_SOCKET_GROUPS)
    # Keep metadata in the saved image for debugging while staying visually empty.
    fig.text(0.01, 0.01, f"{emotion_type} level {level} frame {frame}", fontsize=8, alpha=0.0)
    fig.tight_layout(pad=0)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_dir: Path = args.data_dir
    out_dir: Path = args.out_dir

    if not data_dir.is_dir():
        raise FileNotFoundError(f"data dir not found: {data_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    emotion_types = discover_types(args.types)

    if not emotion_types:
        raise RuntimeError(f"no template txt found in {data_dir}")

    rendered = 0
    skipped: List[str] = []

    for emotion_type in emotion_types:
        txt_path = resolve_template_path(data_dir, emotion_type)
        if not txt_path.is_file():
            skipped.append(f"{emotion_type}: missing file")
            continue

        frames = load_emotion_template(txt_path)
        if not frames:
            skipped.append(f"{emotion_type}: no valid frames")
            continue

        if emotion_type in BASE_TYPES:
            for level, target_frame in DEFAULT_LEVEL_TO_FRAME.items():
                frame = nearest_frame(frames.keys(), target_frame)
                out_path = out_dir / f"{emotion_type}_level_{level}.png"
                render_level_image(
                    emotion_type=emotion_type,
                    level=level,
                    frame=frame,
                    groups=frames[frame],
                    out_path=out_path,
                    dpi=args.dpi,
                )
                rendered += 1
                print(f"[ok] {emotion_type} level {level} -> {out_path}")
        else:
            frame = confused_frame(frames.keys())
            out_path = out_dir / f"{emotion_type}.png"
            render_level_image(
                emotion_type=emotion_type,
                level=-1,
                frame=frame,
                groups=frames[frame],
                out_path=out_path,
                dpi=args.dpi,
            )
            rendered += 1
            print(f"[ok] {emotion_type} -> {out_path}")

    print(f"rendered {rendered} images into {out_dir}")
    if skipped:
        print("skipped:")
        for item in skipped:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
