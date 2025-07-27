"""Caption synchronization tool.

This module transcribes an input video and overlays word-by-word captions.
It follows the project README guidelines:

1. Extract word-level timestamps using one of the bundled tools
   (``whisper_timestamped`` by default).
2. Group words in batches of three.
3. Create MoviePy clips showing three words at a time where the active
   word is highlighted in green and the others are white with a black
   outline.
4. Render the final video in the ``output/`` directory with ``_captioned``
   appended to the original file name.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Tuple

from moviepy.editor import (
    CompositeVideoClip,
    TextClip,
    VideoFileClip,
)

TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools" / "whisper-timestamped"
if TOOLS_ROOT.exists():
    sys.path.insert(0, str(TOOLS_ROOT))

import whisper_timestamped as whisper_ts


WordTimestamp = Tuple[str, float, float]


def transcribe_audio(video_path: str, model_name: str = "base") -> List[WordTimestamp]:
    """Transcribe ``video_path`` and return a list of word level timestamps.

    Parameters
    ----------
    video_path:
        Path to the input video or audio file.
    model_name:
        Whisper model name to use. Defaults to ``"base"``.

    Returns
    -------
    list of tuples
        Each tuple contains ``(word, start, end)``.
    """

    model = whisper_ts.load_model(model_name)
    result = whisper_ts.transcribe(model, video_path)

    words: List[WordTimestamp] = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            words.append((word["text"].strip(), float(word["start"]), float(word["end"])))
    return words


def group_words(words: Iterable[WordTimestamp], group_size: int = 3) -> List[dict]:
    """Group ``words`` into dictionaries of ``group_size`` elements.

    Returns a list where each item is ``{"words": List[WordTimestamp],
    "start": float, "end": float}`` representing the time span of the group.
    """

    grouped: List[dict] = []
    words = list(words)
    for i in range(0, len(words), group_size):
        chunk = words[i : i + group_size]
        if not chunk:
            continue
        grouped.append(
            {
                "words": chunk,
                "start": chunk[0][1],
                "end": chunk[-1][2],
            }
        )
    return grouped


def _create_base_group_clip(
    words: List[WordTimestamp],
    width: int,
    height: int,
    font_size: int,
    font: str,
    spacing: int,
) -> Tuple[CompositeVideoClip, List[int]]:
    """Create the base (all-white) clip for a group of words.

    Returns the clip and a list with the ``x`` position for each word."""

    word_clips = [
        TextClip(
            w[0],
            fontsize=font_size,
            font=font,
            color="white",
            stroke_color="black",
            stroke_width=2,
        )
        for w in words
    ]

    total_width = sum(c.w for c in word_clips) + spacing * (len(word_clips) - 1)
    start_x = int((width - total_width) / 2)

    positions = []
    x = start_x
    base_clips = []
    for clip in word_clips:
        positions.append(x)
        base_clips.append(clip.set_position((x, height)).set_duration(0))
        x += clip.w + spacing

    base = CompositeVideoClip(base_clips, size=(width, height + font_size))
    return base, positions


def create_caption_clips(
    video_clip: VideoFileClip,
    grouped_words: Iterable[dict],
    font_size: int = 48,
    font: str = "Arial",
) -> List[CompositeVideoClip]:
    """Create caption clips highlighting words in sequence."""

    w, h = video_clip.size
    y_pos = int(h * 0.85)
    spacing = int(font_size / 2)
    clips: List[CompositeVideoClip] = []

    for group in grouped_words:
        words = group["words"]

        base_clip, x_positions = _create_base_group_clip(
            words, w, y_pos, font_size, font, spacing
        )

        base_clip = (
            base_clip.set_start(group["start"])
            .set_end(group["end"])
            .set_position((0, 0))
        )
        clips.append(base_clip)

        for idx, (text, start, end) in enumerate(words):
            highlight = TextClip(
                text,
                fontsize=font_size,
                font=font,
                color="#00FF00",
                stroke_color="black",
                stroke_width=2,
            )
            highlight = (
                highlight.set_position((x_positions[idx], y_pos))
                .set_start(start)
                .set_end(end)
            )
            clips.append(highlight)

    return clips


def process_video(
    video_path: str,
    model_name: str = "base",
    font_size: int = 48,
    font: str = "Arial",
) -> str | None:
    """Process ``video_path`` and generate a captioned video.

    Returns the output file path or ``None`` if the input file does not exist."""

    path = Path(video_path)
    if not path.is_file():
        print(f"Input video '{video_path}' not found. Nothing to do.")
        return None

    video_clip = VideoFileClip(str(path))

    words = transcribe_audio(str(path), model_name)
    groups = group_words(words)
    caption_clips = create_caption_clips(video_clip, groups, font_size, font)

    final_clip = CompositeVideoClip([video_clip, *caption_clips])

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{path.stem}_captioned.mp4"

    final_clip.write_videofile(str(output_path), codec="libx264", audio_codec="aac")
    return str(output_path)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Generate word-level captions")
    parser.add_argument("video", nargs="?", help="Path to input video")
    parser.add_argument("--model", default="base", help="Whisper model name")
    parser.add_argument("--font-size", type=int, default=48, dest="font_size")
    parser.add_argument("--font", default="Arial")
    args = parser.parse_args()

    if args.video:
        process_video(args.video, args.model, args.font_size, args.font)
    else:
        print("No input video provided. Exiting.")

