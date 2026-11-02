"""
Step 1 of the NSL dataset pipeline:
  1. detect and crop the black bars (top/bottom, which also removes the burned-in caption)
  2. cut sentence clips with ffmpeg (re-encoded, so cuts are frame-accurate)
  3. flag frames where a graphic covers the signer, and mark clips to keep or reject

Input CSV columns (times in seconds): video,start,end,text
  video = file name inside --videos, e.g. house_2024_03_01.mp4

Usage:
  python prepare_clips.py --videos videos/ --captions sentences.csv --out clips/
"""
import argparse
import csv
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def sample_frames(video_path, n=20):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(total * 0.05, total * 0.95, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def detect_crop(frames, black_thresh=25):
    """Find the tallest band of non-black rows. Returns (top, height, width), all even."""
    h, w = frames[0].shape[:2]
    row_mean = np.median(
        [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean(axis=1) for f in frames], axis=0
    )
    content = list(row_mean > black_thresh) + [False]
    best, start = (0, 0), None
    for i, is_content in enumerate(content):
        if is_content and start is None:
            start = i
        elif not is_content and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    top, bottom = best
    crop_h = (bottom - top) // 2 * 2
    return top, crop_h, w // 2 * 2


def estimate_background(frames, top, crop_h):
    """Median color of the left/right edge strips = the blue background."""
    strips = []
    for f in frames:
        band = f[top:top + crop_h]
        s = int(band.shape[1] * 0.08)
        strips.append(band[:, :s].reshape(-1, 3))
        strips.append(band[:, -s:].reshape(-1, 3))
    return np.median(np.concatenate(strips), axis=0)


def cut_clip(video_path, start, end, out_path, top, crop_h, crop_w, pad):
    s = max(0.0, start - pad)
    duration = (end + pad) - s
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{s:.3f}", "-i", str(video_path), "-t", f"{duration:.3f}",
        "-vf", f"crop={crop_w}:{crop_h}:0:{top}",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-an",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def graphic_ratio(clip_path, bg, tol=40, min_match=0.6):
    """Share of frames where the edge strips no longer look like the blue background."""
    cap = cv2.VideoCapture(str(clip_path))
    bad = total = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        s = int(f.shape[1] * 0.08)
        edges = np.concatenate([f[:, :s].reshape(-1, 3), f[:, -s:].reshape(-1, 3)])
        dist = np.linalg.norm(edges.astype(np.float32) - bg, axis=1)
        if (dist < tol).mean() < min_match:
            bad += 1
        total += 1
    cap.release()
    return bad / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--captions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pad", type=float, default=1.0, help="seconds added each side")
    ap.add_argument("--max_bad", type=float, default=0.2, help="reject clip above this graphic ratio")
    ap.add_argument("--min_len", type=float, default=1.0)
    ap.add_argument("--max_len", type=float, default=15.0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_video = defaultdict(list)
    with open(args.captions, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_video[row["video"]].append(row)

    manifest = []
    for video_name, rows in by_video.items():
        video_path = Path(args.videos) / video_name
        frames = sample_frames(video_path)
        if not frames:
            print(f"skip (cannot read): {video_name}")
            continue
        top, crop_h, crop_w = detect_crop(frames)
        bg = estimate_background(frames, top, crop_h)
        print(f"{video_name}: crop top={top} h={crop_h} w={crop_w}, bg(BGR)={bg.astype(int)}")

        for i, row in enumerate(rows):
            start, end = float(row["start"]), float(row["end"])
            length = end - start
            clip_path = out_dir / f"{video_path.stem}_{i:04d}.mp4"
            if length < args.min_len or length > args.max_len:
                manifest.append([clip_path.name, row["text"], start, end, "", "reject_length"])
                continue
            cut_clip(video_path, start, end, clip_path, top, crop_h, crop_w, args.pad)
            ratio = graphic_ratio(clip_path, bg)
            status = "keep" if ratio <= args.max_bad else "reject_graphics"
            manifest.append([clip_path.name, row["text"], start, end, f"{ratio:.3f}", status])

    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip", "text", "start", "end", "graphic_ratio", "status"])
        w.writerows(manifest)
    kept = sum(1 for m in manifest if m[-1] == "keep")
    print(f"done: {kept} kept of {len(manifest)} clips -> {out_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()