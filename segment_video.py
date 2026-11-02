"""
segment_video.py -- one video in, one-sentence clips (with sound) + keypoints out. No captions needed.

Why the voice: hand pauses also happen in the middle of a sentence, so hands alone cut at random places.
The narration pauses between sentences, so by default the clips follow the voice.

Steps
  1. find and crop the black bars (this also removes the burned-in caption)
  2. run MediaPipe Holistic on every frame: body, face points, and both hands with all 21 finger
     points, plus a signer silhouette. A frame is dropped when no signer is found (full-screen slide)
     or when a graphic touches the signer. A picture in a corner, away from the signer, is ignored.
  3. --mode speech (default): a sentence = a stretch of voice between pauses of --speech_pause seconds.
     The start is where the voice starts. The end is where the voice stops, or later if the signer is
     still signing then: the clip waits until both hands are lowered (rest position) for --rest_hold
     seconds, at most --max_extend seconds. A hand held still in the air does not count as the end.
     --mode hands: a clip = hands raised and moving, split at the longest pauses of the hands
  4. cut each clip with ffmpeg (with sound, soft fade at both ends), and save its keypoints (.npz)
  5. write a review copy of each clip with the keypoints, a box on the signer, and a timeline
     showing the detected sign boundaries (out/review/)

Tuning without waiting again: add --reuse to skip the slow keypoint step and reuse out/keypoints/all.npz

Install:  pip install opencv-python mediapipe numpy scipy matplotlib   (and ffmpeg)
Run:      python segment_video.py --video test_video.mp4 --out out/
"""
import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from scipy.signal import find_peaks, savgol_filter

# MediaPipe Pose landmark ids: shoulders, wrists, index fingertips
KEYS = [11, 12, 15, 16, 19, 20]


# ---------- crop and background ----------
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
    """Tallest band of non-black rows. Returns (top, height, width), all even."""
    w = frames[0].shape[1]
    row_mean = np.median(
        [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean(axis=1) for f in frames], axis=0
    )
    content = list(row_mean > black_thresh) + [False]
    best, start = (0, 0), None
    for i, c in enumerate(content):
        if c and start is None:
            start = i
        elif not c and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    top, bottom = best
    return top, (bottom - top) // 2 * 2, w // 2 * 2


def estimate_background(frames, top, crop_h):
    strips = []
    for f in frames:
        band = f[top:top + crop_h]
        s = int(band.shape[1] * 0.08)
        strips.append(band[:, :s].reshape(-1, 3))
        strips.append(band[:, -s:].reshape(-1, 3))
    return np.median(np.concatenate(strips), axis=0)


# ---------- per-frame features ----------
def make_model(complexity, static=False):
    """MediaPipe Holistic: pose + face + both hands (21 points each) + signer silhouette."""
    return mp.solutions.holistic.Holistic(
        static_image_mode=static,
        model_complexity=complexity,
        enable_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def graphic_score(frame, res, bg, tol, grow=2, ring=7):
    """Is a graphic touching the signer?
    score = share of a thin ring around the signer's silhouette that is not blue background.
    A graphic far from the signer (for example a picture in a corner) does not count.
    Returns (score, outside mask, signer mask); masks are at a small size (160 px wide)."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (160, max(1, int(160 * h / w))), interpolation=cv2.INTER_AREA)
    nonbg = np.linalg.norm(small.astype(np.float32) - bg, axis=2) > tol
    mask = res.segmentation_mask
    if mask is None:
        return 1.0, nonbg, np.zeros_like(nonbg)
    sil = (cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8)
    if not sil.any():
        return 1.0, nonbg, sil.astype(bool)
    inner = cv2.dilate(sil, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)).astype(bool)
    outer = cv2.dilate(sil, np.ones((2 * ring + 1, 2 * ring + 1), np.uint8)).astype(bool)
    ring_mask = outer & ~inner
    score = float((nonbg & ring_mask).sum() / max(1, ring_mask.sum()))
    return score, nonbg & ~inner, inner


def lm_array(landmarks, n, cols):
    """Landmarks -> (n, cols) array, all NaN when the part was not found. cols: 3 = x,y,z; 4 = + visibility."""
    if landmarks is None:
        return np.full((n, cols), np.nan, dtype=np.float32)
    rows = [[p.x, p.y, p.z, p.visibility][:cols] for p in landmarks.landmark]
    return np.array(rows, dtype=np.float32)


def extract(video_path, top, crop_h, bg, complexity, tol=45, thr=0.12):
    """Returns pose (N,33,4), left_hand (N,21,3), right_hand (N,21,3), bad (N,), fps.
    Coordinates are 0..1 in the cropped frame. Frames with a graphic or no signer are NaN and bad=True."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    model = make_model(complexity)
    poses, lefts, rights, bad = [], [], [], []
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = frame[top:top + crop_h]
        res = model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pose = lm_array(res.pose_landmarks, 33, 4)
        left = lm_array(res.left_hand_landmarks, 21, 3)
        right = lm_array(res.right_hand_landmarks, 21, 3)
        is_bad = True
        if res.pose_landmarks and pose[11, 3] > 0.5 and pose[12, 3] > 0.5:
            score, _, _ = graphic_score(frame, res, bg, tol)
            is_bad = score > thr
        if is_bad:
            pose[:], left[:], right[:] = np.nan, np.nan, np.nan
        poses.append(pose)
        lefts.append(left)
        rights.append(right)
        bad.append(is_bad)
        n += 1
        if n % 500 == 0:
            print(f"  frames {n}/{total}")
    cap.release()
    model.close()
    return np.array(poses), np.array(lefts), np.array(rights), np.array(bad), fps


def hand_points(pose):
    """Shoulders, wrists, index fingertips from the pose, as (N,6,2) in shoulder-widths."""
    xy = pose[:, KEYS, :2]
    center = xy[:, :2].mean(axis=1, keepdims=True)
    width = np.linalg.norm(xy[:, 0] - xy[:, 1], axis=1)[:, None, None] + 1e-6
    return (xy - center) / width  # y grows downward


def hand_speed(pts, bad, fps):
    """Smoothed hand speed per frame, plus smoothed hand positions (N,4,2)."""
    n = len(pts)
    flat = pts[:, 2:6].reshape(n, -1).copy()  # wrists + index tips
    idx = np.arange(n)
    for j in range(flat.shape[1]):
        good = ~np.isnan(flat[:, j])
        flat[:, j] = np.interp(idx, idx[good], flat[good, j]) if good.sum() > 1 else 0.0
    win = max(5, int(fps * 0.2) // 2 * 2 + 1)
    flat = savgol_filter(flat, win, 2, axis=0)
    vel = np.diff(flat, axis=0, prepend=flat[:1])
    speed = np.linalg.norm(vel.reshape(n, 4, 2), axis=2).mean(axis=1) * fps
    speed[bad] = 0.0
    return speed, flat.reshape(n, 4, 2)


# ---------- segmentation ----------
def runs_from_mask(mask, offset=0):
    runs, start = [], None
    for i, m in enumerate(np.append(mask, False)):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append([start + offset, i + offset])
            start = None
    return runs


def find_clips(active, bad, gap_frames, min_frames):
    """Active runs, merged over short stills, never merged across a graphic."""
    clips = []
    for s, e in runs_from_mask(~bad):  # good stretches only
        merged = []
        for r in runs_from_mask(active[s:e], offset=s):
            if merged and r[0] - merged[-1][1] <= gap_frames:
                merged[-1][1] = r[1]
            else:
                merged.append(r)
        clips += [r for r in merged if r[1] - r[0] >= min_frames]
    return clips


def split_long(clip, moving, max_frames, min_pause, min_frames):
    """Split a clip longer than max_frames at its longest pause (hands not moving), repeatedly."""
    s, e = clip
    if e - s <= max_frames:
        return [clip]
    best = None
    for a, b in runs_from_mask(~moving[s:e], offset=s):
        if b - a >= min_pause and a - s >= min_frames and e - b >= min_frames:
            if best is None or b - a > best[1] - best[0]:
                best = (a, b)
    if best is None:
        return [clip]
    return (split_long([s, best[0]], moving, max_frames, min_pause, min_frames)
            + split_long([best[1], e], moving, max_frames, min_pause, min_frames))


def sign_boundaries(speed, start, end, fps, prominence, min_sign):
    seg = speed[start:end]
    seg = seg / (np.percentile(seg, 95) + 1e-6)
    peaks, _ = find_peaks(-seg, prominence=prominence, distance=max(1, int(fps * min_sign)))
    return [start + int(p) for p in peaks]


# ---------- output ----------
def cut_clip(video_path, t0, t1, out_path, top, crop_h, crop_w, fade=0.15):
    d = t1 - t0
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-ss", f"{t0:.3f}", "-i", str(video_path), "-t", f"{d:.3f}",
        "-vf", f"crop={crop_w}:{crop_h}:0:{top}",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-c:a", "aac", "-b:a", "128k",
    ]
    if fade > 0:
        cmd += ["-af", f"afade=t=in:st=0:d={fade},afade=t=out:st={max(0.0, d - fade):.3f}:d={fade}"]
    subprocess.run(cmd + [str(out_path)], check=True)


def draw_points(frame, pts, conns, color, min_vis=None):
    """Draw lines and dots for landmarks (0..1 coordinates). Returns the pixel points that were drawn."""
    h, w = frame.shape[:2]
    xy = pts[:, :2] * np.array([w, h], dtype=np.float32)
    ok = ~np.isnan(xy).any(axis=1)
    if min_vis is not None:
        ok &= np.nan_to_num(pts[:, 3]) > min_vis
    for a, b in conns:
        if ok[a] and ok[b]:
            cv2.line(frame, (int(xy[a, 0]), int(xy[a, 1])), (int(xy[b, 0]), int(xy[b, 1])), color, 2)
    for k in np.where(ok)[0]:
        cv2.circle(frame, (int(xy[k, 0]), int(xy[k, 1])), 3, color, -1)
    return xy[ok]


def render_overlay(clip_path, out_path, kp, fps, sign_range, bounds):
    """Copy of a clip with body/face/finger points, a green box on the signer, and a boundary timeline.
    sign_range = (first, last) frame inside the clip where signing was detected; bounds = sign boundary frames."""
    pose, left, right = kp["pose"], kp["left_hand"], kp["right_hand"]
    cap = cv2.VideoCapture(str(clip_path))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = Path(out_path).with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    pose_conn = list(mp.solutions.holistic.POSE_CONNECTIONS)
    hand_conn = list(mp.solutions.holistic.HAND_CONNECTIONS)
    total = len(pose)
    scale = max(0.5, w / 1000)
    bar_y, bar_h = h - 16, 10
    i = 0
    while i < total:
        ok, frame = cap.read()
        if not ok:
            break
        shown = [
            draw_points(frame, pose[i], pose_conn, (0, 255, 255), min_vis=0.3),
            draw_points(frame, left[i], hand_conn, (255, 0, 255)),
            draw_points(frame, right[i], hand_conn, (255, 255, 0)),
        ]
        shown = [p for p in shown if len(p)]
        if shown:
            p = np.concatenate(shown)
            x0, y0 = p.min(axis=0)
            x1, y1 = p.max(axis=0)
            mx, my = 0.06 * (x1 - x0), 0.06 * (y1 - y0)
            # pose points start at the eyes, so lift the top by the nose-to-shoulder distance to include the head
            head = 0.0
            if not np.isnan(pose[i][[0, 11, 12], 1]).any():
                head = max(0.0, (pose[i][11, 1] + pose[i][12, 1]) / 2 - pose[i][0, 1]) * h
            cv2.rectangle(frame, (int(x0 - mx), int(y0 - my - head)), (int(x1 + mx), int(y1 + my)), (0, 200, 0), 2)
        else:
            cv2.putText(frame, "NO SIGNER / GRAPHIC", (10, int(60 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (0, 0, 255), 2)
        # timeline: gray = whole clip, green = detected signing, red ticks = sign boundaries, white = now
        cv2.rectangle(frame, (0, bar_y), (w, bar_y + bar_h), (60, 60, 60), -1)
        cv2.rectangle(frame, (int(sign_range[0] / total * w), bar_y), (int(sign_range[1] / total * w), bar_y + bar_h),
                      (0, 160, 0), -1)
        for b in bounds:
            cv2.line(frame, (int(b / total * w), bar_y), (int(b / total * w), bar_y + bar_h), (0, 0, 255), 2)
        cv2.line(frame, (int(i / total * w), bar_y - 4), (int(i / total * w), bar_y + bar_h + 4), (255, 255, 255), 2)
        sign_no = 1 + sum(1 for b in bounds if b <= i)
        cv2.putText(frame, f"{i / fps:5.1f}s  sign {sign_no}/{len(bounds) + 1}", (10, int(30 * scale) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
        if any(abs(i - b) <= 2 for b in bounds):
            cv2.putText(frame, "BOUNDARY", (w // 2 - int(90 * scale), int(30 * scale) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), 2)
        writer.write(frame)
        i += 1
    cap.release()
    writer.release()
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(tmp), "-i", str(clip_path),
                    "-map", "0:v", "-map", "1:a:0?", "-c:v", "libx264", "-crf", "20", "-c:a", "copy",
                    str(out_path)], check=True)
    tmp.unlink()


def save_plot(speed, moving, f0, f1, s, e, bounds, fps, path, voice=None, next_voice=None):
    """Hand speed around the clip, so you can see whether the hands are still moving when the cut happens.
    green lines = where the clip starts and ends, blue = the voice, gray dashes = the next sentence's voice."""
    a, b = max(0, f0 - int(1.0 * fps)), min(len(speed), f1 + int(4.0 * fps))
    t = np.arange(a, b) / fps
    plt.figure(figsize=(9, 2.6))
    plt.plot(t, speed[a:b], lw=1, color="0.3")
    for x0, x1 in runs_from_mask(moving[a:b], offset=a):
        plt.axvspan(x0 / fps, x1 / fps, color="orange", alpha=0.25, lw=0)
    for x in (f0, f1):
        plt.axvline(x / fps, color="green", lw=2)
    if voice:
        for x in voice:
            plt.axvline(x, color="blue", lw=1)
    if next_voice is not None:
        plt.axvline(next_voice, color="gray", lw=1, ls="--")
    for x in bounds:
        plt.axvline(x / fps, color="r", lw=0.6)
    plt.xlabel("time (s)   green = clip start/end, blue = voice, orange = hands up")
    plt.ylabel("hand speed")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


# ---------- voice ----------
def load_audio(video_path, sr=16000):
    """Mono audio of the video as float32, or an empty array when the video has no sound."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(video_path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0 or not res.stdout:
        return np.zeros(0, np.float32), sr
    return np.frombuffer(res.stdout, np.int16).astype(np.float32) / 32768.0, sr


def speech_spans(audio, sr, min_pause, min_span, hop=0.02, reach=0.3):
    """Voiced stretches separated by silences of at least min_pause seconds.
    Loud parts are found first, then each stretch is widened (up to `reach` seconds each side)
    over the quieter start and end of words, so soft consonants and fading endings are kept.
    Returns (spans, pauses) as lists of (start_s, end_s)."""
    n = int(sr * hop)
    frames = audio[: len(audio) // n * n].reshape(-1, n)
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-8)
    db = np.convolve(np.pad(db, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    floor, loud = np.percentile(db, 10), np.percentile(db, 90)
    core = db > floor + 0.3 * (loud - floor)
    edge = db > floor + 0.15 * (loud - floor)
    ext = int(reach / hop)
    voiced = np.zeros(len(db), bool)
    for a, b in runs_from_mask(core):
        lo = a
        while lo > 0 and edge[lo - 1] and a - lo < ext:
            lo -= 1
        hi = b
        while hi < len(db) and edge[hi] and hi - b < ext:
            hi += 1
        voiced[lo:hi] = True
    spans = []
    for a, b in runs_from_mask(voiced):
        t0, t1 = a * hop, b * hop
        if spans and t0 - spans[-1][1] < min_pause:
            spans[-1][1] = t1
        else:
            spans.append([t0, t1])
    spans = [sp for sp in spans if sp[1] - sp[0] >= min_span]
    # follow the fading tail of each sentence down to the room noise of the pause after it
    for k, sp in enumerate(spans):
        nxt = spans[k + 1][0] if k + 1 < len(spans) else len(db) * hop
        a, b = int((sp[1] + 0.15) / hop), int((nxt - 0.1) / hop)
        noise = float(np.percentile(db[a:b], 20)) if b - a >= 4 else float(floor)
        i = int(sp[1] / hop)
        stop = min(len(db), int((sp[1] + 0.6) / hop), int((nxt - 0.2) / hop))
        while i < stop and db[i] > noise + 4.0:
            i += 1
        sp[1] = max(sp[1], i * hop)
    spans = [tuple(sp) for sp in spans]
    pauses = [(spans[i][1], spans[i + 1][0]) for i in range(len(spans) - 1)]
    return spans, pauses


def estimate_lag(spans, moving, fps, lo=-1.0, hi=5.0):
    """How many seconds the hands are behind the voice: the shift that best matches 'voice on' with
    'hands moving' over the whole video. Returns (lag_s, match 0..1)."""
    n = len(moving)
    v = np.zeros(n)
    for t0, t1 in spans:
        v[int(t0 * fps):int(t1 * fps)] = 1.0
    m = moving.astype(float)
    v, m = v - v.mean(), m - m.mean()
    lags = list(range(int(lo * fps), int(hi * fps) + 1))
    curve = []
    for L in lags:
        a_, b_ = (v[:n - L], m[L:]) if L >= 0 else (v[-L:], m[:n + L])
        d = np.linalg.norm(a_) * np.linalg.norm(b_)
        curve.append(float((a_ * b_).sum() / d) if d > 0 else 0.0)
    k = int(np.argmax(curve))
    return lags[k] / fps, curve[k]


def find_end(lowered, e_voice, limit, hold):
    """Where the signing ends once the voice has stopped (all values are frames):
    the start of the first stretch of at least `hold` frames with both hands lowered, after the voice end.
    A hand held still in the air is still signing. If the hands never rest, `limit`."""
    if limit <= e_voice:
        return e_voice
    for a, b in runs_from_mask(lowered[e_voice:limit], offset=e_voice):
        if b - a >= hold:
            return a
    return limit


def moving_after(moving, f, hold, cap):
    """How many frames the hands keep moving after frame f (stills shorter than `hold` do not count as a stop)."""
    last, still_run, i = f, 0, f
    while i < min(len(moving), f + cap):
        if moving[i]:
            last, still_run = i + 1, 0
        else:
            still_run += 1
            if still_run >= hold:
                break
        i += 1
    return last - f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--mode", choices=["speech", "hands"], default="speech",
                    help="speech = cut at pauses in the voice (sentence ends); hands = cut at pauses in hand motion")
    # speech mode
    ap.add_argument("--speech_pause", type=float, default=0.8, help="silence (seconds) that ends a sentence")
    ap.add_argument("--min_span", type=float, default=0.8, help="shortest sentence (seconds)")
    ap.add_argument("--lag", type=float, default=0.0,
                    help="seconds the signer is behind the voice (a hint is printed; start with 0)")
    ap.add_argument("--max_extend", type=float, default=3.0,
                    help="seconds a clip may wait after the voice ends for the signer to finish (0 = never wait)")
    ap.add_argument("--rest_hold", type=float, default=0.3,
                    help="seconds both hands must stay down before the sentence counts as finished")
    ap.add_argument("--overlap", action="store_true", help="(already the default, kept so old commands still work)")
    ap.add_argument("--no_overlap", action="store_true",
                    help="never let a clip run into the next sentence's voice (cuts the signer short if he is behind)")
    ap.add_argument("--max_bad", type=float, default=0.2, help="skip a clip when more of its frames than this are graphics")
    # hands mode
    ap.add_argument("--rest_y", type=float, default=1.5,
                    help="a wrist lower than this (in shoulder-widths below the shoulders) = resting")
    ap.add_argument("--gap", type=float, default=0.5, help="seconds the hands must stay still to end a clip")
    ap.add_argument("--still", type=float, default=0.1,
                    help="hand speed below this share of the typical fast speed counts as still")
    ap.add_argument("--max_len", type=float, default=12.0,
                    help="a clip longer than this (seconds) is split again at its longest pause")
    ap.add_argument("--min_pause", type=float, default=0.15, help="shortest pause (s) allowed as a split point")
    ap.add_argument("--min_clip", type=float, default=1.0, help="shortest clip, seconds")
    # both
    ap.add_argument("--min_sign", type=float, default=0.25, help="shortest sign, seconds")
    ap.add_argument("--prominence", type=float, default=0.15,
                    help="how deep a speed dip must be to count as a sign boundary (higher = fewer)")
    ap.add_argument("--pad_start", type=float, default=None, help="seconds added before (speech 0.3, hands 0.4)")
    ap.add_argument("--pad_end", type=float, default=None, help="seconds added after the end (speech 0.6, hands 0.8)")
    ap.add_argument("--fade", type=float, default=0.15, help="audio fade in/out at the clip ends (seconds)")
    ap.add_argument("--reuse", action="store_true", help="reuse out/keypoints/all.npz instead of running MediaPipe")
    ap.add_argument("--no_overlay", action="store_true", help="skip the review videos")
    ap.add_argument("--tol", type=float, default=45, help="color distance that still counts as blue background")
    ap.add_argument("--thr", type=float, default=0.12,
                    help="share of the ring around the signer covered by a graphic before the frame is dropped")
    ap.add_argument("--complexity", type=int, default=0, choices=[0, 1, 2],
                    help="MediaPipe model: 0 = fastest, 2 = most accurate")
    args = ap.parse_args()

    out = Path(args.out)
    for sub in ("clips", "review", "plots", "keypoints"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    frames = sample_frames(args.video)
    if not frames:
        raise SystemExit(f"cannot read {args.video}")
    top, crop_h, crop_w = detect_crop(frames)
    bg = estimate_background(frames, top, crop_h)
    print(f"crop: top={top} height={crop_h} width={crop_w}   background BGR={bg.astype(int)}")

    kp_file = out / "keypoints" / "all.npz"
    if args.reuse and kp_file.exists():
        d = np.load(kp_file)
        pose, left, right, bad, fps = d["pose"], d["left_hand"], d["right_hand"], d["bad"], float(d["fps"])
        print("reusing saved keypoints")
    else:
        print("extracting body, face and finger keypoints (this is the slow part)...")
        pose, left, right, bad, fps = extract(args.video, top, crop_h, bg, args.complexity, args.tol, args.thr)
        np.savez_compressed(kp_file, pose=pose, left_hand=left, right_hand=right, bad=bad, fps=fps)
    n = len(bad)
    print(f"{n} frames at {fps:.1f} fps, {bad.mean() * 100:.1f}% unusable (graphic or no signer)")

    speed, hands = hand_speed(hand_points(pose), bad, fps)

    mode = args.mode
    if mode == "speech":
        audio, sr = load_audio(args.video)
        if len(audio) == 0:
            print("this video has no sound, so using --mode hands instead")
            mode = "hands"
    pad_start = args.pad_start if args.pad_start is not None else (0.3 if mode == "speech" else 0.4)
    pad_end = args.pad_end if args.pad_end is not None else (0.6 if mode == "speech" else 0.8)
    ps, pe = int(pad_start * fps), int(pad_end * fps)

    raised = (hands[:, :2, 1].min(axis=1) < args.rest_y) & ~bad
    still = args.still * np.percentile(speed[~bad], 90) if (~bad).any() else 0.0
    moving = raised & (speed > still)
    lowered = (hands[:, :2, 1].min(axis=1) >= args.rest_y) & ~bad   # both wrists down = rest position
    print(f"hands up in {raised.mean() * 100:.1f}% of frames, both hands down in {lowered.mean() * 100:.1f}%")

    # each window: (sign_start, sign_end, cut_start, cut_end) in frames, plus the voice span in seconds
    windows = []
    if mode == "speech":
        spans, pauses = speech_spans(audio, sr, args.speech_pause, args.min_span)
        print(f"voice: {len(spans)} sentences (silences of {args.speech_pause}s or more end a sentence)")
        lag_hint, match = estimate_lag(spans, moving, fps)
        if match >= 0.2 and abs(lag_hint - args.lag) >= 0.3:
            print(f"hint: the hands follow the voice by about {lag_hint:.1f}s (match {match:.2f}); "
                  f"try --lag {lag_hint:.1f}")
        else:
            print(f"hint: hands and voice line up (delay {lag_hint:.1f}s, match {match:.2f})")
        hold = max(1, int(args.rest_hold * fps))
        for k, (t0, t1) in enumerate(spans):
            s = min(n, max(0, int(round((t0 + args.lag) * fps))))
            e_voice = min(n, max(0, int(round((t1 + args.lag) * fps))))
            next_v = min(n, max(0, int(round((spans[k + 1][0] + args.lag) * fps)))) if k + 1 < len(spans) else n
            # the end: where the voice stops, or later while the hands are still moving
            limit = min(n, e_voice + int(args.max_extend * fps))
            if args.no_overlap:
                limit = min(limit, max(e_voice, next_v - int(0.1 * fps)))
            e = find_end(lowered, e_voice, limit, hold)
            f0 = max(0, int(round((t0 + min(0.0, args.lag)) * fps)) - ps)
            f1 = min(n, e + pe)
            if args.no_overlap:
                f1 = min(f1, max(e, next_v - int(0.05 * fps)))  # never run into the next sentence
            nv = spans[k + 1][0] if k + 1 < len(spans) else None
            windows.append((s, e, f0, f1, t0, t1, nv))
    else:
        print(f"hands raised in {raised.mean() * 100:.1f}% of frames, moving in {moving.mean() * 100:.1f}%")
        min_frames = int(args.min_clip * fps)
        clips = find_clips(moving, bad, int(args.gap * fps), min_frames)
        first_pass = len(clips)
        clips = [c for clip in clips for c in
                 split_long(clip, moving, int(args.max_len * fps), int(args.min_pause * fps), min_frames)]
        print(f"found {first_pass} clips, {len(clips)} after splitting those longer than {args.max_len:.0f}s")
        for s, e in clips:
            windows.append((s, e, max(0, s - ps), min(n, e + pe), None, None, None))

    segments = []
    cut_moving = 0
    for i, (s, e, f0, f1, v0, v1, nv) in enumerate(windows, 1):
        name = f"clip_{i:04d}"
        bad_ratio = float(bad[f0:f1].mean()) if f1 > f0 else 1.0
        seg = {
            "id": name,
            "start_s": round(s / fps, 3),
            "end_s": round(e / fps, 3),
            "duration_s": round((e - s) / fps, 2),
            "voice_start_s": None if v0 is None else round(v0, 3),
            "voice_end_s": None if v1 is None else round(v1, 3),
            "graphic_ratio": round(bad_ratio, 3),
            "cut_start_s": round(f0 / fps, 3),
            "cut_end_s": round(f1 / fps, 3),
            "room_after_voice_s": None if v1 is None else round(f1 / fps - v1, 2),
            "hands_after_end_s": None,
            "status": "ok",
            "estimated_signs": 0,
            "sign_boundaries_s": [],
        }
        if f1 - f0 < 2 or bad_ratio > args.max_bad:
            seg["status"] = "skipped_graphic"
            segments.append(seg)
            continue
        bounds = sign_boundaries(speed, s, e, fps, args.prominence, args.min_sign) if e - s >= 3 else []
        clip_file = out / "clips" / f"{name}.mp4"
        cut_clip(args.video, f0 / fps, f1 / fps, clip_file, top, crop_h, crop_w, args.fade)
        after = moving_after(raised, e, max(1, int(0.2 * fps)), int(4.0 * fps)) / fps  # hands still up after the detected end
        seg["hands_after_end_s"] = round(after, 2)
        cut_moving += after > 0.2
        save_plot(speed, raised, f0, f1, s, e, bounds, fps, out / "plots" / f"{name}.png",
                  voice=None if v0 is None else (v0, v1), next_voice=nv)
        np.savez_compressed(out / "keypoints" / f"{name}.npz", pose=pose[f0:f1], left_hand=left[f0:f1],
                            right_hand=right[f0:f1], bad=bad[f0:f1], fps=fps, first_frame=f0)
        if not args.no_overlay:
            kp = {"pose": pose[f0:f1], "left_hand": left[f0:f1], "right_hand": right[f0:f1]}
            render_overlay(clip_file, out / "review" / f"{name}.mp4", kp, fps,
                           (max(0, s - f0), min(f1 - f0, e - f0)), [b - f0 for b in bounds])
        seg["file_start_s"] = round(f0 / fps, 3)
        seg["estimated_signs"] = len(bounds) + 1
        seg["sign_boundaries_s"] = [round(b / fps, 3) for b in bounds]
        segments.append(seg)

    with open(out / "segments.json", "w", encoding="utf-8") as f:
        json.dump({"video": str(args.video), "fps": fps, "mode": mode, "lag_s": args.lag,
                   "crop": {"top": top, "height": crop_h, "width": crop_w},
                   "clips": segments}, f, indent=2)
    with open(out / "segments.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "start_s", "end_s", "duration_s", "voice_start_s", "voice_end_s",
                    "cut_start_s", "cut_end_s", "room_after_voice_s", "graphic_ratio", "hands_after_end_s", "status"])
        for sg in segments:
            w.writerow([sg["id"], sg["start_s"], sg["end_s"], sg["duration_s"], sg["voice_start_s"],
                        sg["voice_end_s"], sg["cut_start_s"], sg["cut_end_s"], sg["room_after_voice_s"],
                        sg["graphic_ratio"], sg["hands_after_end_s"], sg["status"]])
    ok = sum(1 for sg in segments if sg["status"] == "ok")
    print(f"done: {ok} clips made, {len(segments) - ok} skipped (graphics)")
    print(f"{cut_moving} of {ok} clips end while the hands are still up "
          f"(more than 0.2 s more signing after the end; see hands_after_end_s in segments.csv)")
    print(f"  {out}/clips/ (clean), {out}/review/ (with keypoints), {out}/keypoints/, {out}/segments.csv")


if __name__ == "__main__":
    main()