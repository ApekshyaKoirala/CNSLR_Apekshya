"""
debug_view.py -- see what the detector sees, before trusting the clips.

Writes debug/debug.mp4:
  - green outline = the signer's silhouette (red outline when a graphic or no signer is found)
  - red tint = anything that is not blue background and not the signer (a graphic anywhere in the frame)
  - state GRAPHIC only when a graphic TOUCHES the signer (near = share of a ring around him that is covered);
    a picture in a corner away from the signer stays OK
  - skeleton on the body, eyes and mouth points, and all finger points on both hands
  - label with time, state (OK / GRAPHIC / NO SIGNER) and the graphic percentage
Also writes debug/ranges.csv with the time ranges that are not OK.

With --step 1 the video plays at normal speed and gets the sound (debug_sound.mp4).

Examples:
  python debug_view.py --video test_video.mp4 --out debug/
  python debug_view.py --video test_video.mp4 --out debug/ --start 100 --seconds 60 --step 1

Keep segment_video.py in the same folder (this file imports from it).
"""
import argparse
import csv
import subprocess
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from segment_video import detect_crop, estimate_background, graphic_score, make_model, sample_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="debug")
    ap.add_argument("--step", type=int, default=5, help="use every Nth frame (1 = normal speed, with sound)")
    ap.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    ap.add_argument("--seconds", type=float, default=0.0, help="how many seconds to check (0 = to the end)")
    ap.add_argument("--complexity", type=int, default=0, choices=[0, 1, 2], help="MediaPipe model: 0 = fastest")
    ap.add_argument("--tol", type=float, default=45, help="color distance that still counts as background")
    ap.add_argument("--thr", type=float, default=0.12,
                    help="share of the ring around the signer covered by a graphic before the frame counts as GRAPHIC")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frames = sample_frames(args.video)
    if not frames:
        raise SystemExit(f"cannot read {args.video}")
    top, crop_h, crop_w = detect_crop(frames)
    bg = estimate_background(frames, top, crop_h)
    print(f"crop: top={top} height={crop_h} width={crop_w}   background BGR={bg.astype(int)}")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    first = int(args.start * fps)
    last = int((args.start + args.seconds) * fps) if args.seconds > 0 else 10 ** 9
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    model = make_model(args.complexity, static=True)
    draw = mp.solutions.drawing_utils

    silent = out / "debug.mp4"
    writer, states, scores, thumbs, i = None, [], [], [], first
    while i < last:
        if (i - first) % args.step:
            if not cap.grab():
                break
            i += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        frame = frame[top:top + crop_h, :crop_w].copy()
        h, w = frame.shape[:2]

        res = model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            score, outside, sil = graphic_score(frame, res, bg, args.tol)
            state = "GRAPHIC" if score > args.thr else "OK"
        else:
            score, outside, sil, state = 0.0, None, None, "NO SIGNER"
        color = (0, 200, 0) if state == "OK" else (0, 0, 255)

        if outside is not None:
            tint = cv2.resize(outside.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            frame[tint] = (0.5 * frame[tint] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
            big = cv2.resize(sil.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, color, 2)
        if res.pose_landmarks:
            draw.draw_landmarks(frame, res.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)
        for hand in (res.left_hand_landmarks, res.right_hand_landmarks):
            if hand:
                draw.draw_landmarks(frame, hand, mp.solutions.holistic.HAND_CONNECTIONS)
        scale = max(0.5, w / 1000)
        cv2.putText(frame, f"{i / fps:7.1f}s  {state}  near={score * 100:.0f}%", (10, int(30 * scale) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

        if writer is None:
            writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"),
                                     max(1.0, fps / args.step), (w, h))
        writer.write(frame)
        states.append((i / fps, state))
        scores.append(score)
        if len(states) % 10 == 1:
            thumbs.append(cv2.resize(frame, (400, max(1, int(400 * h / w)))))
        i += 1
        if len(states) % 200 == 0:
            print(f"  {i / fps:.0f}s done")
    cap.release()
    if writer:
        writer.release()

    # add sound when playing at normal speed
    final = silent
    if writer and args.step == 1:
        final = out / "debug_sound.mp4"
        cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(silent), "-ss", f"{args.start:.3f}"]
        if args.seconds > 0:
            cmd += ["-t", f"{args.seconds:.3f}"]
        cmd += ["-i", args.video, "-map", "0:v", "-map", "1:a:0?", "-c:v", "libx264", "-crf", "20",
                "-c:a", "aac", "-shortest", str(final)]
        subprocess.run(cmd, check=True)

    # time ranges that are not OK
    ranges, cur = [], None
    for t, s in states:
        if s != "OK":
            if cur and cur[2] == s:
                cur[1] = t + args.step / fps
            else:
                cur = [t, t + args.step / fps, s]
                ranges.append(cur)
        else:
            cur = None
    with open(out / "ranges.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["start_s", "end_s", "state"])
        for a, b, s in ranges:
            wr.writerow([round(a, 1), round(b, 1), s])
            print(f"  {a:7.1f}s to {b:7.1f}s  {s}")

    if thumbs:  # contact sheet: up to 6 frames spread over the checked part
        pick = [thumbs[int(k)] for k in np.linspace(0, len(thumbs) - 1, min(6, len(thumbs)))]
        while len(pick) % 2:
            pick.append(np.zeros_like(pick[0]))
        rows = [np.hstack(pick[k:k + 2]) for k in range(0, len(pick), 2)]
        cv2.imwrite(str(out / "sample_frames.png"), np.vstack(rows))
        print(f"sample frames -> {out / 'sample_frames.png'}")
    if scores:
        print(f"near-signer graphic: median {np.median(scores) * 100:.1f}%, 90th percentile {np.percentile(scores, 90) * 100:.1f}%  (threshold {args.thr * 100:.0f}%)")
    n = max(1, len(states))
    for s in ("OK", "GRAPHIC", "NO SIGNER"):
        print(f"{s}: {sum(1 for _, x in states if x == s) / n * 100:.1f}% of checked frames")
    print(f"done -> {final}, {out / 'ranges.csv'}")


if __name__ == "__main__":
    main()