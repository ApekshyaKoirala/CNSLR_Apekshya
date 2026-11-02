"""
parse_vtt.py -- turn a YouTube auto-caption .vtt (word-level timestamps) into sentences.csv
for prepare_clips.py.

YouTube's rolling captions repeat every line, so only the line that carries word timestamps
is used. Auto-captions have little punctuation, so a segment ends at a danda (।), ? or !,
at a pause between words longer than --gap, or (if still too long) at its biggest pause.

Usage:
  python parse_vtt.py --vtt test_captions_ne-orig.vtt --video test_video.mp4 --out sentences.csv
"""
import argparse
import csv
import re

TIME = r"\d\d:\d\d:\d\d\.\d+"
CUE = re.compile(rf"({TIME}) --> ({TIME})")
TAG_TIME = re.compile(rf"<({TIME})>")
END_MARKS = ("।", "?", "!")


def to_sec(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def read_words(path):
    """Returns [(start_s, word)] and the last cue end."""
    with open(path, encoding="utf-8") as f:
        blocks = f.read().split("\n\n")
    words, last_end = [], 0.0
    for block in blocks:
        lines = block.strip("\n").split("\n")
        m = CUE.search(lines[0]) if lines else None
        if not m:
            continue
        cue_start, cue_end = to_sec(m.group(1)), to_sec(m.group(2))
        last_end = max(last_end, cue_end)
        timed = [l for l in lines[1:] if "<c>" in l or TAG_TIME.search(l)]
        if not timed:
            continue  # untimed duplicate cue
        parts = TAG_TIME.split(timed[-1])  # text0, time1, text1, time2, text2, ...
        t = cue_start
        for i, part in enumerate(parts):
            if i % 2 == 1:
                t = to_sec(part)
                continue
            clean = re.sub(r"</?c>", "", part)
            for w in clean.split():
                words.append((t, w))
    # drop exact repeats of the same word at the same time
    seen, uniq = set(), []
    for t, w in words:
        if (t, w) not in seen:
            seen.add((t, w))
            uniq.append((t, w))
    return uniq, last_end


def split_at_gaps(seg, gap):
    out, cur = [], [seg[0]]
    for prev, item in zip(seg, seg[1:]):
        if item[0] - prev[0] > gap:
            out.append(cur)
            cur = []
        cur.append(item)
    out.append(cur)
    return out


def split_long(seg, max_len):
    """Split at the biggest pause until every piece is at most max_len seconds."""
    if len(seg) < 2 or seg[-1][0] - seg[0][0] <= max_len:
        return [seg]
    k = max(range(1, len(seg)), key=lambda j: seg[j][0] - seg[j - 1][0])
    return split_long(seg[:k], max_len) + split_long(seg[k:], max_len)


def build_sentences(words, last_end, gap, max_len, max_tail=1.0):
    segments, cur = [], []
    for i, (t, w) in enumerate(words):
        if re.fullmatch(r"\[.*\]", w):  # [संगीत], [Music]
            continue
        cur.append((t, w, i))
        if w.endswith(END_MARKS):
            segments.append(cur)
            cur = []
    if cur:
        segments.append(cur)

    pieces = []
    for seg in segments:
        for part in split_at_gaps(seg, gap):
            pieces += split_long(part, max_len)

    rows = []
    for s in pieces:
        start = s[0][0]
        last_t, _, last_i = s[-1]
        nxt = words[last_i + 1][0] if last_i + 1 < len(words) else last_end
        end = min(nxt, last_t + max_tail)
        rows.append((round(start, 3), round(end, 3), " ".join(w for _, w, _ in s)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vtt", required=True)
    ap.add_argument("--video", required=True, help="video file name, written into the CSV")
    ap.add_argument("--out", default="sentences.csv")
    ap.add_argument("--gap", type=float, default=1.5, help="pause (s) between words that starts a new segment")
    ap.add_argument("--max_len", type=float, default=12.0, help="longest segment (s) before forced split")
    args = ap.parse_args()

    words, last_end = read_words(args.vtt)
    rows = build_sentences(words, last_end, args.gap, args.max_len)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "start", "end", "text"])
        for start, end, text in rows:
            w.writerow([args.video, start, end, text])

    lens = [e - s for s, e, _ in rows]
    print(f"{len(words)} words -> {len(rows)} segments")
    if lens:
        print(f"length: min {min(lens):.1f}s, median {sorted(lens)[len(lens) // 2]:.1f}s, max {max(lens):.1f}s")
        print(f"over 15s: {sum(l > 15 for l in lens)}, under 1s: {sum(l < 1 for l in lens)}")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()