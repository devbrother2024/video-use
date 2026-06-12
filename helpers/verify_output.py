"""Machine-verify a rendered output against its EDL — delivery gate.

Turns the manual post-render checklist into one command:

  1. Duration: ffprobe output length vs the EDL's range sum and
     total_duration_s (±0.2s) — catches dropped/duplicated segments.
  2. Stream profile: reports width/height/fps/pix_fmt; if the EDL carries an
     `output` profile ({width,height,fps}), mismatches are errors.
  3. --banned-words: re-transcribes the FINAL output (fresh Scribe pass,
     fingerprint-cached per rendered file) and scans for forbidden terms —
     live-stream chatter that must not survive into client deliveries
     (라이브/구독/댓글/...). Whitelist contexts suppress product-UI phrases
     like "채팅 히스토리". See helpers/examples/banned_words_delivery_kr.json.

Writes a markdown report (default <edit>/verify/report.md) usable as the
pre-delivery PASS evidence. Exit code 0 = PASS, 1 = FAIL.

Usage:
    python helpers/verify_output.py <final.mp4> --edl <edl.json>
    python helpers/verify_output.py <final.mp4> --edl <edl.json> \
        --banned-words helpers/examples/banned_words_delivery_kr.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

DURATION_TOL = 0.2       # arithmetic tolerance (EDL-internal consistency)
PER_SEGMENT_TOL = 0.08   # encoding drift allowance per concatenated segment


# -------- probes ---------------------------------------------------------------


def probe_streams(video: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=codec_type,codec_name,width,height,avg_frame_rate,pix_fmt,sample_rate,channels",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.stdout)
    info: dict = {"duration": float(data["format"]["duration"])}
    for s in data.get("streams", []):
        if s.get("codec_type") == "video" and "width" not in info:
            num, _, den = (s.get("avg_frame_rate") or "0/1").partition("/")
            fps = float(num) / float(den) if float(den or 1) else 0.0
            info.update(
                width=s.get("width"), height=s.get("height"),
                fps=round(fps, 3), pix_fmt=s.get("pix_fmt"),
                vcodec=s.get("codec_name"),
            )
        elif s.get("codec_type") == "audio" and "sample_rate" not in info:
            info.update(
                sample_rate=s.get("sample_rate"), channels=s.get("channels"),
                acodec=s.get("codec_name"),
            )
    return info


# -------- checks (pure) ----------------------------------------------------------


def check_duration(
    probed_duration: float,
    edl: dict,
    tol: float = DURATION_TOL,
    per_segment_tol: float = PER_SEGMENT_TOL,
) -> tuple[list[str], list[str]]:
    """Compare output duration against the EDL. Returns (errors, notes).

    Encoding drift (fps resample + AAC frame padding) adds up to ~0.07s per
    concatenated segment, so the output tolerance scales with segment count —
    a fixed ±0.2s would false-FAIL any long multi-segment cut.
    """
    errors: list[str] = []
    notes: list[str] = []

    ranges = edl.get("ranges", [])
    ranges_total = sum(float(r["end"]) - float(r["start"]) for r in ranges)
    enc_tol = max(tol, per_segment_tol * len(ranges))
    notes.append(
        f"output {probed_duration:.2f}s vs ranges sum {ranges_total:.2f}s "
        f"(±{enc_tol:.2f}s for {len(ranges)} segment(s))"
    )
    if abs(probed_duration - ranges_total) > enc_tol:
        errors.append(
            f"duration mismatch: output {probed_duration:.2f}s, "
            f"EDL ranges sum {ranges_total:.2f}s (±{enc_tol:.2f}s) — dropped or duplicated segment?"
        )

    declared = edl.get("total_duration_s")
    if declared is not None and abs(float(declared) - ranges_total) > tol:
        errors.append(
            f"EDL inconsistent: total_duration_s={float(declared):.2f} but ranges sum to {ranges_total:.2f}"
        )
    return errors, notes


def check_profile(info: dict, edl: dict) -> tuple[list[str], list[str]]:
    """If the EDL declares an output profile, enforce it. Returns (errors, notes)."""
    errors: list[str] = []
    notes = [
        f"video {info.get('width')}x{info.get('height')}@{info.get('fps')} "
        f"{info.get('vcodec')}/{info.get('pix_fmt')}, "
        f"audio {info.get('acodec')} {info.get('sample_rate')}Hz ch={info.get('channels')}"
    ]
    profile = edl.get("output")
    if not isinstance(profile, dict):
        return errors, notes
    for key, probed in (("width", info.get("width")), ("height", info.get("height"))):
        want = profile.get(key)
        if want is not None and probed != want:
            errors.append(f"profile mismatch: {key} {probed} != requested {want}")
    want_fps = profile.get("fps")
    if want_fps is not None and abs(float(info.get("fps", 0)) - float(want_fps)) > 0.06:
        errors.append(f"profile mismatch: fps {info.get('fps')} != requested {want_fps}")
    return errors, notes


def _doc_from_words(words: list[dict]) -> tuple[str, list[tuple[int, int, int]]]:
    """Concatenate word tokens into one string; map char spans → word index."""
    parts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    pos = 0
    for i, w in enumerate(words):
        if w.get("type") != "word":
            continue
        text = (w.get("text") or "").strip()
        if not text:
            continue
        if parts:
            parts.append(" ")
            pos += 1
        spans.append((pos, pos + len(text), i))
        parts.append(text)
        pos += len(text)
    return "".join(parts), spans


def scan_banned(words: list[dict], cfg: dict) -> list[dict]:
    """Find banned terms in the transcript. Whitelist contexts suppress hits."""
    doc, spans = _doc_from_words(words)

    whitelist_spans: list[tuple[int, int]] = []
    for phrase in cfg.get("whitelist_contexts", []):
        start = 0
        while (idx := doc.find(phrase, start)) != -1:
            whitelist_spans.append((idx, idx + len(phrase)))
            start = idx + 1

    def word_at(char_pos: int) -> dict | None:
        for a, b, i in spans:
            if a <= char_pos < b:
                return words[i]
        return None

    findings: list[dict] = []
    for term in cfg.get("banned", []):
        start = 0
        while (idx := doc.find(term, start)) != -1:
            end = idx + len(term)
            start = idx + 1
            if any(a <= idx and end <= b for a, b in whitelist_spans):
                continue
            w = word_at(idx)
            findings.append({
                "term": term,
                "time": float(w["start"]) if w and w.get("start") is not None else -1.0,
                "context": doc[max(0, idx - 15): min(len(doc), end + 15)],
            })
    findings.sort(key=lambda f: f["time"])
    return findings


# -------- banned-word scan wiring -------------------------------------------------


def transcribe_output(video: Path, verify_dir: Path) -> list[dict]:
    """Fresh word-level transcript of the rendered output (fingerprint-cached).

    transcribe.py caches by stem, but a re-rendered final.mp4 keeps its name —
    so the cache is invalidated whenever the file's size/mtime changes.
    """
    import transcribe  # same directory

    transcripts_dir = verify_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    st = video.stat()
    fingerprint = f"{st.st_size}-{st.st_mtime_ns}"
    fp_path = transcripts_dir / f"{video.stem}.fingerprint"
    tr_path = transcripts_dir / f"{video.stem}.json"
    if tr_path.exists() and (not fp_path.exists() or fp_path.read_text() != fingerprint):
        tr_path.unlink()

    transcribe.transcribe_one(
        video=video, edit_dir=verify_dir, api_key=transcribe.load_api_key()
    )
    fp_path.write_text(fingerprint)
    return json.loads(tr_path.read_text()).get("words", [])


# -------- report -------------------------------------------------------------------


def write_report(
    report_path: Path,
    video: Path,
    errors: list[str],
    notes: list[str],
    findings: list[dict] | None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    verdict = "PASS" if not errors and not findings else "FAIL"
    lines = [
        f"# verify report — {video.name}",
        "",
        f"- date: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- verdict: **{verdict}**",
        "",
        "## checks",
        "",
    ]
    lines += [f"- {n}" for n in notes]
    if errors:
        lines += ["", "## errors", ""] + [f"- ❌ {e}" for e in errors]
    if findings is not None:
        lines += ["", "## banned-word scan", ""]
        if findings:
            lines += [
                f"- ❌ `{f['term']}` at {f['time']:.1f}s — …{f['context']}…"
                for f in findings
            ]
        else:
            lines += ["- ✅ no banned terms found"]
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify a rendered output against its EDL")
    ap.add_argument("video", type=Path)
    ap.add_argument("--edl", type=Path, required=True)
    ap.add_argument("--banned-words", type=Path, default=None,
                    help="JSON with {banned:[], whitelist_contexts:[]} — re-transcribes the output")
    ap.add_argument("--report", type=Path, default=None,
                    help="Report path (default <edl_dir>/verify/report.md)")
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")
    edl_path = args.edl.resolve()
    if not edl_path.exists():
        sys.exit(f"edl not found: {edl_path}")
    edl = json.loads(edl_path.read_text())
    edit_dir = edl_path.parent
    verify_dir = edit_dir / "verify"
    report_path = args.report or (verify_dir / "report.md")

    info = probe_streams(video)
    errors, notes = check_duration(info["duration"], edl)
    p_errors, p_notes = check_profile(info, edl)
    errors += p_errors
    notes += p_notes

    findings: list[dict] | None = None
    if args.banned_words:
        cfg_path = args.banned_words.resolve()
        if not cfg_path.exists():
            sys.exit(f"banned-words config not found: {cfg_path}")
        cfg = json.loads(cfg_path.read_text())
        print("banned-word scan: re-transcribing the rendered output…")
        words = transcribe_output(video, verify_dir)
        findings = scan_banned(words, cfg)

    write_report(report_path, video, errors, notes, findings)

    for n in notes:
        print(f"  {n}")
    for e in errors:
        print(f"ERROR: {e}")
    for f in findings or []:
        print(f"BANNED: '{f['term']}' at {f['time']:.1f}s — …{f['context']}…")
    print(f"report → {report_path}")
    if errors or findings:
        sys.exit(f"verify: FAIL ({len(errors)} error(s), {len(findings or [])} banned hit(s))")
    print("verify: PASS")


if __name__ == "__main__":
    main()
