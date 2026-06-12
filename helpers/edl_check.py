"""Snap and lint an EDL against the Scribe word timeline before rendering.

Hard Rules 6/7 (never cut inside a word, pad every cut edge) are enforced
here as code instead of relying on the editing agent to honor them:

  snap  — move every range edge to the nearest word boundary, then apply
          asymmetric drift padding (lead before the first word, tail after
          the last). Idempotent: re-running converges to the same values.
          The pre-snap EDL is backed up to edl.source.json for audit.
  lint  — validate the EDL (word-interior cuts, inverted ranges, ranges
          beyond source duration, overlay windows outside the output,
          total_duration_s drift). render.py runs this automatically and
          refuses to render on errors (--skip-lint to override).

Sources without a transcript (montage footage, music) skip the word checks.

Usage:
    python helpers/edl_check.py lint <edl.json>
    python helpers/edl_check.py snap <edl.json> [--lead 0.05] [--tail 0.08]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Scribe timestamps drift 50-100ms (Hard Rule 7's 30-200ms working window).
DEFAULT_LEAD = 0.05   # seconds of padding before the first kept word
DEFAULT_TAIL = 0.08   # seconds of padding after the last kept word
MAX_SNAP = 0.5        # edges farther than this from any word are left alone
WORD_INTERIOR_MARGIN = 0.02  # tolerance before an edge counts as inside a word
NEIGHBOR_GAP = 0.005  # minimum clearance kept from a neighboring word
DURATION_TOL = 0.2    # allowed drift for total_duration_s and overlay ends


def words_of(transcript: dict) -> list[dict]:
    """Word-type entries with usable timestamps, in order."""
    return [
        w for w in transcript.get("words", [])
        if w.get("type") == "word"
        and w.get("start") is not None
        and w.get("end") is not None
    ]


def probe_duration(video: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
            capture_output=True, text=True, check=True,
        )
        return float(out.stdout.strip())
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return None


# -------- snap ----------------------------------------------------------------


def _snap_edge(
    t: float,
    words: list[dict],
    *,
    is_start: bool,
    pad: float,
    max_snap: float,
) -> tuple[float, str | None]:
    """Snap one edge to its nearest word boundary and pad it outward.

    Returns (new_time, note) — note is None when the edge is left alone.
    """
    boundary_key = "start" if is_start else "end"
    anchor = min(words, key=lambda w: abs(w[boundary_key] - t))
    boundary = anchor[boundary_key]
    if abs(boundary - t) > max_snap:
        return t, None

    if is_start:
        padded = boundary - pad
        # Do not intrude into the previous word: stay inside the silence gap.
        prev_ends = [w["end"] for w in words if w["end"] <= boundary - 1e-9]
        if prev_ends:
            prev_end = max(prev_ends)
            if padded < prev_end + NEIGHBOR_GAP:
                padded = (prev_end + boundary) / 2
        padded = max(0.0, padded)
    else:
        padded = boundary + pad
        next_starts = [w["start"] for w in words if w["start"] >= boundary + 1e-9]
        if next_starts:
            next_start = min(next_starts)
            if padded > next_start - NEIGHBOR_GAP:
                padded = (boundary + next_start) / 2

    note = None
    if abs(padded - t) > 1e-6:
        note = f"{t:.3f} → {padded:.3f} (word {boundary_key} {boundary:.3f}, pad {pad:.3f})"
    return round(padded, 3), note


def snap_ranges(
    edl: dict,
    words_by_source: dict[str, list[dict] | None],
    source_durations: dict[str, float | None] | None = None,
    lead: float = DEFAULT_LEAD,
    tail: float = DEFAULT_TAIL,
    max_snap: float = MAX_SNAP,
) -> tuple[dict, list[str]]:
    """Return a copy of the EDL with every range edge snapped + padded."""
    out = json.loads(json.dumps(edl))
    log: list[str] = []
    durations = source_durations or {}

    for i, r in enumerate(out.get("ranges", [])):
        src = r["source"]
        words = words_by_source.get(src)
        if not words:
            log.append(f"[{i:02d}] {src}: no transcript — left unchanged")
            continue

        new_start, note_s = _snap_edge(
            float(r["start"]), words, is_start=True, pad=lead, max_snap=max_snap
        )
        new_end, note_e = _snap_edge(
            float(r["end"]), words, is_start=False, pad=tail, max_snap=max_snap
        )
        src_dur = durations.get(src)
        if src_dur is not None:
            new_end = min(new_end, round(src_dur, 3))

        r["start"], r["end"] = new_start, new_end
        if note_s:
            log.append(f"[{i:02d}] {src} start: {note_s}")
        if note_e:
            log.append(f"[{i:02d}] {src} end:   {note_e}")

    if "total_duration_s" in out:
        out["total_duration_s"] = round(
            sum(float(r["end"]) - float(r["start"]) for r in out.get("ranges", [])), 3
        )

    return out, log


# -------- lint ------------------------------------------------------------------


def _inside_word(t: float, words: list[dict]) -> dict | None:
    for w in words:
        if w["start"] + WORD_INTERIOR_MARGIN < t < w["end"] - WORD_INTERIOR_MARGIN:
            return w
    return None


def _edge_padding(t: float, words: list[dict], is_start: bool) -> float | None:
    """Distance from the edge to the word boundary it protects, or None."""
    key = "start" if is_start else "end"
    candidates = (
        [w[key] for w in words if w[key] >= t - 1e-9]
        if is_start
        else [w[key] for w in words if w[key] <= t + 1e-9]
    )
    if not candidates:
        return None
    nearest = min(candidates) if is_start else max(candidates)
    return abs(nearest - t)


def lint_edl(
    edl: dict,
    words_by_source: dict[str, list[dict] | None],
    source_durations: dict[str, float | None],
    edit_dir: Path | None = None,
) -> tuple[list[str], list[str]]:
    """Validate an EDL. Returns (errors, warnings) as printable strings."""
    errors: list[str] = []
    warnings: list[str] = []

    no_transcript_sources: set[str] = set()
    ranges = edl.get("ranges", [])
    for i, r in enumerate(ranges):
        src = r["source"]
        start, end = float(r["start"]), float(r["end"])
        tag = f"range[{i:02d}] {src} {start:.2f}-{end:.2f}"

        if end <= start:
            errors.append(f"{tag}: end <= start")
            continue

        src_dur = source_durations.get(src)
        if src_dur is not None and (end > src_dur + DURATION_TOL or start < 0):
            errors.append(f"{tag}: exceeds source duration ({src_dur:.2f}s)")

        words = words_by_source.get(src)
        if not words:
            no_transcript_sources.add(src)
            continue

        for label, t, is_start in (("start", start, True), ("end", end, False)):
            w = _inside_word(t, words)
            if w:
                errors.append(
                    f"{tag}: {label} cuts inside word "
                    f"'{w.get('text', '?')}' ({w['start']:.2f}-{w['end']:.2f}) — Hard Rule 6"
                )
                continue
            pad = _edge_padding(t, words, is_start)
            if pad is not None and pad < 0.005:
                warnings.append(
                    f"{tag}: {label} sits exactly on the word boundary — "
                    f"no drift padding (Hard Rule 7, want 30-200ms)"
                )

    for src in sorted(no_transcript_sources):
        warnings.append(f"{src}: no transcript — word-boundary checks skipped")

    ranges_total = sum(float(r["end"]) - float(r["start"]) for r in ranges if float(r["end"]) > float(r["start"]))

    declared = edl.get("total_duration_s")
    if declared is not None and abs(float(declared) - ranges_total) > DURATION_TOL:
        errors.append(
            f"total_duration_s={float(declared):.2f} but ranges sum to {ranges_total:.2f} "
            f"(±{DURATION_TOL}s) — update the EDL"
        )

    for j, ov in enumerate(edl.get("overlays") or []):
        t = float(ov.get("start_in_output", -1))
        dur = float(ov.get("duration", 0))
        otag = f"overlay[{j}] {ov.get('file', '?')}"
        if t < 0 or dur <= 0:
            errors.append(f"{otag}: invalid window (start_in_output={t}, duration={dur})")
            continue
        if t + dur > ranges_total + DURATION_TOL:
            errors.append(
                f"{otag}: window {t:.2f}+{dur:.2f}s ends past the output ({ranges_total:.2f}s)"
            )
        if edit_dir is not None:
            ov_path = Path(ov["file"])
            if not ov_path.is_absolute():
                ov_path = (edit_dir / ov_path).resolve()
            if not ov_path.exists():
                errors.append(f"{otag}: file not found ({ov_path})")

    return errors, warnings


# -------- filesystem wiring -----------------------------------------------------


def load_context(edl: dict, edit_dir: Path) -> tuple[dict, dict]:
    """Load per-source words + durations from disk for snap/lint."""
    transcripts_dir = edit_dir / "transcripts"
    words_by_source: dict[str, list[dict] | None] = {}
    durations: dict[str, float | None] = {}
    for src, path in edl.get("sources", {}).items():
        src_path = Path(path)
        if not src_path.is_absolute():
            src_path = (edit_dir / src_path).resolve()
        durations[src] = probe_duration(src_path)
        tr_path = transcripts_dir / f"{src}.json"
        if not tr_path.exists():
            tr_path = transcripts_dir / f"{src_path.stem}.json"
        if tr_path.exists():
            words_by_source[src] = words_of(json.loads(tr_path.read_text()))
        else:
            words_by_source[src] = None
    return words_by_source, durations


def lint_for_render(edl: dict, edit_dir: Path) -> tuple[list[str], list[str]]:
    """Entry point for render.py's pre-render gate."""
    words_by_source, durations = load_context(edl, edit_dir)
    return lint_edl(edl, words_by_source, durations, edit_dir=edit_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Snap/lint an EDL against Scribe word boundaries")
    ap.add_argument("command", choices=["snap", "lint"])
    ap.add_argument("edl", type=Path)
    ap.add_argument("--lead", type=float, default=DEFAULT_LEAD)
    ap.add_argument("--tail", type=float, default=DEFAULT_TAIL)
    ap.add_argument("--max-snap", type=float, default=MAX_SNAP)
    args = ap.parse_args()

    edl_path = args.edl.resolve()
    if not edl_path.exists():
        sys.exit(f"edl not found: {edl_path}")
    edl = json.loads(edl_path.read_text())
    edit_dir = edl_path.parent
    words_by_source, durations = load_context(edl, edit_dir)

    if args.command == "snap":
        snapped, log = snap_ranges(
            edl, words_by_source, durations,
            lead=args.lead, tail=args.tail, max_snap=args.max_snap,
        )
        backup = edl_path.with_suffix(".source.json")
        backup.write_text(json.dumps(edl, indent=2, ensure_ascii=False))
        edl_path.write_text(json.dumps(snapped, indent=2, ensure_ascii=False))
        for line in log:
            print(line)
        print(f"snapped {len(edl.get('ranges', []))} range(s) → {edl_path.name} "
              f"(pre-snap backup: {backup.name})")
        edl = snapped

    errors, warnings = lint_edl(edl, words_by_source, durations, edit_dir=edit_dir)
    for w in warnings:
        print(f"warning: {w}")
    for e in errors:
        print(f"ERROR: {e}")
    if errors:
        sys.exit(f"lint: {len(errors)} error(s)")
    print("lint: OK")


if __name__ == "__main__":
    main()
