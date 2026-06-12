"""Apply the channel's permanent ASR correction table to Scribe transcripts.

Korean ASR repeatedly mangles tech terms (클로드 → Claude, 미소스 파이브 →
Mythos 5). Fixing them inline per-session loses the fix forever; this module
applies `helpers/corrections.json` so every fix is permanent and shared by
subtitles, packed transcripts, and banned-word scans alike.

Two layers, applied in order:

  phrase_merges       — consecutive word tokens fullmatching a regex list are
                        merged into one token. The LAST pattern may capture a
                        suffix; backrefs in the replacement expand against it
                        (e.g. ["미소스", "파이브(.*)"] + "Mythos 5\\1" keeps
                        Korean particles: 파이브라고 → Mythos 5라고).
                        Use this layer for terms that are real Korean words in
                        other contexts (리소스, 테이블) — context protects them.
  regex_replacements  — re.sub per word token. Particles survive because the
                        replacement is a substring (클로드가 → Claude가).
                        Only put unambiguous strings here.

Idempotent: the original token text is preserved in `raw_text` and every
re-application recomputes from it, so cached transcripts can be re-corrected
whenever the table grows (transcribe.py does this on cache hits).

Usage:
    python helpers/corrections.py apply <transcript.json> [...]
    python helpers/corrections.py apply <transcript.json> --table custom.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_TABLE_PATH = Path(__file__).resolve().parent / "corrections.json"


def load_table(path: Path | None = None) -> dict:
    p = path or DEFAULT_TABLE_PATH
    if not p.exists():
        return {"regex_replacements": [], "phrase_merges": []}
    return json.loads(p.read_text())


def _base_text(word: dict) -> str:
    return word.get("raw_text", word.get("text") or "")


def _try_merge(words: list[dict], start_idx: int, rule: dict) -> dict | None:
    """Match `rule["match"]` against consecutive word tokens from start_idx."""
    patterns = rule["match"]
    matched: list[dict] = []
    idx = start_idx
    last_match: re.Match | None = None
    for pat in patterns:
        while idx < len(words) and words[idx].get("type") != "word":
            idx += 1
        if idx >= len(words):
            return None
        token = words[idx]
        if token.get("merged"):
            return None
        m = re.fullmatch(pat, _base_text(token))
        if not m:
            return None
        matched.append(token)
        last_match = m
        idx += 1

    text = last_match.expand(rule["replacement"]) if last_match else rule["replacement"]
    merged = dict(matched[0])
    merged.update(
        text=text,
        start=matched[0]["start"],
        end=matched[-1]["end"],
        raw_text=" ".join(_base_text(t) for t in matched),
        merged=True,
    )
    merged["_consumed"] = [id(t) for t in matched]
    return merged


def apply_to_words(words: list[dict], table: dict) -> tuple[list[dict], int]:
    """Return (corrected word list, number of tokens changed this pass)."""
    changes = 0

    # Layer 1: phrase merges (context-protected multi-token terms)
    out: list[dict] = []
    i = 0
    while i < len(words):
        w = words[i]
        if w.get("type") != "word" or w.get("merged"):
            out.append(w)
            i += 1
            continue
        merged = None
        for rule in table.get("phrase_merges", []):
            merged = _try_merge(words, i, rule)
            if merged:
                break
        if merged:
            # skip everything (incl. interleaved spacing) up to the last consumed token
            j = i
            remaining = set(merged.pop("_consumed"))
            while j < len(words) and remaining:
                remaining.discard(id(words[j]))
                j += 1
            out.append(merged)
            changes += 1
            i = j
        else:
            out.append(w)
            i += 1

    # Layer 2: per-token regex substitutions (recomputed from raw_text → idempotent)
    for w in out:
        if w.get("type") != "word" or w.get("merged"):
            continue
        base = _base_text(w)
        text = base
        for rule in table.get("regex_replacements", []):
            text = re.sub(rule["pattern"], rule["replacement"], text)
        if text != w.get("text"):
            if "raw_text" not in w:
                w["raw_text"] = w.get("text") or ""
            w["text"] = text
            changes += 1

    return out, changes


def apply_to_transcript(transcript: dict, table: dict) -> tuple[dict, int]:
    words = transcript.get("words")
    if not isinstance(words, list):
        return transcript, 0
    new_words, n = apply_to_words(words, table)
    transcript["words"] = new_words
    return transcript, n


def apply_file(path: Path, table_path: Path | None = None) -> int:
    """Correct one transcript JSON in place. Returns tokens changed."""
    table = load_table(table_path)
    transcript = json.loads(path.read_text())
    transcript, n = apply_to_transcript(transcript, table)
    if n:
        path.write_text(json.dumps(transcript, indent=2, ensure_ascii=False))
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply the ASR correction table to transcripts")
    ap.add_argument("command", choices=["apply"])
    ap.add_argument("transcripts", nargs="+", type=Path)
    ap.add_argument("--table", type=Path, default=None)
    args = ap.parse_args()

    total = 0
    for path in args.transcripts:
        if not path.exists():
            sys.exit(f"transcript not found: {path}")
        n = apply_file(path, args.table)
        total += n
        print(f"{path.name}: {n} fix(es)")
    print(f"done: {total} fix(es) across {len(args.transcripts)} file(s)")


if __name__ == "__main__":
    main()
