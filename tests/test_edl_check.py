import copy

import pytest

from edl_check import lint_edl, snap_ranges


def W(text, start, end):
    return {"type": "word", "text": text, "start": start, "end": end}


WORDS = [
    W("안녕", 1.00, 1.40),
    W("하세요", 1.50, 2.00),
    W("여러분", 2.60, 3.00),
    W("오늘은", 3.10, 3.60),
]


def make_edl(ranges, overlays=None, total=None):
    edl = {
        "version": 1,
        "sources": {"C01": "/abs/C01.mp4"},
        "ranges": ranges,
    }
    if overlays is not None:
        edl["overlays"] = overlays
    if total is not None:
        edl["total_duration_s"] = total
    return edl


# -------- snap_ranges --------------------------------------------------------


def test_snap_moves_edges_to_word_boundaries_with_padding():
    edl = make_edl([{"source": "C01", "start": 1.1, "end": 2.9}])
    snapped, log = snap_ranges(edl, {"C01": WORDS}, lead=0.05, tail=0.08)
    r = snapped["ranges"][0]
    # start snaps to 안녕.start (1.00) minus lead
    assert r["start"] == pytest.approx(0.95, abs=1e-6)
    # end snaps to 여러분.end (3.00) plus tail
    assert r["end"] == pytest.approx(3.08, abs=1e-6)
    assert log  # adjustments reported


def test_snap_is_idempotent():
    edl = make_edl([{"source": "C01", "start": 1.1, "end": 2.9}])
    once, _ = snap_ranges(edl, {"C01": WORDS})
    twice, _ = snap_ranges(copy.deepcopy(once), {"C01": WORDS})
    assert once["ranges"] == twice["ranges"]


def test_snap_pad_does_not_intrude_into_neighbor_word():
    # 하세요 ends 2.00, 여러분 starts 2.60 → gap 0.6, fine.
    # But 안녕 ends 1.40, 하세요 starts 1.50 → gap 0.1 < default lead+margin.
    edl = make_edl([{"source": "C01", "start": 1.52, "end": 2.0}])
    snapped, _ = snap_ranges(edl, {"C01": WORDS}, lead=0.2, tail=0.08)
    r = snapped["ranges"][0]
    # lead pad of 0.2 would land at 1.30, inside 안녕 (1.00-1.40).
    # Must stay inside the gap between 안녕.end and 하세요.start.
    assert 1.40 <= r["start"] < 1.50


def test_snap_leaves_far_edges_alone():
    # start 10s away from any word — montage section, do not yank it to speech
    edl = make_edl([{"source": "C01", "start": 13.0, "end": 14.0}])
    snapped, log = snap_ranges(edl, {"C01": WORDS}, max_snap=0.5)
    r = snapped["ranges"][0]
    assert r["start"] == 13.0 and r["end"] == 14.0


def test_snap_skips_sources_without_transcript():
    edl = make_edl([{"source": "C01", "start": 1.1, "end": 2.9}])
    snapped, _ = snap_ranges(edl, {"C01": None})
    assert snapped["ranges"][0]["start"] == 1.1


def test_snap_recomputes_total_duration():
    edl = make_edl([{"source": "C01", "start": 1.1, "end": 2.9}], total=1.8)
    snapped, _ = snap_ranges(edl, {"C01": WORDS})
    # ranges became 0.95-3.08 → total must follow, keeping the EDL lint-clean
    assert snapped["total_duration_s"] == pytest.approx(2.13, abs=1e-6)


# -------- lint_edl -----------------------------------------------------------


def test_lint_flags_cut_inside_word():
    edl = make_edl([{"source": "C01", "start": 1.2, "end": 2.8}])  # 1.2 inside 안녕
    errors, _ = lint_edl(edl, {"C01": WORDS}, {"C01": 60.0})
    assert any("inside word" in e for e in errors)


def test_lint_accepts_padded_edges():
    edl = make_edl(
        [{"source": "C01", "start": 0.95, "end": 3.08}], total=2.13
    )
    errors, _ = lint_edl(edl, {"C01": WORDS}, {"C01": 60.0})
    assert errors == []


def test_lint_flags_inverted_range():
    edl = make_edl([{"source": "C01", "start": 2.0, "end": 1.0}])
    errors, _ = lint_edl(edl, {"C01": WORDS}, {"C01": 60.0})
    assert any("end <= start" in e for e in errors)


def test_lint_flags_range_beyond_source_duration():
    edl = make_edl([{"source": "C01", "start": 55.0, "end": 65.0}])
    errors, _ = lint_edl(edl, {"C01": None}, {"C01": 60.0})
    assert any("exceeds source duration" in e for e in errors)


def test_lint_flags_unknown_source():
    # a typo'd source must die in lint, not as a KeyError inside render
    edl = make_edl([{"source": "C99", "start": 1.0, "end": 2.0}])
    errors, _ = lint_edl(edl, {"C01": WORDS}, {"C01": 60.0})
    assert any("unknown source" in e for e in errors)


def test_lint_flags_negative_start_even_without_probed_duration():
    edl = make_edl([{"source": "C01", "start": -0.5, "end": 2.0}])
    errors, _ = lint_edl(edl, {"C01": None}, {"C01": None})  # ffprobe failed
    assert any("start < 0" in e for e in errors)


def test_lint_flags_total_duration_mismatch():
    edl = make_edl([{"source": "C01", "start": 0.95, "end": 3.08}], total=10.0)
    errors, _ = lint_edl(edl, {"C01": WORDS}, {"C01": 60.0})
    assert any("total_duration_s" in e for e in errors)


def test_lint_flags_overlay_outside_output():
    edl = make_edl(
        [{"source": "C01", "start": 0.95, "end": 3.08}],
        overlays=[{"file": "x.mp4", "start_in_output": 2.0, "duration": 5.0}],
        total=2.13,
    )
    errors, _ = lint_edl(edl, {"C01": WORDS}, {"C01": 60.0})
    assert any("overlay" in e for e in errors)


def test_lint_missing_transcript_skips_word_checks():
    edl = make_edl([{"source": "C01", "start": 1.2, "end": 2.8}], total=1.6)
    errors, warnings = lint_edl(edl, {"C01": None}, {"C01": 60.0})
    assert errors == []
    assert any("no transcript" in w for w in warnings)


def test_lint_warns_on_unpadded_edge():
    # start exactly on the word boundary — violates Hard Rule 7 padding
    edl = make_edl([{"source": "C01", "start": 1.0, "end": 3.08}], total=2.08)
    errors, warnings = lint_edl(edl, {"C01": WORDS}, {"C01": 60.0})
    assert errors == []
    assert any("padding" in w for w in warnings)
