from verify_output import check_duration, scan_banned


def W(text, start, end):
    return {"type": "word", "text": text, "start": start, "end": end}


WORDS = [
    W("오늘", 0.0, 0.4),
    W("라이브", 0.5, 1.0),
    W("시작합니다", 1.1, 1.8),
    W("채팅", 5.0, 5.4),
    W("히스토리를", 5.5, 6.0),
    W("보면", 6.1, 6.4),
    W("채팅으로", 9.0, 9.6),
    W("알려주세요", 9.7, 10.4),
]

CFG = {
    "banned": ["라이브", "채팅", "구독"],
    "whitelist_contexts": ["채팅 히스토리"],
}


def test_scan_finds_banned_terms_with_timestamps():
    findings = scan_banned(WORDS, CFG)
    terms = {(f["term"], round(f["time"], 1)) for f in findings}
    assert ("라이브", 0.5) in terms
    assert ("채팅", 9.0) in terms  # the bare 채팅으로 usage IS flagged


def test_scan_whitelist_suppresses_product_ui_terms():
    findings = scan_banned(WORDS, CFG)
    # 채팅 inside "채팅 히스토리" (5.0s) must be suppressed
    assert not any(f["term"] == "채팅" and abs(f["time"] - 5.0) < 0.2 for f in findings)


def test_scan_clean_transcript_returns_nothing():
    findings = scan_banned([W("안녕하세요", 0.0, 1.0)], CFG)
    assert findings == []


def test_check_duration_within_tolerance_passes():
    edl = {"ranges": [{"source": "C01", "start": 0.0, "end": 10.0}], "total_duration_s": 10.0}
    errors, _ = check_duration(10.1, edl, tol=0.2)
    assert errors == []


def test_check_duration_mismatch_fails():
    edl = {"ranges": [{"source": "C01", "start": 0.0, "end": 10.0}], "total_duration_s": 10.0}
    errors, _ = check_duration(12.5, edl, tol=0.2)
    assert errors


def test_check_duration_tolerance_scales_with_segment_count():
    # fps resample + AAC padding drift ~0.07s per segment; a 30-segment cut
    # may legitimately run ~1.5s long without any dropped segment.
    ranges = [{"source": "C01", "start": i * 10.0, "end": i * 10.0 + 10.0} for i in range(30)]
    edl = {"ranges": ranges, "total_duration_s": 300.0}
    errors, _ = check_duration(301.5, edl)
    assert errors == []
    # ...but the same absolute drift on a single segment is a real failure
    edl_one = {"ranges": ranges[:1], "total_duration_s": 10.0}
    errors, _ = check_duration(11.5, edl_one)
    assert errors
