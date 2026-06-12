import copy

import pytest

from corrections import apply_to_words


def W(text, start, end):
    return {"type": "word", "text": text, "start": start, "end": end}


def SP():
    return {"type": "spacing", "text": " "}


TABLE = {
    "version": 1,
    "regex_replacements": [
        {"pattern": "클로드", "replacement": "Claude"},
        {"pattern": "엔트로픽|앤트로픽", "replacement": "Anthropic"},
    ],
    "phrase_merges": [
        {
            "match": ["미소스|밋소스|리소스", "파이브(.*)"],
            "replacement": "Mythos 5\\1",
        }
    ],
}


def test_regex_preserves_korean_suffix():
    words = [W("클로드가", 0.0, 0.5)]
    out, n = apply_to_words(words, TABLE)
    assert n == 1
    assert out[0]["text"] == "Claude가"
    assert out[0]["raw_text"] == "클로드가"


def test_apply_is_idempotent():
    words = [W("클로드가", 0.0, 0.5), SP(), W("엔트로픽", 0.6, 1.0)]
    once, n1 = apply_to_words(copy.deepcopy(words), TABLE)
    twice, n2 = apply_to_words(copy.deepcopy(once), TABLE)
    assert [w.get("text") for w in once] == [w.get("text") for w in twice]
    assert n2 == 0  # nothing left to change on second pass


def test_phrase_merge_joins_tokens_and_keeps_timing():
    words = [W("미소스", 1.0, 1.4), SP(), W("파이브라고", 1.5, 2.0), SP(), W("했다", 2.1, 2.4)]
    out, n = apply_to_words(words, TABLE)
    texts = [w["text"] for w in out if w.get("type") == "word"]
    assert texts == ["Mythos 5라고", "했다"]
    merged = [w for w in out if w.get("type") == "word"][0]
    assert merged["start"] == pytest.approx(1.0)
    assert merged["end"] == pytest.approx(2.0)
    assert merged["raw_text"] == "미소스 파이브라고"


def test_phrase_merge_requires_full_context():
    # 리소스 alone must NOT be touched — only the 파이브-context merge may match it
    words = [W("리소스를", 0.0, 0.5), SP(), W("정리했다", 0.6, 1.0)]
    out, n = apply_to_words(words, TABLE)
    assert n == 0
    assert out[0]["text"] == "리소스를"


def test_non_word_tokens_pass_through():
    words = [{"type": "audio_event", "text": "(laughs)", "start": 0.0, "end": 1.0}]
    out, n = apply_to_words(words, TABLE)
    assert n == 0
    assert out[0]["text"] == "(laughs)"


def test_phrase_merge_aborts_on_interleaved_audio_event():
    # an audio_event between the phrase words must survive — no silent deletion
    event = {"type": "audio_event", "text": "(laughs)", "start": 1.45, "end": 1.48}
    words = [W("미소스", 1.0, 1.4), event, W("파이브라고", 1.5, 2.0)]
    out, _ = apply_to_words(words, TABLE)
    assert event in out
    texts = [w["text"] for w in out if w.get("type") == "word"]
    assert "미소스" in texts[0]  # merge aborted, original token kept
