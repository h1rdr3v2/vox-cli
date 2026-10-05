import json

from vox import formats
from vox.engines.base import Segment, Transcription
from vox.text import (
    PAUSE_BETWEEN_PARAGRAPHS,
    chunk_text,
    clean_markdown,
    split_sentences,
)


def test_split_sentences_keeps_abbreviations():
    text = "Dr. Smith arrived at 5 p.m. today. He said hi! Did you hear? Yes."
    assert split_sentences(text) == ["Dr. Smith arrived at 5 p.m. today.", "He said hi!", "Did you hear?", "Yes."]


def test_split_sentences_cjk():
    assert split_sentences("今日は晴れです。明日は雨です。") == ["今日は晴れです。", "明日は雨です。"]


def test_chunks_respect_limit_and_paragraphs():
    long_sentence = ", ".join(["word " * 20] * 6) + "."
    text = "Short one. Another short one.\n\nNew paragraph here. " + long_sentence
    chunks = chunk_text(text, max_chars=200)
    assert chunks[0].text == "Short one. Another short one."
    assert chunks[0].pause_after == PAUSE_BETWEEN_PARAGRAPHS
    assert all(len(c.text) <= 200 for c in chunks)
    assert chunks[-1].pause_after == 0.0
    joined = " ".join(c.text for c in chunks)
    assert joined.split() == text.split()


def test_chunk_empty():
    assert chunk_text("   \n\n  ") == []


def test_clean_markdown():
    md = """# Title

Some **bold** and _italic_ text with a [link](https://example.com) and `code`.

- first item
- [x] done item

```python
print("skip me")
```

| a | b |
|---|---|
| 1 | 2 |
"""
    text = clean_markdown(md)
    assert "Title." in text
    assert "Some bold and italic text with a link and code." in text
    assert "first item" in text and "done item" in text
    assert "print" not in text and "**" not in text and "](" not in text and "---" not in text


def _result():
    return Transcription(
        text="Hello there. General Kenobi.",
        language="en",
        duration=3725.5,
        segments=[
            Segment(id=0, start=0.0, end=1.25, text=" Hello there."),
            Segment(id=1, start=3723.0, end=3725.5, text=" General Kenobi."),
        ],
    )


def test_srt():
    assert formats.to_srt(_result()) == (
        "1\n00:00:00,000 --> 00:00:01,250\nHello there.\n\n"
        "2\n01:02:03,000 --> 01:02:05,500\nGeneral Kenobi.\n"
    )


def test_vtt():
    out = formats.to_vtt(_result())
    assert out.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:01.250\nHello there.\n")


def test_json_formats():
    assert json.loads(formats.render(_result(), "json")) == {"text": "Hello there. General Kenobi."}
    verbose = json.loads(formats.render(_result(), "verbose_json", "english"))
    assert verbose["language"] == "english"
    assert verbose["duration"] == 3725.5
    assert [s["text"] for s in verbose["segments"]] == [" Hello there.", " General Kenobi."]
    assert formats.render(_result(), "text") == "Hello there. General Kenobi.\n"
