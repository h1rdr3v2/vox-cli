"""Transcript output formats (OpenAI response_format values)."""

from __future__ import annotations

import json

from vox.engines.base import Transcription

RESPONSE_FORMATS = ("json", "text", "srt", "vtt", "verbose_json")
# CLI --format value -> API response_format
CLI_TO_API = {"txt": "text", "srt": "srt", "vtt": "vtt", "json": "verbose_json"}
MEDIA_TYPES = {
    "json": "application/json",
    "verbose_json": "application/json",
    "text": "text/plain; charset=utf-8",
    "srt": "application/x-subrip; charset=utf-8",
    "vtt": "text/vtt; charset=utf-8",
}


def timestamp(seconds: float, separator: str) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{separator}{ms:03d}"


def to_text(result: Transcription) -> str:
    return result.text.strip() + "\n"


def to_srt(result: Transcription) -> str:
    blocks = []
    for i, seg in enumerate((s for s in result.segments if s.text.strip()), start=1):
        blocks.append(
            f"{i}\n{timestamp(seg.start, ',')} --> {timestamp(seg.end, ',')}\n{seg.text.strip()}\n"
        )
    return "\n".join(blocks)


def to_vtt(result: Transcription) -> str:
    blocks = ["WEBVTT\n"]
    for seg in result.segments:
        if seg.text.strip():
            blocks.append(f"{timestamp(seg.start, '.')} --> {timestamp(seg.end, '.')}\n{seg.text.strip()}\n")
    return "\n".join(blocks)


def verbose(result: Transcription, language_name: str | None = None, words: bool = False) -> dict:
    data = {
        "task": "transcribe",
        "language": language_name or result.language or "",
        "duration": round(result.duration, 3),
        "text": result.text,
        "segments": [
            {
                "id": seg.id,
                "seek": 0,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text,
                "tokens": seg.tokens,
                "temperature": seg.temperature,
                "avg_logprob": seg.avg_logprob,
                "compression_ratio": seg.compression_ratio,
                "no_speech_prob": seg.no_speech_prob,
            }
            for seg in result.segments
        ],
    }
    if words:
        data["words"] = [
            {"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3)}
            for seg in result.segments
            for w in (seg.words or [])
        ]
    return data


def render(result: Transcription, response_format: str, language_name: str | None = None, words: bool = False) -> str:
    if response_format == "json":
        return json.dumps({"text": result.text}, ensure_ascii=False)
    if response_format == "verbose_json":
        return json.dumps(verbose(result, language_name, words), ensure_ascii=False, indent=2) + "\n"
    if response_format == "text":
        return to_text(result)
    if response_format == "srt":
        return to_srt(result)
    if response_format == "vtt":
        return to_vtt(result)
    raise ValueError(response_format)
