"""Preparing text for speech: Markdown cleanup and sentence-sized chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_CHUNK_CHARS = 350
PAUSE_BETWEEN_CHUNKS = 0.08  # seconds
PAUSE_BETWEEN_PARAGRAPHS = 0.35

ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "inc", "ltd", "co",
    "corp", "no", "fig", "approx", "dept", "est", "mt", "jan", "feb", "mar", "apr", "jun",
    "jul", "aug", "sep", "sept", "oct", "nov", "dec", "e.g", "i.e", "a.m", "p.m", "u.s",
}  # fmt: skip

# End of sentence: terminal punctuation, optional closing quotes or brackets.
_SENTENCE_END = re.compile(r"""([.!?…]+["'”’)\]]*)\s+|([。！？]+)""")


@dataclass
class Chunk:
    text: str
    pause_after: float


def clean_markdown(text: str) -> str:
    """Strip Markdown syntax so it is not read aloud."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)  # code blocks
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)  # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # links
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"<[^>\n]+>", " ", text)  # html tags
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.fullmatch(r"([-*_=]\s*){3,}", stripped) or re.fullmatch(r"\|?[\s:|-]+\|?", stripped or "x"):
            continue  # rules and table separators
        heading = re.match(r"#{1,6}\s+(.*)", stripped)
        if heading:
            title = heading.group(1).strip().rstrip("#").strip()
            if title and title[-1] not in ".!?:":
                title += "."
            lines += ["", title, ""]
            continue
        stripped = re.sub(r"^>\s?", "", stripped)
        stripped = re.sub(r"^([-*+]|\d+[.)])\s+(\[[ xX]\]\s+)?", "", stripped)
        stripped = stripped.strip("|").replace(" | ", ", ")
        lines.append(stripped)
    text = "\n".join(lines)
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text)
    text = re.sub(r"(?<![\w*])([*_])(?!\s)(.+?)(?<!\s)\1(?![\w*])", r"\2", text)
    text = re.sub(r"~~(.+?)~~", r"\1", text)
    return text


def split_sentences(paragraph: str) -> list[str]:
    sentences = []
    start = 0
    for match in _SENTENCE_END.finditer(paragraph):
        end = match.end(1) if match.group(1) else match.end(2)
        candidate = paragraph[start:end].strip()
        if match.group(1) and match.group(1).startswith("."):
            last_word = re.split(r"\s+", candidate[:-1].strip())[-1].lower() if candidate[:-1].strip() else ""
            if last_word.rstrip(".") in ABBREVIATIONS or re.fullmatch(r"[a-z]", last_word):
                continue  # "Dr. Smith", "J. Doe"
        if candidate:
            sentences.append(candidate)
        start = match.end()
    rest = paragraph[start:].strip()
    if rest:
        sentences.append(rest)
    return sentences


def _split_long(sentence: str, limit: int) -> list[str]:
    if len(sentence) <= limit:
        return [sentence]
    parts = []
    current = ""
    for piece in re.split(r"(?<=[,;:])\s+", sentence):
        if len(piece) > limit:
            words = piece.split()
            for word in words:
                if current and len(current) + 1 + len(word) > limit:
                    parts.append(current)
                    current = word
                else:
                    current = f"{current} {word}".strip()
            continue
        if current and len(current) + 1 + len(piece) > limit:
            parts.append(current)
            current = piece
        else:
            current = f"{current} {piece}".strip()
    if current:
        parts.append(current)
    return parts


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    """Split text into sentence-sized chunks, never crossing a paragraph."""
    chunks: list[Chunk] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        current = ""
        for sentence in split_sentences(paragraph):
            for piece in _split_long(sentence, max_chars):
                if current and len(current) + 1 + len(piece) > max_chars:
                    chunks.append(Chunk(current, PAUSE_BETWEEN_CHUNKS))
                    current = piece
                else:
                    current = f"{current} {piece}".strip()
        if current:
            chunks.append(Chunk(current, PAUSE_BETWEEN_PARAGRAPHS))
    if chunks:
        chunks[-1].pause_after = 0.0
    return chunks
