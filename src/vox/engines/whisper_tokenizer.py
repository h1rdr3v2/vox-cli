"""Whisper's tokenizer, built from the vocabularies shipped in vox/resources/whisper.

MLX Whisper repos (mlx-community/whisper-*-mlx) hold only config.json and
weights, so vox brings the tokenizer itself. Adapted from OpenAI Whisper and
mlx-whisper; the vocabulary files are OpenAI's. MIT licensed, see
vox/resources/whisper/LICENSE.
"""

from __future__ import annotations

import base64
import string
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from importlib import resources

import tiktoken

from vox.languages import LANGUAGES

# Languages written without spaces: words are split at unicode boundaries instead.
NO_SPACE_LANGUAGES = {"zh", "ja", "th", "lo", "my", "yue"}


@lru_cache(maxsize=None)
def get_encoding(name: str, num_languages: int) -> tiktoken.Encoding:
    """The "gpt2" (English-only models) or "multilingual" vocabulary plus Whisper's special tokens."""
    text = resources.files("vox").joinpath(f"resources/whisper/{name}.tiktoken").read_text(encoding="ascii")
    ranks = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in text.splitlines() if line)}
    specials = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *[f"<|{code}|>" for code in list(LANGUAGES)[:num_languages]],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
    ]
    special_tokens = {token: len(ranks) + i for i, token in enumerate(specials)}
    return tiktoken.Encoding(
        name=name,
        explicit_n_vocab=len(ranks) + len(specials),
        pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        mergeable_ranks=ranks,
        special_tokens=special_tokens,
    )


@lru_cache(maxsize=None)
def get_tokenizer(multilingual: bool, *, num_languages: int, language: str | None = None, task: str | None = None) -> Tokenizer:
    if multilingual:
        language = language or "en"
        if language not in LANGUAGES:
            raise ValueError(f"unsupported language '{language}'")
        return Tokenizer(get_encoding("multilingual", num_languages), num_languages, language, task or "transcribe")
    return Tokenizer(get_encoding("gpt2", num_languages), num_languages)


@dataclass
class Tokenizer:
    """The interface mlx_audio's Whisper decoding and word timing expect."""

    encoding: tiktoken.Encoding
    num_languages: int
    language: str | None = None
    task: str | None = None
    special_tokens: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.special_tokens = {t: self.encoding.encode_single_token(t) for t in self.encoding.special_tokens_set}

    def encode(self, text: str, **kwargs) -> list[int]:
        return self.encoding.encode(text, **kwargs)

    def decode(self, token_ids: list[int], **kwargs) -> str:
        return self.encoding.decode([t for t in token_ids if t < self.timestamp_begin], **kwargs)

    def decode_with_timestamps(self, token_ids: list[int], **kwargs) -> str:
        return self.encoding.decode(token_ids, **kwargs)

    @cached_property
    def eot(self) -> int:
        return self.encoding.eot_token

    @cached_property
    def sot(self) -> int:
        return self.special_tokens["<|startoftranscript|>"]

    @cached_property
    def transcribe(self) -> int:
        return self.special_tokens["<|transcribe|>"]

    @cached_property
    def translate(self) -> int:
        return self.special_tokens["<|translate|>"]

    @cached_property
    def sot_lm(self) -> int:
        return self.special_tokens["<|startoflm|>"]

    @cached_property
    def sot_prev(self) -> int:
        return self.special_tokens["<|startofprev|>"]

    @cached_property
    def no_speech(self) -> int:
        return self.special_tokens["<|nospeech|>"]

    @cached_property
    def no_timestamps(self) -> int:
        return self.special_tokens["<|notimestamps|>"]

    @cached_property
    def timestamp_begin(self) -> int:
        return self.special_tokens["<|0.00|>"]

    @cached_property
    def language_token(self) -> int:
        if self.language is None:
            raise ValueError("this tokenizer has no language token")
        return self.special_tokens[f"<|{self.language}|>"]

    @cached_property
    def sot_sequence(self) -> tuple[int, ...]:
        sequence = [self.sot]
        if self.language is not None:
            sequence.append(self.language_token)
        if self.task is not None:
            sequence.append(self.transcribe if self.task == "transcribe" else self.translate)
        return tuple(sequence)

    @cached_property
    def sot_sequence_including_notimestamps(self) -> tuple[int, ...]:
        return (*self.sot_sequence, self.no_timestamps)

    @cached_property
    def all_language_codes(self) -> tuple[str, ...]:
        return tuple(list(LANGUAGES)[: self.num_languages])

    @cached_property
    def all_language_tokens(self) -> tuple[int, ...]:
        return tuple(self.special_tokens[f"<|{code}|>"] for code in self.all_language_codes)

    @cached_property
    def non_speech_tokens(self) -> tuple[int, ...]:
        """Tokens suppressed so the model does not invent speaker tags or annotations such as ♪♪♪."""
        symbols = list('"#()*+/:;<=>@[\\]^_`{|}~「」『』')
        symbols += "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪".split()
        # Musical symbols may take several tokens; their first token is safe to suppress.
        miscellaneous = set("♩♪♫♬♭♮♯")
        # Hyphens and apostrophes are allowed inside words, not at their start.
        result = {self.encode(" -")[0], self.encode(" '")[0]}
        for symbol in symbols + list(miscellaneous):
            for tokens in (self.encode(symbol), self.encode(" " + symbol)):
                if len(tokens) == 1 or symbol in miscellaneous:
                    result.add(tokens[0])
        return tuple(sorted(result))

    def split_to_word_tokens(self, tokens: list[int]) -> tuple[list[str], list[list[int]]]:
        if self.language in NO_SPACE_LANGUAGES:
            return self.split_tokens_on_unicode(tokens)
        return self.split_tokens_on_spaces(tokens)

    def split_tokens_on_unicode(self, tokens: list[int]) -> tuple[list[str], list[list[int]]]:
        decoded_full = self.decode_with_timestamps(tokens)
        replacement_char = "�"
        words, word_tokens, current, offset = [], [], [], 0
        for token in tokens:
            current.append(token)
            decoded = self.decode_with_timestamps(current)
            if replacement_char not in decoded or decoded_full[offset + decoded.index(replacement_char)] == replacement_char:
                words.append(decoded)
                word_tokens.append(current)
                current = []
                offset += len(decoded)
        return words, word_tokens

    def split_tokens_on_spaces(self, tokens: list[int]) -> tuple[list[str], list[list[int]]]:
        subwords, subword_tokens_list = self.split_tokens_on_unicode(tokens)
        words, word_tokens = [], []
        for subword, subword_tokens in zip(subwords, subword_tokens_list):
            special = subword_tokens[0] >= self.eot
            with_space = subword.startswith(" ")
            punctuation = subword.strip() in string.punctuation
            if special or with_space or punctuation or not words:
                words.append(subword)
                word_tokens.append(subword_tokens)
            else:
                words[-1] += subword
                word_tokens[-1].extend(subword_tokens)
        return words, word_tokens
