import pytest

from vox.catalog import CatalogError, load_catalog, parse_catalog


def test_shipped_catalog_parses():
    catalog = load_catalog()
    ids = [e.id for e in catalog.entries]
    assert ids == [
        "whisper-tiny",
        "whisper-base",
        "whisper-small",
        "whisper-medium",
        "whisper-large-v3-turbo",
        "whisper-large-v3",
        "kokoro-82m",
    ]
    assert catalog.version == 1
    for entry in catalog.entries:
        assert entry.repo.count("/") == 1
        assert entry.size_mb > 0
        assert entry.engine in ("mlx-whisper", "kokoro")


def test_recommended_first():
    catalog = load_catalog()
    assert catalog.recommended("stt").id == "whisper-large-v3-turbo"
    assert catalog.recommended("tts").id == "kokoro-82m"
    stt = catalog.of_type("stt")
    assert stt[0].id == "whisper-large-v3-turbo"
    # Others keep catalog order.
    assert [e.id for e in stt[1:]] == ["whisper-tiny", "whisper-base", "whisper-small", "whisper-medium", "whisper-large-v3"]


def test_rank_puts_recommended_and_catalog_models_first():
    catalog = load_catalog()
    assert catalog.rank("whisper-large-v3-turbo") < catalog.rank("whisper-tiny") < catalog.rank("hf:a/b")


GOOD = """
version = 1
[[model]]
id = "a"
type = "stt"
engine = "mlx-whisper"
repo = "org/a"
size_mb = 10
"""


def test_parse_minimal():
    catalog = parse_catalog(GOOD)
    assert catalog.get("a").note == ""
    assert catalog.get("missing") is None


@pytest.mark.parametrize(
    "text, message",
    [
        ("not toml [", "not valid TOML"),
        ("[[model]]\nid='a'", "no integer 'version'"),
        ("version = 99", "newer than this vox supports"),
        (GOOD.replace('repo = "org/a"\n', ""), "missing 'repo'"),
        (GOOD.replace('type = "stt"', 'type = "xyz"'), "type must be stt or tts"),
        (GOOD.replace('engine = "mlx-whisper"', 'engine = "kokoro"'), "unknown stt engine"),
        (GOOD.replace("size_mb = 10", "size_mb = -1"), "size_mb must be a positive integer"),
        (GOOD + "color = 'red'", "unknown keys"),
        (GOOD + GOOD.replace("version = 1", ""), "duplicate model id"),
        (GOOD.replace('id = "a"', 'id = "hf:x/y"'), "must be a short name"),
    ],
)
def test_parse_errors(text, message):
    with pytest.raises(CatalogError, match=message):
        parse_catalog(text)
