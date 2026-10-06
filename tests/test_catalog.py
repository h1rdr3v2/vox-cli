import pytest

from vox.catalog import CatalogError, load_catalog, parse_catalog

IDS = [
    "whisper-tiny",
    "whisper-base",
    "whisper-small",
    "whisper-medium",
    "whisper-large-v3-turbo",
    "whisper-large-v3",
    "kokoro-82m",
]


@pytest.mark.parametrize(
    "backend, engines",
    [("mlx", {"mlx-whisper", "kokoro"}), ("portable", {"faster-whisper", "kokoro-onnx"})],
)
def test_shipped_catalog_has_every_model_on_every_platform(backend, engines):
    catalog = load_catalog(backend)
    assert [e.id for e in catalog.entries] == IDS
    assert catalog.version == 2 and catalog.backend == backend
    assert {e.engine for e in catalog.entries} == engines
    for entry in catalog.entries:
        assert entry.repo.count("/") == 1
        assert entry.size_mb > 0


def test_platform_sources_differ():
    mlx, portable = load_catalog("mlx"), load_catalog("portable")
    assert mlx.get("whisper-small").repo == "mlx-community/whisper-small-mlx"
    assert portable.get("whisper-small").repo == "Systran/faster-whisper-small"
    assert portable.get("whisper-large-v3-turbo").repo == "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    assert portable.get("kokoro-82m").repo == "onnx-community/Kokoro-82M-v1.0-ONNX"
    assert portable.get("kokoro-82m").default_voice == "af_heart"


def test_default_backend_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("VOX_BACKEND", "portable")
    assert load_catalog().backend == "portable"


def test_recommended_first():
    catalog = load_catalog("mlx")
    assert catalog.recommended("stt").id == "whisper-large-v3-turbo"
    assert catalog.recommended("tts").id == "kokoro-82m"
    stt = catalog.of_type("stt")
    assert stt[0].id == "whisper-large-v3-turbo"
    # Others keep catalog order.
    assert [e.id for e in stt[1:]] == ["whisper-tiny", "whisper-base", "whisper-small", "whisper-medium", "whisper-large-v3"]


def test_rank_puts_recommended_and_catalog_models_first():
    catalog = load_catalog("mlx")
    assert catalog.rank("whisper-large-v3-turbo") < catalog.rank("whisper-tiny") < catalog.rank("hf:a/b")


GOOD = """
version = 2
[[model]]
id = "a"
type = "stt"
mlx = { engine = "mlx-whisper", repo = "org/a", size_mb = 10 }
"""


def test_parse_minimal():
    catalog = parse_catalog(GOOD, "mlx")
    assert catalog.get("a").note == ""
    assert catalog.get("a").engine == "mlx-whisper"
    assert catalog.get("missing") is None


def test_models_without_a_source_for_this_platform_are_hidden():
    assert parse_catalog(GOOD, "portable").entries == ()


@pytest.mark.parametrize(
    "text, message",
    [
        ("not toml [", "not valid TOML"),
        ("[[model]]\nid='a'", "no integer 'version'"),
        ("version = 99", "newer than this vox supports"),
        (GOOD.replace('type = "stt"\n', ""), "missing 'type'"),
        (GOOD.replace('repo = "org/a", ', ""), "missing 'repo'"),
        (GOOD.replace('type = "stt"', 'type = "xyz"'), "type must be stt or tts"),
        (GOOD.replace('engine = "mlx-whisper"', 'engine = "kokoro"'), "unknown stt engine"),
        (GOOD.replace('engine = "mlx-whisper"', 'engine = "faster-whisper"'), "unknown stt engine"),
        (GOOD.replace("size_mb = 10", "size_mb = -1"), "size_mb must be a positive integer"),
        (GOOD.replace('repo = "org/a"', 'repo = "a"'), "repo must look like org/name"),
        (GOOD + "color = 'red'", "unknown keys"),
        (GOOD.replace("size_mb = 10", "size_mb = 10, sha = 'x'"), "unknown keys"),
        (GOOD + GOOD.replace("version = 2", ""), "duplicate model id"),
        (GOOD.replace('id = "a"', 'id = "hf:x/y"'), "must be a short name"),
        (GOOD.replace("mlx = {", "other = {"), "unknown keys"),
        ('version = 2\n[[model]]\nid = "a"\ntype = "stt"\n', "no source"),
    ],
)
def test_parse_errors(text, message):
    with pytest.raises(CatalogError, match=message):
        parse_catalog(text, "mlx")
