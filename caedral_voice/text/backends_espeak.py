"""DEV-ONLY espeak-ng reference backend (GPLv3) — NEVER a Caedral dependency path.

espeak-ng is GPLv3 and its maintainers refuse linking exceptions
(docs/research/TEXT-FRONTEND.md §2.1). This module exists ONLY to produce
dev-time silver/reference IPA for experiments (CV-G1-E0001-e). It must never
be imported by Caedral runtime code; the GPL landmine stays quarantined here:

  - importing this module pulls in phonemizer + espeakng_loader (GPL);
  - importing caedral_voice.text does NOT import this module;
  - tests/test_text_frontend.py asserts that quarantine.

Wiring follows benchmarks/run_baseline_kittentts.py: point phonemizer's
EspeakWrapper at the pip-provided library/data paths BEFORE constructing a
backend.
"""
from __future__ import annotations

_GPL_WARNING = (
    "backends_espeak is a DEV-ONLY GPLv3 reference backend (espeak-ng via "
    "phonemizer/espeakng_loader). It must never be imported from Caedral "
    "runtime or shipped code."
)


def get_espeak_ref_backend(language: str = "en-us"):
    """Construct an espeak-ng-backed phonemizer callable, or raise RuntimeError
    with the GPL explanation if the dev tooling is unavailable."""
    try:
        import espeakng_loader  # type: ignore
        import phonemizer.backend.espeak.wrapper as _wrapper  # type: ignore
        from phonemizer.backend import EspeakBackend  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on dev env
        raise RuntimeError(
            f"{_GPL_WARNING} Dev tooling not importable: {exc}"
        ) from exc

    _wrapper.EspeakWrapper.set_library(espeakng_loader.get_library_path())
    _wrapper.EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    backend = EspeakBackend(
        language=language,
        preserve_punctuation=False,
        with_stress=True,
    )
    return backend


def phonemize(text: str, language: str = "en-us") -> str:
    """One-shot convenience: normalized-or-raw text -> espeak-ng IPA."""
    backend = get_espeak_ref_backend(language)
    return backend.phonemize([text])[0]
