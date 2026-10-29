"""Translation service: the LangChain chain, wrapped in a small, testable API.

The chain is intentionally built with LCEL (LangChain Expression Language):

    prompt | llm | StrOutputParser()

Every collaborator is injected, so tests can swap the model for a fake without
touching the code under test.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from .config import Settings
from .languages import AUTO_DETECT, language_name, supported_codes
from .prompts import (
    detection_prompt,
    quality_prompt,
    translation_prompt,
)


def build_llm(settings: Settings) -> BaseChatModel:
    """Create the chat model described by ``settings``.

    Imports are local so that importing the package never requires both
    provider SDKs to be installed.
    """

    if settings.provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.model,
            base_url=settings.base_url,
            temperature=settings.temperature,
            num_predict=settings.max_tokens,
        )
    if settings.provider == "openai":
        if not settings.api_key:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.model,
            api_key=settings.api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            timeout=settings.timeout,
        )
    raise ValueError(
        f"Unsupported provider '{settings.provider}'. Use 'ollama' or 'openai'."
    )


@dataclass
class TranslationResult:
    text: str
    source_lang: str
    target_lang: str
    detected: bool = False


class TranslationService:
    """High-level API around the translation, detection and review chains."""

    def __init__(self, llm: BaseChatModel, settings: Settings | None = None):
        self.llm = llm
        self.settings = settings or Settings()
        self.translate_chain = translation_prompt | llm | StrOutputParser()
        self.detect_chain = detection_prompt | llm | StrOutputParser()
        self.review_chain = quality_prompt | llm | StrOutputParser()

    # ------------------------------------------------------------------ utils
    def detect_language(self, text: str) -> str:
        """Return an ISO 639-1 code, falling back to a heuristic if needed."""

        raw = self.detect_chain.invoke({"text": text}).strip().lower()
        code = "".join(ch for ch in raw[:12] if ch.isalpha())
        if len(code) >= 2:
            code = code[:2]
            if code in supported_codes():
                return code
        return heuristic_language(text)

    # --------------------------------------------------------------- public API
    def translate(
        self,
        text: str,
        target_lang: str,
        source_lang: str = AUTO_DETECT,
        context: str = "",
        glossary: dict[str, str] | None = None,
        tone: str = "",
    ) -> TranslationResult:
        if not text or not text.strip():
            raise ValueError("Nothing to translate: the input text is empty.")
        if target_lang == AUTO_DETECT:
            raise ValueError("The target language must be explicit.")

        detected = source_lang == AUTO_DETECT
        resolved_source = self.detect_language(text) if detected else source_lang
        if resolved_source == target_lang:
            return TranslationResult(text, resolved_source, target_lang, detected)

        translated = self.translate_chain.invoke(
            {
                "text": text,
                "source_lang": language_name(resolved_source),
                "target_lang": language_name(target_lang),
                "context_line": f"Domain/context: {context}" if context else "",
                "glossary_line": format_glossary(glossary),
                "tone_line": f"Style: {tone}" if tone else "",
            }
        ).strip()

        return TranslationResult(
            text=clean_output(translated),
            source_lang=resolved_source,
            target_lang=target_lang,
            detected=detected,
        )

    def review(self, source: str, candidate: str, source_lang="en", target_lang="fr") -> str:
        """Ask the model to critique a translation (LLM-as-a-judge)."""

        return self.review_chain.invoke(
            {
                "source": source,
                "candidate": candidate,
                "source_lang": language_name(source_lang),
                "target_lang": language_name(target_lang),
            }
        ).strip()


# --------------------------------------------------------------------- helpers
_LEADING_NOISE = (
    "sure, here is the translation:",
    "here is the translation:",
    "translation:",
)


def clean_output(text: str) -> str:
    """Strip common small-model chatter around an otherwise correct answer.

    Two failure modes are observed with small local models:

    1. a preface ("Sure, here is the translation:");
    2. trailing commentary on its own line, usually starting with "(" or
       "Note:". Both are removed while multi-line translations are preserved.
    """

    cleaned = _drop_trailing_commentary(text.strip())
    lowered = cleaned.lower()
    for noise in _LEADING_NOISE:
        if lowered.startswith(noise):
            cleaned = cleaned[len(noise) :].strip()
            break
    if len(cleaned) > 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "\"'":
        # Models sometimes wrap the answer in quotes; keep inner quotes intact.
        if cleaned.count(cleaned[0]) == 2:
            cleaned = cleaned[1:-1].strip()
    return cleaned


_COMMENTARY_PREFIXES = ("(", "[", "note:", "translation note", "explanation:", "n.b.")


def _drop_trailing_commentary(text: str) -> str:
    """Remove a trailing line that is clearly meta-commentary."""

    lines = text.splitlines()
    while len(lines) > 1 and lines[-1].strip().lower().startswith(_COMMENTARY_PREFIXES):
        lines.pop()
    return "\n".join(lines).strip()


def format_glossary(glossary: dict[str, str] | None) -> str:
    if not glossary:
        return ""
    pairs = "; ".join(f"{src} => {dst}" for src, dst in glossary.items())
    return f"Mandatory terminology: {pairs}"


def heuristic_language(text: str) -> str:
    """Cheap script-based fallback used when the model answer is unusable."""

    if any("\u0600" <= ch <= "\u06ff" for ch in text):
        return "ar"
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return "zh"
    if any("\u3040" <= ch <= "\u30ff" for ch in text):
        return "ja"
    if any("\u0400" <= ch <= "\u04ff" for ch in text):
        return "ru"
    if any(ch in "ğışçöüİĞŞÇÖÜ" for ch in text):
        return "tr"
    words = {w.strip(".,!?;:").lower() for w in text.split()}
    markers = {
        "fr": {"le", "la", "les", "des", "est", "et", "une", "dans", "pour"},
        "es": {"el", "la", "los", "es", "una", "para", "con", "que", "del"},
        "de": {"der", "die", "das", "und", "ist", "nicht", "ein", "mit"},
        "it": {"il", "la", "gli", "una", "per", "con", "che", "sono"},
        "pt": {"o", "a", "os", "uma", "para", "com", "que", "não"},
    }
    scores = {code: len(words & vocab) for code, vocab in markers.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "en"


def build_default_service(settings: Settings | None = None) -> TranslationService:
    settings = settings or Settings.from_env()
    return TranslationService(build_llm(settings), settings)
