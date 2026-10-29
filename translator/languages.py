"""Static language table: ISO code -> English name + native label.

Keeping this local (instead of an online registry) makes the UI deterministic
and testable. The model receives the English name, which frontier models handle
most reliably, while the user sees the native label.
"""

from __future__ import annotations

LANGUAGES: dict[str, dict[str, str]] = {
    "ar": {"name": "Arabic", "native": "العربية"},
    "de": {"name": "German", "native": "Deutsch"},
    "en": {"name": "English", "native": "English"},
    "es": {"name": "Spanish", "native": "Español"},
    "fr": {"name": "French", "native": "Français"},
    "it": {"name": "Italian", "native": "Italiano"},
    "ja": {"name": "Japanese", "native": "日本語"},
    "pt": {"name": "Portuguese", "native": "Português"},
    "ru": {"name": "Russian", "native": "Русский"},
    "tr": {"name": "Turkish", "native": "Türkçe"},
    "zh": {"name": "Chinese (Simplified)", "native": "简体中文"},
}

AUTO_DETECT = "auto"


def language_name(code: str) -> str:
    """Return the English name of a language code, or the code itself."""

    return LANGUAGES.get(code.lower(), {}).get("name", code)


def language_label(code: str) -> str:
    """Return a UI label such as ``French (Français)``."""

    entry = LANGUAGES.get(code.lower())
    if not entry:
        return code
    return f"{entry['name']} ({entry['native']})"


def supported_codes() -> list[str]:
    return sorted(LANGUAGES)
