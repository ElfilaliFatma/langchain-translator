"""Runtime configuration, read once from the environment.

The app never hard-codes a provider. It defaults to a local Ollama model so the
project runs offline and free, but any chat model served through LangChain can
be swapped in with two environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

try:  # optional dependency: only needed when a .env file is used
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the application settings."""

    provider: str = "ollama"
    model: str = "mistral"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = 600
    timeout: int = 300
    api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> "Settings":
        provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
        default_model = "mistral" if provider == "ollama" else "gpt-4o-mini"
        api_key = os.getenv("OPENAI_API_KEY") if provider == "openai" else None
        return cls(
            provider=provider,
            model=os.getenv("LLM_MODEL", default_model),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "600")),
            timeout=int(os.getenv("LLM_TIMEOUT", "300")),
            api_key=api_key,
        )
