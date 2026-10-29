"""LangChain-powered translation toolkit.

Public surface:
    TranslationService  -- orchestrates prompt | llm | parse | detect
    build_default_service() -- wires the pipeline from environment settings
"""

from .config import Settings
from .service import TranslationService, build_default_service, build_llm

__all__ = [
    "Settings",
    "TranslationService",
    "build_default_service",
    "build_llm",
]

__version__ = "1.0.0"
