"""Unit tests: no network, no model server required.

A scripted fake chat model stands in for the LLM, so the tests assert the
contract of the chain (prompt contents, parsing, fallbacks) rather than the
quality of a specific model.
"""

from __future__ import annotations

from typing import Any, List, Optional

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from translator.languages import language_name
from translator.service import (
    TranslationService,
    clean_output,
    format_glossary,
    heuristic_language,
)


class ScriptedChatModel(BaseChatModel):
    """Returns queued responses and records the prompts it received."""

    responses: List[str]
    calls: List[List[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        text = self.responses.pop(0) if self.responses else ""
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def make_service(*responses: str) -> TranslationService:
    return TranslationService(ScriptedChatModel(responses=list(responses)))


def test_translate_returns_clean_text():
    service = make_service("Bonjour le monde")
    result = service.translate("Hello world", target_lang="fr", source_lang="en")
    assert result.text == "Bonjour le monde"
    assert result.source_lang == "en"
    assert result.detected is False


def test_prompt_contains_language_names_and_payload():
    service = make_service("ok")
    service.translate("Hello", target_lang="de", source_lang="en")
    human = service.llm.calls[-1][-1].content
    assert "English" in human and "German" in human
    assert "<source>\nHello\n</source>" in human
    assert language_name("de") == "German"


def test_auto_detection_populates_source_language():
    service = make_service("fr", "Bonjour")
    result = service.translate("Hello", target_lang="de", source_lang="auto")
    assert result.source_lang == "fr"
    assert result.detected is True
    assert result.text == "Bonjour"
    assert len(service.llm.calls) == 2  # detect, then translate


def test_detection_equal_to_target_short_circuits():
    service = make_service("fr")
    result = service.translate("Bonjour", target_lang="fr", source_lang="auto")
    assert result.text == "Bonjour"
    assert len(service.llm.calls) == 1


def test_same_language_short_circuits_without_model_call():
    service = make_service()
    result = service.translate("Bonjour", target_lang="fr", source_lang="fr")
    assert result.text == "Bonjour"
    assert service.llm.calls == []


def test_empty_input_rejected():
    service = make_service()
    with pytest.raises(ValueError):
        service.translate("   ", target_lang="fr", source_lang="en")


def test_auto_target_rejected():
    service = make_service()
    with pytest.raises(ValueError):
        service.translate("Hello", target_lang="auto", source_lang="en")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Sure, here is the translation: Bonjour", "Bonjour"),
        ('"Bonjour"', "Bonjour"),
        ("Bonjour", "Bonjour"),
        ("Here is the translation:\nBonjour\n", "Bonjour"),
        ("Bonjour\n(The translation is approximate)", "Bonjour"),
        ("Bonjour\nNote: informal register", "Bonjour"),
        # a genuine multi-line translation keeps its last line
        ("Line one\nLine two", "Line one\nLine two"),
    ],
)
def test_clean_output_strips_chatter(raw, expected):
    assert clean_output(raw) == expected


def test_glossary_line_is_rendered():
    service = make_service("ok")
    service.translate(
        "The cloud is fast.",
        target_lang="fr",
        source_lang="en",
        glossary={"cloud": "nuage"},
    )
    assert "cloud => nuage" in service.llm.calls[-1][-1].content
    assert format_glossary(None) == ""


def test_detection_falls_back_to_heuristics():
    service = make_service("unknown-language")
    assert service.detect_language("مرحبا بالعالم") == "ar"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("مرحبا بالعالم", "ar"),
        ("你好世界", "zh"),
        ("Привет мир", "ru"),
        ("Le chat est dans la maison", "fr"),
        ("Hello there", "en"),
    ],
)
def test_heuristic_language(text, expected):
    assert heuristic_language(text) == expected
