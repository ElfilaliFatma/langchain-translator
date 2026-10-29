"""Streamlit UI for the translator."""

from __future__ import annotations

import streamlit as st

from translator.config import Settings
from translator.languages import AUTO_DETECT, LANGUAGES, language_label
from translator.service import build_default_service

st.set_page_config(page_title="LangChain Translator", page_icon="🌍", layout="wide")


@st.cache_resource(show_spinner=False)
def get_service(provider: str, model: str, temperature: float) -> object:
    settings = Settings.from_env()
    settings = Settings(
        provider=provider,
        model=model,
        base_url=settings.base_url,
        temperature=temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout,
        api_key=settings.api_key,
    )
    return build_default_service(settings)


def main() -> None:
    st.title("🌍 LangChain Translator")
    st.caption("Prompt design · language detection · glossary · automatic review")

    with st.sidebar:
        st.header("Model")
        settings = Settings.from_env()
        provider = st.selectbox("Provider", ["ollama", "openai"], index=0)
        model = st.text_input("Model", value=settings.model if provider == settings.provider else "mistral")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
        tone = st.selectbox("Style", ["", "formal", "casual", "technical", "marketing"])
        review = st.toggle("Ask the model to review the output", value=False)

    col_left, col_right = st.columns(2)
    codes = [AUTO_DETECT, *LANGUAGES.keys()]
    with col_left:
        source_lang = st.selectbox(
            "From",
            codes,
            format_func=lambda c: "Detect automatically" if c == AUTO_DETECT else language_label(c),
        )
        source_text = st.text_area("Source text", height=260, placeholder="Type or paste text…")
    with col_right:
        targets = [c for c in codes if c != AUTO_DETECT]
        # Default to French rather than relying on a positional index, which
        # silently pointed at Spanish once "auto" was added to the source list.
        default_target = targets.index("fr") if "fr" in targets else 0
        target_lang = st.selectbox(
            "To", targets, index=default_target, format_func=language_label
        )
        output_box = st.empty()

    with st.expander("Advanced: context and glossary"):
        context = st.text_input("Domain context", placeholder="e.g. legal contract, medical leaflet")
        glossary_raw = st.text_area(
            "Glossary (one per line: source = target)",
            placeholder="cloud = cloud computing\nlatency = latence",
            height=100,
        )

    glossary = {}
    for line in glossary_raw.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            if key.strip():
                glossary[key.strip()] = value.strip()

    if st.button("Translate", type="primary", disabled=not source_text.strip()):
        try:
            service = get_service(provider, model, temperature)
            with st.spinner("Translating…"):
                result = service.translate(
                    source_text,
                    target_lang=target_lang,
                    source_lang=source_lang,
                    context=context,
                    glossary=glossary,
                    tone=tone,
                )
            with output_box.container():
                st.text_area(
                    f"Translation ({language_label(result.target_lang)})",
                    value=result.text,
                    height=260,
                )
                st.caption(
                    f"Detected source: {language_label(result.source_lang)}"
                    if result.detected
                    else f"Source: {language_label(result.source_lang)}"
                )
                st.download_button(
                    "Download .txt",
                    result.text.encode("utf-8"),
                    file_name=f"translation_{result.target_lang}.txt",
                )
                if review:
                    st.subheader("Automated review")
                    st.write(service.review(source_text, result.text,
                                            result.source_lang, result.target_lang))
        except Exception as exc:  # surfaced in the UI on purpose
            st.error(f"Translation failed: {exc}")


if __name__ == "__main__":
    main()
