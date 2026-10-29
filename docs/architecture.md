# Architecture

```
            ┌──────────────────────────────┐
  text ───► │ TranslationService.translate │
            └──────────────┬───────────────┘
                           │ source == auto ?
                 ┌─────────▼──────────┐
                 │ detection chain    │  detection_prompt | llm | parser
                 └─────────┬──────────┘
                           │ ISO code (heuristic fallback)
                 ┌─────────▼──────────┐
                 │ translation chain  │  translation_prompt | llm | parser
                 └─────────┬──────────┘
                           │ raw text
                 ┌─────────▼──────────┐
                 │ clean_output()     │  strip chatter / quotes
                 └─────────┬──────────┘
                           ▼
                  TranslationResult
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
    Streamlit UI        CLI           evaluation script
                                            │
                                   quality chain (judge)
                                   BLEU via sacreBLEU
```

## Responsibilities

| Module | Responsibility | Depends on |
| --- | --- | --- |
| `config.py` | read `.env`, freeze settings | env only |
| `languages.py` | code ↔ name mapping | stdlib |
| `prompts.py` | prompt contracts | `langchain_core` |
| `service.py` | orchestration, parsing, fallbacks | LangChain, provider SDK |
| `cli.py` / `app.py` | presentation | `service.py` |

The service layer has no knowledge of Streamlit or argparse, and the tests
instantiate it with a fake chat model. That separation is what makes the
project testable without a GPU or an API key.
