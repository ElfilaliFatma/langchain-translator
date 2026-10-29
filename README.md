# LangChain Translator

A translation application built with **LangChain** (LCEL chains, prompt
templates, output parsing) that runs against a local Ollama model or OpenAI.

![flow](docs/architecture.md)

## Features

| Capability | How it works |
| --- | --- |
| Translation | `ChatPromptTemplate \| ChatModel \| StrOutputParser` chain |
| Automatic source detection | dedicated detection chain + script heuristic fallback |
| Glossary & tone control | optional prompt slots (`cloud => nuage`, `formal`) |
| Output hygiene | `clean_output()` removes "Sure, here is the translation:" chatter and wrapping quotes |
| Quality review | LLM-as-a-judge chain that lists concrete defects |
| Evaluation | BLEU via sacreBLEU + judge score on `examples/evaluation_set.json` |
| Interfaces | Streamlit web UI and a CLI (`python -m translator.cli`) |

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt

# Local model
ollama pull mistral          # if not already present
ollama serve                 # usually already running

copy .env.example .env
```

### CLI

```powershell
python -m translator.cli "The deployment is scheduled for Monday." --to fr
python -m translator.cli --file notes.txt --to ar --tone formal --review
```

### Web UI

```powershell
streamlit run translator/app.py
```

### Tests and evaluation

```powershell
pytest
python scripts/evaluate.py --limit 4
```

Tests use a scripted fake chat model, so they run without Ollama and without a
network connection. The evaluation script does call the model.

## Project layout

```
translator/
  config.py      settings from environment
  languages.py   ISO codes, English names, native labels
  prompts.py     system + human templates (injection-safe <source> tags)
  service.py     TranslationService, LLM factory, helpers
  cli.py         command line entry point
  app.py         Streamlit interface
scripts/
  evaluate.py    BLEU + LLM-judge evaluation
examples/
  evaluation_set.json
tests/
```

## Why LangChain

LangChain supplies the pieces this app would otherwise hand-roll: provider
abstraction (`ChatOllama` / `ChatOpenAI` behind `BaseChatModel`), composable
prompt templates with named slots, the `|` operator for pipelines, and output
parsers. Swapping provider or model is a configuration change, not a rewrite,
and the same `TranslationService` is reused by the CLI, the UI and the
evaluation script.

## Limitations

- Poor translation quality is inherent to small local models; the judge chain
  helps detect it but does not fix it.
- Long documents are sent in one prompt, so they can exceed the context window.
- Bleu is a shallow metric: it rewards literal overlap and misses fluency.
- Language detection is a second model call, which doubles latency.

## Improvements

- Chunk long documents with `RecursiveCharacterTextSplitter` and translate
  chunk-by-chunk with an overlap, keeping paragraph boundaries.
- Cache translations by hash (`langchain.cache`) to cut repeated cost.
- Add terminology extraction and an automatic glossary.
- Batch the evaluation and add COMET / BERTScore for semantic quality.
- Stream tokens to the UI with `chain.stream()`.

## Report

See [`docs/REPORT.md`](docs/REPORT.md) for the technical write-up: objectives,
architecture, component analysis, test protocol, results, limitations and
improvements.
