"""Command line interface.

Examples:
    python -m translator.cli "Hello world" --to fr
    python -m translator.cli --file notes.txt --to ar --tone formal
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .languages import AUTO_DETECT, supported_codes
from .service import build_default_service


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="translator",
        description="Translate text with a LangChain pipeline.",
    )
    parser.add_argument("text", nargs="?", help="Text to translate.")
    parser.add_argument("--file", type=Path, help="Read the source text from a file.")
    parser.add_argument("--to", required=True, help="Target language code, e.g. fr.")
    parser.add_argument(
        "--from",
        dest="source",
        default=AUTO_DETECT,
        help="Source language code, or 'auto' to detect it (default: auto).",
    )
    parser.add_argument("--tone", default="", help="Optional style, e.g. 'formal'.")
    parser.add_argument("--context", default="", help="Optional domain context.")
    parser.add_argument("--review", action="store_true", help="Print a quality review.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252; Arabic, Japanese or accent-heavy
    # translations would crash the printer without this.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    args = parse_args(argv)
    if args.to not in supported_codes():
        print(f"Unknown target language '{args.to}'. Supported: {', '.join(supported_codes())}")
        return 2

    if args.file:
        if not args.file.exists():
            print(f"File not found: {args.file}")
            return 2
        text = args.file.read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        print("Provide text, --file, or pipe it on stdin.")
        return 2

    service = build_default_service()
    result = service.translate(
        text,
        target_lang=args.to,
        source_lang=args.source,
        tone=args.tone,
        context=args.context,
    )

    print(f"[{result.source_lang} -> {result.target_lang}]")
    print(result.text)
    if args.review:
        print("\nReview:")
        print(service.review(text, result.text, result.source_lang, result.target_lang))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
