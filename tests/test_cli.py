from __future__ import annotations

from translator import cli


def test_cli_rejects_unknown_language(capsys):
    code = cli.main(["hello", "--to", "xx"])
    assert code == 2
    assert "Unknown target language" in capsys.readouterr().out


def test_cli_parses_arguments():
    args = cli.parse_args(["hello", "--to", "fr", "--from", "en", "--tone", "formal"])
    assert args.text == "hello"
    assert args.to == "fr"
    assert args.source == "en"
    assert args.tone == "formal"
