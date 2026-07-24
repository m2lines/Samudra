# /// script
# dependencies = [
#   "markdown==3.8.2",
#   "weasyprint==69.0",
# ]
# ///

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Render experiment Markdown to a letter-sized PDF with KaTeX math.

Run with:

    uv run --script scripts/render_markdown_pdf.py docs/experiments/report.md

Each input is written beside its source with a ``.pdf`` suffix. KaTeX is
resolved through ``npx`` and WeasyPrint is supplied by the script's inline uv
dependencies.
"""

from __future__ import annotations

import argparse
import html
import re
import subprocess
import tempfile
from pathlib import Path

import markdown  # type: ignore[import-untyped]
from weasyprint import HTML  # type: ignore[import-not-found]

CODE_PATTERN = re.compile(r"(```.*?```|~~~.*?~~~|`[^`\n]*`)", re.DOTALL)
MATH_PATTERN = re.compile(
    r"\\\[(?P<display>.*?)\\\]|\\\((?P<inline>.*?)\\\)",
    re.DOTALL,
)
HEADING_PATTERN = re.compile(r"^#\s+(.+)$", re.MULTILINE)

DOCUMENT_CSS = """
@page {
  size: letter;
  margin: 0.65in 0.68in 0.7in;
  @bottom-center {
    content: counter(page);
    color: #666;
    font: 8pt "DejaVu Sans", sans-serif;
  }
}

html {
  font-size: 10pt;
}

body {
  color: #171717;
  font-family: "DejaVu Serif", serif;
  line-height: 1.36;
}

h1, h2, h3, h4 {
  color: #111;
  font-family: "DejaVu Sans", sans-serif;
  line-height: 1.18;
  page-break-after: avoid;
}

h1 {
  font-size: 20pt;
  margin: 0 0 0.55em;
}

h2 {
  border-bottom: 0.6pt solid #bbb;
  font-size: 13.5pt;
  margin: 1.1em 0 0.42em;
  padding-bottom: 0.12em;
}

h3 {
  font-size: 11.2pt;
  margin: 0.9em 0 0.3em;
}

p {
  margin: 0.35em 0 0.62em;
  orphans: 3;
  widows: 3;
}

blockquote {
  background: #f4f5f7;
  border-left: 3pt solid #7b8794;
  margin: 0.6em 0;
  padding: 0.4em 0.7em;
}

blockquote p {
  margin: 0;
}

table {
  border-collapse: collapse;
  font-family: "DejaVu Sans", sans-serif;
  font-size: 8.2pt;
  margin: 0.65em 0 0.9em;
  page-break-inside: avoid;
  width: 100%;
}

th, td {
  border: 0.5pt solid #999;
  padding: 0.28em 0.38em;
  text-align: left;
  vertical-align: top;
}

th {
  background: #eceff3;
  font-weight: 700;
}

code {
  background: #f1f2f4;
  font-family: "DejaVu Sans Mono", monospace;
  font-size: 0.88em;
  padding: 0.05em 0.16em;
}

pre {
  background: #f1f2f4;
  border: 0.5pt solid #c7c9cc;
  font-size: 8pt;
  line-height: 1.25;
  overflow-wrap: anywhere;
  padding: 0.55em;
  white-space: pre-wrap;
}

pre code {
  padding: 0;
}

ul, ol {
  margin: 0.35em 0 0.65em;
  padding-left: 1.55em;
}

li {
  margin: 0.18em 0;
}

a {
  color: #145ea8;
  text-decoration: none;
}

.katex {
  font-size: 1em;
}

.katex-display {
  margin: 0.65em 0 0.8em;
  overflow: visible;
  page-break-inside: avoid;
}
"""


def _run(command: list[str], *, input_text: str | None = None) -> str:
    return subprocess.run(
        command,
        input=input_text,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def katex_cli() -> Path:
    resolved = _run(
        [
            "npx",
            "--yes",
            "--package",
            "katex",
            "sh",
            "-c",
            'readlink -f "$(command -v katex)"',
        ]
    )
    path = Path(resolved)
    if not path.is_file():
        raise FileNotFoundError(f"Could not resolve the KaTeX CLI: {path}")
    return path


def katex_css(cli: Path) -> str:
    distribution = cli.parent / "dist"
    css_path = distribution / "katex.min.css"
    if not css_path.is_file():
        raise FileNotFoundError(f"Could not locate KaTeX CSS: {css_path}")
    css = css_path.read_text(encoding="utf-8")
    fonts_uri = (distribution / "fonts").as_uri() + "/"
    return css.replace("url(fonts/", f"url({fonts_uri}")


def protect_code(source: str) -> tuple[str, dict[str, str]]:
    protected: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        token = f"CODEPLACEHOLDER{len(protected):06d}TOKEN"
        protected[token] = match.group(0)
        return token

    return CODE_PATTERN.sub(replace, source), protected


def render_math(
    source: str,
    cli: Path,
) -> tuple[str, dict[str, str], set[str]]:
    rendered: dict[str, str] = {}
    display_tokens: set[str] = set()
    cache: dict[tuple[str, bool], str] = {}

    def replace(match: re.Match[str]) -> str:
        display = match.group("display") is not None
        tex = match.group("display") if display else match.group("inline")
        assert tex is not None
        key = tex.strip(), display
        if key not in cache:
            command = ["node", str(cli)]
            if display:
                command.append("--display-mode")
            cache[key] = _run(command, input_text=key[0])
        token = f"MATHPLACEHOLDER{len(rendered):06d}TOKEN"
        rendered[token] = cache[key]
        if display:
            display_tokens.add(token)
        return token

    return MATH_PATTERN.sub(replace, source), rendered, display_tokens


def markdown_html(source: str, cli: Path) -> str:
    protected_source, protected_code = protect_code(source)
    protected_source, rendered_math, display_tokens = render_math(protected_source, cli)
    for token, code in protected_code.items():
        protected_source = protected_source.replace(token, code)

    body = markdown.markdown(
        protected_source,
        extensions=(
            "extra",
            "fenced_code",
            "sane_lists",
            "smarty",
            "tables",
        ),
        output_format="html5",
    )
    for token in display_tokens:
        body = body.replace(f"<p>{token}</p>", rendered_math[token])
    for token, rendered in rendered_math.items():
        body = body.replace(token, rendered)
    return body


def render(source: Path, destination: Path, cli: Path, css: str) -> None:
    markdown_source = source.read_text(encoding="utf-8")
    heading = HEADING_PATTERN.search(markdown_source)
    title = heading.group(1) if heading else source.stem.replace("_", " ").title()
    body = markdown_html(markdown_source, cli)
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>{css}</style>
  <style>{DOCUMENT_CSS}</style>
</head>
<body>{body}</body>
</html>
"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        HTML(string=document, base_url=str(source.parent)).write_pdf(temporary_path)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    arguments = parser.parse_args()

    cli = katex_cli()
    css = katex_css(cli)
    for source in arguments.sources:
        if source.suffix.lower() != ".md":
            parser.error(f"Expected a Markdown source, got {source}.")
        if not source.is_file():
            parser.error(f"Markdown source does not exist: {source}.")
        destination = source.with_suffix(".pdf")
        render(source.resolve(), destination.resolve(), cli, css)
        print(destination)


if __name__ == "__main__":
    main()
