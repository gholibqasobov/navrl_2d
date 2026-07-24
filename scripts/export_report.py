"""Render report/REPORT.md to a styled HTML (then PDF + DOCX via LibreOffice).

    python scripts/export_report.py       # writes report/REPORT.html
    # then:  soffice --headless --convert-to pdf  --outdir report report/REPORT.html
    #        soffice --headless --convert-to docx --outdir report report/REPORT.html

The Markdown image links are relative (figs/*.png), so LibreOffice embeds the
figures when it opens the HTML from inside report/.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"

CSS = """
@page { size: A4; margin: 2cm; }
body { font-family: 'Liberation Serif','DejaVu Serif',serif; font-size: 11pt;
       line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 21pt; margin-bottom: 0.2em; }
h2 { font-size: 14pt; margin-top: 1.1em; border-bottom: 1px solid #cccccc;
     padding-bottom: 2px; }
p { margin: 0.5em 0; text-align: justify; }
img { display: block; margin: 8px auto 2px auto; }
/* image captions are the italic line right under each figure */
img + em, p > em:only-child { display: block; text-align: center; color: #555555;
                              font-size: 9.5pt; }
table { border-collapse: collapse; margin: 10px 0; }
th, td { border: 1px solid #999999; padding: 4px 9px; font-size: 10pt; }
th { background: #f0f0f0; }
code { font-family: 'Liberation Mono', monospace; font-size: 9.5pt;
       background: #f4f4f4; }
pre { background: #f4f4f4; padding: 9px; font-size: 9.5pt; overflow-x: auto; }
a { color: #2460c8; text-decoration: none; }
"""


def inline_images(html: str) -> str:
    """Embed each figs/*.png as a base64 data URI so PDF *and* DOCX carry the image."""

    def repl(match: re.Match) -> str:
        src = match.group(1)
        path = REPORT / src
        if path.exists():
            data = base64.b64encode(path.read_bytes()).decode()
            return f'src="data:image/png;base64,{data}"'
        return match.group(0)

    return re.sub(r'src="([^"]+)"', repl, html)


def main() -> None:
    md = (REPORT / "REPORT.md").read_text()
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
    body = inline_images(body)
    # LibreOffice ignores CSS width on <img> but honors the pixel width attribute;
    # 620px fits the A4 text column (17 cm ≈ 643px at 96 dpi) without overflowing.
    body = body.replace("<img ", '<img width="620" ')
    html = (
        f"<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"utf-8\">\n"
        f"<title>2D Navigation RL — Report</title>\n<style>{CSS}</style>\n</head>\n"
        f"<body>\n{body}\n</body>\n</html>\n"
    )
    out = REPORT / "REPORT.html"
    out.write_text(html)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
