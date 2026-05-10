"""USER_GUIDE.md → USER_GUIDE.pdf 변환 (1회용 스크립트).

WeasyPrint + python-markdown 으로 변환한다. 한글 렌더링을 위해 시스템에
설치된 CJK 폰트(WenQuanYi Zen Hei, Unifont) 를 fallback 으로 지정한다.
"""

from pathlib import Path

import markdown
from weasyprint import CSS, HTML

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "USER_GUIDE.md"
DST = ROOT / "USER_GUIDE.pdf"

md_text = SRC.read_text(encoding="utf-8")
html_body = markdown.markdown(
    md_text,
    extensions=["extra", "toc", "tables", "fenced_code", "codehilite"],
)

html_doc = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>auto_ml 사용 설명서</title>
</head>
<body>
{html_body}
</body>
</html>
"""

css = CSS(string="""
@page {
  size: A4;
  margin: 22mm 18mm;
  @bottom-center {
    content: counter(page) " / " counter(pages);
    font-size: 9pt;
    color: #666;
  }
}
body {
  font-family: "Noto Sans CJK KR", "Noto Sans CJK", "WenQuanYi Zen Hei", sans-serif;
  font-size: 10.5pt;
  line-height: 1.55;
  color: #222;
}
h1 { font-size: 22pt; border-bottom: 2px solid #333; padding-bottom: 6px; }
h2 { font-size: 16pt; margin-top: 24px; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
h3 { font-size: 13pt; margin-top: 18px; }
h4 { font-size: 11.5pt; }
code {
  font-family: "DejaVu Sans Mono", "Noto Sans Mono CJK KR", monospace;
  background: #f4f4f4;
  padding: 1px 4px;
  border-radius: 3px;
  font-size: 9.5pt;
}
pre {
  background: #f4f4f4;
  padding: 10px 12px;
  border-radius: 4px;
  border-left: 3px solid #888;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 9.5pt;
}
pre code { background: transparent; padding: 0; }
table {
  border-collapse: collapse;
  width: 100%;
  margin: 10px 0;
  font-size: 10pt;
}
th, td { border: 1px solid #bbb; padding: 6px 8px; text-align: left; }
th { background: #eee; }
blockquote {
  border-left: 4px solid #6aa;
  background: #f0f7f7;
  margin: 8px 0;
  padding: 6px 12px;
  color: #335;
}
hr { border: none; border-top: 1px solid #ccc; margin: 18px 0; }
ul, ol { padding-left: 22px; }
a { color: #0366d6; text-decoration: none; }
""")

HTML(string=html_doc, base_url=str(ROOT)).write_pdf(str(DST), stylesheets=[css])
print(f"wrote {DST} ({DST.stat().st_size:,} bytes)")
