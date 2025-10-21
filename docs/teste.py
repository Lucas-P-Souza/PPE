# salvar como md_to_html_math.py na pasta docs (onde está ARCHITECTURE.md)
import pathlib
from markdown import markdown

# Resolve the path to ARCHITECTURE.md relative to this script's directory
script_dir = pathlib.Path(__file__).resolve().parent
p = script_dir / 'ARCHITECTURE.md'
if not p.exists():
  raise FileNotFoundError(f"{p} not found. Ensure 'ARCHITECTURE.md' is located in {script_dir}")
text = p.read_text(encoding='utf8')

# converter Markdown para HTML (ativa extensões básicas)
html_body = markdown(text, extensions=['fenced_code', 'codehilite', 'tables'])

html = f'''<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>ARCHITECTURE</title>
<!-- MathJax -->
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
  body {{ font-family: DejaVu Serif, Georgia, serif; margin: 1in; }}
  pre {{ background:#f6f8fa; padding: .6em; overflow:auto; }}
</style>
</head>
<body>
{html_body}
</body>
</html>'''

path = p.with_suffix('.html')
path.write_text(html, encoding='utf8')
print('Wrote', path)