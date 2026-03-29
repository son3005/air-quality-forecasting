# -*- coding: utf-8 -*-
"""
Convert architecture report .md files to styled HTML with Mermaid diagrams.
Open the HTML in browser -> Ctrl+P -> Save as PDF.
"""
import re
import os

REPORTS = [
    {
        'md_path': r'C:\Users\ADMIN\.gemini\antigravity\brain\03461b6e-84ca-4443-8c43-38d0a3e115b2\stxlinear_architecture_report.md',
        'html_path': r'e:\University\Year 3 -2\DA2\CODE\report\stxlinear_architecture_report.html',
        'title': 'ST-XLinear Architecture Report'
    },
    {
        'md_path': r'C:\Users\ADMIN\.gemini\antigravity\brain\03461b6e-84ca-4443-8c43-38d0a3e115b2\ensemble_architecture_report.md',
        'html_path': r'e:\University\Year 3 -2\DA2\CODE\report\ensemble_architecture_report.html',
        'title': 'Ensemble Architecture Report'
    },
    {
        'md_path': r'C:\Users\ADMIN\.gemini\antigravity\brain\03461b6e-84ca-4443-8c43-38d0a3e115b2\preprocessing_workflow_report.md',
        'html_path': r'e:\University\Year 3 -2\DA2\CODE\report\preprocessing_workflow_report.html',
        'title': 'Preprocessing Workflow Report'
    },
]

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>mermaid.initialize({{startOnLoad:true, theme:'default'}});</script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400&display=swap');
  
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  
  body {{
    font-family: 'Inter', sans-serif;
    line-height: 1.7;
    color: #1a1a2e;
    background: #ffffff;
    max-width: 900px;
    margin: 0 auto;
    padding: 40px 50px;
  }}
  
  h1 {{
    font-size: 28px;
    color: #0f3460;
    border-bottom: 3px solid #e94560;
    padding-bottom: 12px;
    margin-bottom: 24px;
  }}
  
  h2 {{
    font-size: 22px;
    color: #16213e;
    margin-top: 36px;
    margin-bottom: 16px;
    padding-left: 12px;
    border-left: 4px solid #e94560;
  }}
  
  h3 {{
    font-size: 17px;
    color: #0f3460;
    margin-top: 24px;
    margin-bottom: 10px;
  }}
  
  p {{ margin-bottom: 12px; }}
  
  code {{
    font-family: 'JetBrains Mono', monospace;
    background: #f0f0f5;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.9em;
    color: #e94560;
  }}
  
  pre {{
    background: #1a1a2e;
    color: #e0e0e0;
    padding: 16px 20px;
    border-radius: 8px;
    overflow-x: auto;
    margin: 16px 0;
    font-size: 13px;
    line-height: 1.5;
  }}
  
  pre code {{
    background: none;
    color: #e0e0e0;
    padding: 0;
  }}
  
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    font-size: 14px;
  }}
  
  th {{
    background: #0f3460;
    color: white;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
  }}
  
  td {{
    padding: 8px 14px;
    border-bottom: 1px solid #e0e0e0;
  }}
  
  tr:nth-child(even) td {{ background: #f8f9fa; }}
  tr:hover td {{ background: #e8f4fd; }}
  
  blockquote {{
    border-left: 4px solid #e94560;
    padding: 12px 16px;
    margin: 16px 0;
    background: #fff3f5;
    border-radius: 0 8px 8px 0;
  }}
  
  .mermaid {{
    text-align: center;
    margin: 24px 0;
    padding: 20px;
    background: #fafbfc;
    border-radius: 12px;
    border: 1px solid #e0e0e0;
  }}
  
  hr {{
    border: none;
    border-top: 2px solid #e0e0e0;
    margin: 32px 0;
  }}
  
  ul, ol {{
    margin: 8px 0 16px 24px;
  }}
  
  li {{ margin-bottom: 4px; }}
  
  strong {{ color: #0f3460; }}
  
  @media print {{
    body {{ padding: 20px; max-width: 100%; }}
    .mermaid {{ page-break-inside: avoid; }}
    h2 {{ page-break-before: auto; }}
  }}
</style>
</head>
<body>
{content}
</body>
</html>
"""


def md_to_html(md_text):
    """Simple markdown to HTML converter with Mermaid support."""
    lines = md_text.split('\n')
    html_parts = []
    i = 0
    in_table = False
    table_header_done = False
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Mermaid code blocks
        if stripped.startswith('```mermaid'):
            mermaid_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() != '```':
                mermaid_lines.append(lines[i])
                i += 1
            html_parts.append(f'<div class="mermaid">\n' + '\n'.join(mermaid_lines) + '\n</div>')
            i += 1
            continue
        
        # Python/other code blocks
        if stripped.startswith('```'):
            lang = stripped[3:].strip()
            code_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() != '```':
                code_lines.append(lines[i].replace('<', '&lt;').replace('>', '&gt;'))
                i += 1
            html_parts.append(f'<pre><code>' + '\n'.join(code_lines) + '</code></pre>')
            i += 1
            continue
        
        # Tables
        if '|' in stripped and stripped.startswith('|'):
            if stripped.replace('|', '').replace('-', '').replace(' ', '') == '':
                # separator row
                i += 1
                continue
            
            cells = [c.strip() for c in stripped.split('|')[1:-1]]
            
            if not in_table:
                html_parts.append('<table>')
                in_table = True
                tag = 'th'
                table_header_done = False
            else:
                tag = 'td'
            
            row = '<tr>' + ''.join(f'<{tag}>{format_inline(c)}</{tag}>' for c in cells) + '</tr>'
            html_parts.append(row)
            
            # Check if next line is separator
            if i + 1 < len(lines) and lines[i+1].strip().replace('|', '').replace('-', '').replace(' ', '') == '':
                table_header_done = True
            
            i += 1
            continue
        elif in_table:
            html_parts.append('</table>')
            in_table = False
            table_header_done = False
        
        # Headers
        if stripped.startswith('# '):
            html_parts.append(f'<h1>{format_inline(stripped[2:])}</h1>')
            i += 1
            continue
        if stripped.startswith('## '):
            html_parts.append(f'<h2>{format_inline(stripped[3:])}</h2>')
            i += 1
            continue
        if stripped.startswith('### '):
            html_parts.append(f'<h3>{format_inline(stripped[4:])}</h3>')
            i += 1
            continue
        
        # Horizontal rule
        if stripped == '---':
            html_parts.append('<hr>')
            i += 1
            continue
        
        # Blockquote (alerts)
        if stripped.startswith('>'):
            bq_lines = []
            while i < len(lines) and lines[i].strip().startswith('>'):
                bq_lines.append(lines[i].strip().lstrip('>').strip())
                i += 1
            content = ' '.join(bq_lines)
            # Remove [!IMPORTANT], [!NOTE] etc markers
            content = re.sub(r'\[!(IMPORTANT|NOTE|TIP|WARNING|CAUTION)\]', '', content).strip()
            html_parts.append(f'<blockquote><p>{format_inline(content)}</p></blockquote>')
            continue
        
        # Unordered list
        if stripped.startswith('- ') or stripped.startswith('* '):
            list_items = []
            while i < len(lines) and (lines[i].strip().startswith('- ') or lines[i].strip().startswith('* ')):
                list_items.append(lines[i].strip()[2:])
                i += 1
            html_parts.append('<ul>' + ''.join(f'<li>{format_inline(item)}</li>' for item in list_items) + '</ul>')
            continue
        
        # Empty line
        if stripped == '':
            i += 1
            continue
        
        # Regular paragraph
        html_parts.append(f'<p>{format_inline(stripped)}</p>')
        i += 1
    
    if in_table:
        html_parts.append('</table>')
    
    return '\n'.join(html_parts)


def format_inline(text):
    """Format inline markdown: bold, code, etc."""
    # Code
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    # Bold
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    # Italic
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<em>\1</em>', text)
    return text


def main():
    os.makedirs(r'e:\University\Year 3 -2\DA2\CODE\report', exist_ok=True)
    
    for report in REPORTS:
        with open(report['md_path'], 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        html_content = md_to_html(md_content)
        full_html = HTML_TEMPLATE.format(title=report['title'], content=html_content)
        
        with open(report['html_path'], 'w', encoding='utf-8') as f:
            f.write(full_html)
        
        print(f"Generated: {report['html_path']}")
    
    print("\nDone! Open the HTML files in browser, then Ctrl+P to save as PDF.")


if __name__ == '__main__':
    main()
