"""
Script to generate Prompts Dossier PDF from markdown.
"""
from pathlib import Path
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
import html
import re

def _esc(t: str) -> str:
    return html.escape(t) if t else ""

def generate_dossier_pdf():
    md_path = Path("docs/prompts_dossier.md")
    pdf_path = Path("Prompts_Dossier.pdf")
    
    if not md_path.exists():
        print("docs/prompts_dossier.md not found!")
        return

    content = md_path.read_text(encoding="utf-8")
    
    styles = getSampleStyleSheet()
    h1_style = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=22, spaceAfter=14, textColor=colors.HexColor("#1A1A2E"))
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=16, spaceBefore=12, spaceAfter=8, textColor=colors.HexColor("#2C2C54"))
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=11, leading=16, spaceAfter=8)
    code_style = ParagraphStyle("Code", parent=styles["Code"], fontSize=9, textColor=colors.HexColor("#2E8B57"), leftIndent=20, spaceAfter=8)

    doc = SimpleDocTemplate(str(pdf_path), pagesize=LETTER, rightMargin=inch, leftMargin=inch, topMargin=inch, bottomMargin=inch)
    story = []
    
    lines = content.split('\n')
    in_code_block = False
    code_buffer = []

    for line in lines:
        line = line.rstrip()
        
        # Handle code blocks
        if line.startswith("```"):
            if in_code_block:
                story.append(Paragraph(_esc("\n".join(code_buffer)).replace("\n", "<br/>"), code_style))
                story.append(Spacer(1, 0.1*inch))
                in_code_block = False
                code_buffer = []
            else:
                in_code_block = True
            continue
            
        if in_code_block:
            code_buffer.append(line)
            continue
            
        if not line:
            continue
            
        if line == "---":
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CCCCEE"), spaceAfter=12, spaceBefore=12))
            continue
            
        if line.startswith("# "):
            story.append(Paragraph(_esc(line[2:]), h1_style))
        elif line.startswith("## "):
            story.append(Paragraph(_esc(line[3:]), h2_style))
        elif line.startswith("**"):
            # Bold rendering (basic)
            text = line.replace("**", "")
            story.append(Paragraph(f"<b>{_esc(text)}</b>", body_style))
        else:
            story.append(Paragraph(_esc(line), body_style))

    doc.build(story)
    print(f"Generated: {pdf_path.absolute()}")

if __name__ == "__main__":
    generate_dossier_pdf()
