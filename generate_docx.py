import os
import re
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def set_table_borders(table, color="CCCCCC", sz="4", val="single"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:left w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:right w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def build_docx():
    doc = docx.Document()

    # Set page margins to 1 inch (25.4 mm)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        # Header / footer
        footer = section.footer
        f_p = footer.paragraphs[0]
        f_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        # Add simple text in footer
        f_run = f_p.add_run("ITM University, Gwalior | B.Tech CSE Project")
        f_run.font.name = "Times New Roman"
        f_run.font.size = Pt(9)
        f_run.font.color.rgb = RGBColor(140, 140, 140)

    # Set base Normal style to Times New Roman, 12pt
    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Times New Roman"
    normal_style.font.size = Pt(12)
    normal_style.font.color.rgb = RGBColor(30, 30, 30)

    with open("Road_Damage_Detection_Project_Report.md", "r", encoding="utf-8") as f:
        md_text = f.read()

    lines = md_text.split("\n")
    i = 0
    in_code_block = False
    code_lines = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Handle code blocks
        if stripped.startswith("```"):
            if in_code_block:
                # Flush code block
                table = doc.add_table(rows=1, cols=1)
                table.alignment = WD_TABLE_ALIGNMENT.CENTER
                cell = table.cell(0, 0)
                set_cell_background(cell, "F8F9FA")
                set_cell_margins(cell, top=120, bottom=120, left=180, right=180)
                p = cell.paragraphs[0]
                p.paragraph_format.space_before = Pt(2)
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.line_spacing = 1.15
                code_text = "\n".join(code_lines)
                c_run = p.add_run(code_text)
                c_run.font.name = "Courier New"
                c_run.font.size = Pt(9.5)
                c_run.font.color.rgb = RGBColor(40, 40, 40)
                set_table_borders(table, color="D1D5DB", sz="4")
                doc.add_paragraph() # spacer
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
                code_lines = []
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # Page breaks
        if "\\pagebreak" in stripped:
            doc.add_page_break()
            i += 1
            continue

        # Horizontal rule
        if stripped == "---":
            # Add a subtle horizontal separator or spacing
            i += 1
            continue

        # Markdown tables
        if stripped.startswith("|") and stripped.endswith("|") and "|" in stripped[1:-1]:
            # Collect all table lines
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                t_row = lines[i].strip()
                # Check if it is separator row like | :--- | :---: |
                cells = [c.strip() for c in t_row.split("|")[1:-1]]
                if not all(re.match(r"^:?-+:?$", c) for c in cells):
                    table_lines.append(cells)
                i += 1
            
            if table_lines:
                num_cols = max(len(row) for row in table_lines)
                doc_table = doc.add_table(rows=len(table_lines), cols=num_cols)
                doc_table.alignment = WD_TABLE_ALIGNMENT.CENTER
                set_table_borders(doc_table, color="94A3B8", sz="4")

                for r_idx, row_cells in enumerate(table_lines):
                    row = doc_table.rows[r_idx]
                    is_header = (r_idx == 0)
                    for c_idx in range(num_cols):
                        cell = row.cells[c_idx]
                        set_cell_margins(cell, top=100, bottom=100, left=140, right=140)
                        val = row_cells[c_idx] if c_idx < len(row_cells) else ""
                        if is_header:
                            set_cell_background(cell, "E2E8F0")
                        else:
                            if r_idx % 2 == 1:
                                set_cell_background(cell, "F8FAFC")
                            else:
                                set_cell_background(cell, "FFFFFF")
                        
                        p = cell.paragraphs[0]
                        p.paragraph_format.space_before = Pt(2)
                        p.paragraph_format.space_after = Pt(2)
                        p.paragraph_format.line_spacing = 1.15
                        
                        # Parse simple bolding in cells
                        parts = re.split(r"(\*\*.*?\*\*)", val)
                        for part in parts:
                            if part.startswith("**") and part.endswith("**"):
                                r = p.add_run(part[2:-2])
                                r.bold = True
                            else:
                                r = p.add_run(part)
                            r.font.name = "Times New Roman"
                            r.font.size = Pt(10) if is_header else Pt(9.5)
                            if is_header:
                                r.bold = True
                                r.font.color.rgb = RGBColor(15, 23, 42)
                p_after = doc.add_paragraph()
                p_after.paragraph_format.space_before = Pt(4)
                p_after.paragraph_format.space_after = Pt(4)
            continue

        # Headings
        if stripped.startswith("# "):
            title_text = stripped[2:].strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(18)
            p.paragraph_format.space_after = Pt(14)
            run = p.add_run(title_text)
            run.bold = True
            run.font.name = "Times New Roman"
            run.font.size = Pt(22)
            run.font.color.rgb = RGBColor(15, 23, 42)
            i += 1
            continue

        if stripped.startswith("## "):
            h2_text = stripped[3:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(22)
            p.paragraph_format.space_after = Pt(8)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(h2_text)
            run.bold = True
            run.font.name = "Times New Roman"
            run.font.size = Pt(15)
            run.font.color.rgb = RGBColor(15, 23, 42)
            i += 1
            continue

        if stripped.startswith("### "):
            h3_text = stripped[4:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(h3_text)
            run.bold = True
            run.font.name = "Times New Roman"
            run.font.size = Pt(12.5)
            run.font.color.rgb = RGBColor(30, 41, 59)
            i += 1
            continue

        if stripped.startswith("#### "):
            h4_text = stripped[5:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(h4_text)
            run.bold = True
            run.font.name = "Times New Roman"
            run.font.size = Pt(11.5)
            run.font.color.rgb = RGBColor(51, 65, 85)
            i += 1
            continue

        # Bullet list
        if stripped.startswith("- ") or stripped.startswith("* "):
            bullet_text = stripped[2:].strip()
            p = doc.add_paragraph(style='List Bullet')
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.15
            
            # Format text in bullet
            format_runs(p, bullet_text)
            i += 1
            continue

        # Numbered list
        num_match = re.match(r"^(\d+)\.\s+(.*)", stripped)
        if num_match:
            num_text = num_match.group(2).strip()
            p = doc.add_paragraph(style='List Number')
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.15
            format_runs(p, num_text)
            i += 1
            continue

        # Blockquote
        if stripped.startswith("> "):
            quote_text = stripped[2:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.4)
            p.paragraph_format.right_indent = Inches(0.4)
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(8)
            p.paragraph_format.line_spacing = 1.15
            format_runs(p, quote_text, italic_default=True)
            i += 1
            continue

        # Standard paragraph
        if stripped:
            # Special centering check for cover page or centered text
            is_centered = False
            if stripped.startswith("<div align=\"center\">") or stripped.endswith("</div>"):
                stripped = re.sub(r"</?div.*?>", "", stripped).strip()
                is_centered = True
            
            p = doc.add_paragraph()
            if is_centered:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.line_spacing = 1.2
            format_runs(p, stripped)
        
        i += 1

    doc.save("Road_Damage_Detection_Project_Report.docx")
    print("SUCCESS: Road_Damage_Detection_Project_Report.docx generated successfully!")

def format_runs(p, text, italic_default=False):
    # Regex split for **bold**, *italic*, `code`
    # Replace <br> with newline
    text = text.replace("<br>", "\n").replace("&nbsp;", " ")
    tokens = re.split(r"(\*\*.*?\*\*|\*.*?\*|`.*?`)", text)
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**"):
            run = p.add_run(tok[2:-2])
            run.bold = True
            if italic_default:
                run.italic = True
        elif tok.startswith("*") and tok.endswith("*"):
            run = p.add_run(tok[1:-1])
            run.italic = True
        elif tok.startswith("`") and tok.endswith("`"):
            run = p.add_run(tok[1:-1])
            run.font.name = "Courier New"
            run.font.size = Pt(10)
        else:
            run = p.add_run(tok)
            if italic_default:
                run.italic = True
        run.font.name = "Times New Roman"

if __name__ == "__main__":
    build_docx()
