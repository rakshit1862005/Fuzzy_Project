import os
import shutil
import docx
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

doc_path = r'd:\Fuzzy_Project\Fuzzy_Project\report\Glass_Box_Insight_Project_Report_Formatted.docx'
final_doc_path = r'd:\Fuzzy_Project\Fuzzy_Project\report\Glass_Box_Insight_Project_Report.docx'

doc = docx.Document(doc_path)

# Set 1-inch margins on all sections
for section in doc.sections:
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)

# Set Normal style
normal_style = doc.styles['Normal']
normal_style.font.name = 'Times New Roman'
normal_style.font.size = Pt(12)
normal_style.font.color.rgb = RGBColor(0, 0, 0)
normal_style.paragraph_format.line_spacing = 1.5
normal_style.paragraph_format.space_after = Pt(4)
normal_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

# Style headings
heading_styles = {
    'Heading 1': (Pt(18), True, RGBColor(0, 0, 0), Pt(16), Pt(6)),
    'Heading 2': (Pt(14), True, RGBColor(0, 0, 0), Pt(12), Pt(4)),
    'Heading 3': (Pt(12), True, RGBColor(0, 0, 0), Pt(8), Pt(2)),
    'Title': (Pt(20), True, RGBColor(0, 0, 0), Pt(12), Pt(12)),
    'Subtitle': (Pt(13), False, RGBColor(0, 0, 0), Pt(4), Pt(12)),
}

for name, (size, bold, color, before, after) in heading_styles.items():
    if name in doc.styles:
        st = doc.styles[name]
        st.font.name = 'Times New Roman'
        st.font.size = size
        st.font.bold = bold
        st.font.color.rgb = color
        st.paragraph_format.space_before = before
        st.paragraph_format.space_after = after
        st.paragraph_format.line_spacing = 1.15

# Front matter headings to center
front_matter_titles = [
    'PUNJAB ENGINEERING COLLEGE',
    '(DEEMED TO BE UNIVERSITY)',
    'DEPARTMENT OF ELECTRICAL ENGINEERING',
    'PROJECT REPORT',
    'COURSE PROJECT REPORT',
    'EXPLAINING NEURAL NETWORK DECISIONS',
    'DIABETES DIAGNOSIS USING FUZZY LOGIC RULES',
    'GLASS-BOX INSIGHT',
    'CERTIFICATE',
    "CANDIDATE'S DECLARATION",
    'ACKNOWLEDGEMENT',
    'ABSTRACT',
    'LIST OF ABBREVIATIONS',
    'TABLE OF CONTENTS',
    'CONTENTS',
    'LIST OF FIGURES',
    'LIST OF TABLES'
]

for p in doc.paragraphs:
    style_name = p.style.name if p.style else ''
    text_strip = p.text.strip()
    text_upper = text_strip.upper().replace('’', "'")
    
    # Strip any stray Pandoc macro residuals from headings
    import re
    if re.match(r'^-2\.5ex\s+1\.2ex\s+', text_strip):
        clean_text = re.sub(r'^-2\.5ex\s+1\.2ex\s+', '', text_strip)
        p.text = clean_text
        text_strip = clean_text
        text_upper = text_strip.upper().replace('’', "'")
        style_name = 'Heading 2'
    elif re.match(r'^-2\.0ex\s+1\.0ex\s+', text_strip):
        clean_text = re.sub(r'^-2\.0ex\s+1\.0ex\s+', '', text_strip)
        p.text = clean_text
        text_strip = clean_text
        text_upper = text_strip.upper().replace('’', "'")
        style_name = 'Heading 3'
    elif re.match(r'^-1\.8ex\s+0\.8ex\s+', text_strip):
        clean_text = re.sub(r'^-1\.8ex\s+0\.8ex\s+', '', text_strip)
        p.text = clean_text
        text_strip = clean_text
        text_upper = text_strip.upper().replace('’', "'")
        style_name = 'Heading 3'
    elif re.match(r'^CHAPTER\s+[0-9]+', text_upper):
        clean_text = re.sub(r'^CHAPTER\s+([0-9]+)[:\.\s]+', r'\1. ', text_strip, flags=re.IGNORECASE)
        p.text = clean_text
        text_strip = clean_text
        text_upper = text_strip.upper().replace('’', "'")
        style_name = 'Heading 1'

    # Check if this paragraph contains an image
    has_image = any('graphic' in r._element.xml for r in p.runs)
    if has_image:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        continue

    # Check if this paragraph is a front matter title or chapter heading
    is_front_title = any(
        (fmt == text_upper or (fmt in text_upper and len(text_strip) < 100))
        for fmt in front_matter_titles
    )
    is_chap_heading = style_name.startswith('Heading 1') or (text_upper.startswith('CHAPTER ') and len(text_strip) < 80)
    
    if is_front_title or is_chap_heading:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.name = 'Times New Roman'
            r.font.color.rgb = RGBColor(0, 0, 0)
            r.font.bold = True
            if is_chap_heading or 'GLASS-BOX INSIGHT' in text_upper or 'EXPLAINING NEURAL NETWORK DECISIONS' in text_upper:
                r.font.size = Pt(16)
            elif 'PUNJAB ENGINEERING COLLEGE' in text_upper:
                r.font.size = Pt(15)
    elif style_name.startswith('Heading 2'):
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)
        for r in p.runs:
            r.font.name = 'Times New Roman'
            r.font.size = Pt(14)
            r.font.bold = True
            r.font.color.rgb = RGBColor(0, 0, 0)
    elif style_name.startswith('Heading 3'):
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(2)
        for r in p.runs:
            r.font.name = 'Times New Roman'
            r.font.size = Pt(12)
            r.font.bold = True
            r.font.color.rgb = RGBColor(0, 0, 0)
    else:
        # Standard body paragraph: strict justification
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        for r in p.runs:
            r.font.name = 'Times New Roman'
            if r.font.color and r.font.color.rgb != RGBColor(0, 0, 0):
                r.font.color.rgb = RGBColor(0, 0, 0)

# Format all tables: Center alignment, clean font
for table in doc.tables:
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.line_spacing = 1.15
                p.paragraph_format.space_after = Pt(2)
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(10)
                    r.font.color.rgb = RGBColor(0, 0, 0)

doc.save(doc_path)
shutil.copyfile(doc_path, final_doc_path)
print(f"Successfully formatted docx and saved to:\n  - {doc_path}\n  - {final_doc_path}")
