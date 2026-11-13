import os
import xml.sax.saxutils as saxutils
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    Paragraph, Spacer, Table, TableStyle, Preformatted, HRFlowable, KeepTogether
)
from reportlab.pdfgen import canvas

# Palette definition
NAVY = colors.HexColor("#0f172a")
BLUE = colors.HexColor("#1e40af")
LIGHT_BLUE = colors.HexColor("#eff6ff")
BORDER_BLUE = colors.HexColor("#bfdbfe")

GREEN = colors.HexColor("#15803d")
LIGHT_GREEN = colors.HexColor("#f0fdf4")
BORDER_GREEN = colors.HexColor("#bbf7d0")

AMBER = colors.HexColor("#b45309")
LIGHT_AMBER = colors.HexColor("#fffbeb")
BORDER_AMBER = colors.HexColor("#fef3c7")

RED = colors.HexColor("#b91c1c")
LIGHT_RED = colors.HexColor("#fef2f2")
BORDER_RED = colors.HexColor("#fecaca")

GRAY_BG = colors.HexColor("#f8fafc")
GRAY_BORDER = colors.HexColor("#e2e8f0")
TEXT_DARK = colors.HexColor("#1e293b")
TEXT_MUTED = colors.HexColor("#475569")
WHITE = colors.HexColor("#ffffff")

# Base styles
base_styles = getSampleStyleSheet()

styles = {
    'CoverTitle': ParagraphStyle(
        'CoverTitle',
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=30,
        textColor=NAVY,
        alignment=0,
        spaceAfter=10
    ),
    'CoverSubtitle': ParagraphStyle(
        'CoverSubtitle',
        fontName='Helvetica',
        fontSize=13,
        leading=18,
        textColor=BLUE,
        alignment=0,
        spaceAfter=20
    ),
    'CoverMeta': ParagraphStyle(
        'CoverMeta',
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=TEXT_MUTED,
        spaceAfter=6
    ),
    'ChapterTitle': ParagraphStyle(
        'ChapterTitle',
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=NAVY,
        spaceBefore=18,
        spaceAfter=10,
        keepWithNext=True
    ),
    'SectionTitle': ParagraphStyle(
        'SectionTitle',
        fontName='Helvetica-Bold',
        fontSize=13.5,
        leading=17,
        textColor=BLUE,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    ),
    'SubsectionTitle': ParagraphStyle(
        'SubsectionTitle',
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=TEXT_DARK,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    ),
    'Body': ParagraphStyle(
        'Body',
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.8,
        textColor=TEXT_DARK,
        spaceBefore=2,
        spaceAfter=5
    ),
    'BodyBold': ParagraphStyle(
        'BodyBold',
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13.8,
        textColor=TEXT_DARK,
        spaceBefore=2,
        spaceAfter=5
    ),
    'Bullet': ParagraphStyle(
        'Bullet',
        fontName='Helvetica',
        fontSize=9.2,
        leading=13.2,
        textColor=TEXT_DARK,
        leftIndent=14,
        firstLineIndent=-10,
        spaceBefore=1,
        spaceAfter=3
    ),
    'Code': ParagraphStyle(
        'Code',
        fontName='Courier',
        fontSize=7.8,
        leading=10.2,
        textColor=NAVY
    ),
    'CalloutHeading': ParagraphStyle(
        'CalloutHeading',
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=NAVY,
        spaceAfter=4
    ),
    'CalloutBody': ParagraphStyle(
        'CalloutBody',
        fontName='Helvetica',
        fontSize=8.8,
        leading=12.5,
        textColor=TEXT_DARK
    ),
    'TableHeader': ParagraphStyle(
        'TableHeader',
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=WHITE
    ),
    'TableCell': ParagraphStyle(
        'TableCell',
        fontName='Helvetica',
        fontSize=8.2,
        leading=11,
        textColor=TEXT_DARK
    ),
    'TableCellBold': ParagraphStyle(
        'TableCellBold',
        fontName='Helvetica-Bold',
        fontSize=8.2,
        leading=11,
        textColor=TEXT_DARK
    ),
    'TableCellMono': ParagraphStyle(
        'TableCellMono',
        fontName='Courier',
        fontSize=8.0,
        leading=10.5,
        textColor=NAVY
    )
}

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber > 1:
            self.saveState()
            self.setFont('Helvetica', 8)
            self.setFillColor(colors.HexColor('#64748b'))
            self.setStrokeColor(colors.HexColor('#cbd5e1'))
            self.setLineWidth(0.5)
            # Header
            self.drawString(40, 755, 'ES 335 Machine Learning | Macro Stress Testing & Credit Risk Analytics | Master Defense Guide')
            self.line(40, 749, 572, 749)
            # Footer
            self.line(40, 38, 572, 38)
            self.drawString(40, 26, 'Author: Kunal Kashyap (IIT Gandhinagar) — ES 335 Master Oral & Technical Defense')
            page_text = f'Page {self._pageNumber} of {page_count}'
            self.drawRightString(572, 26, page_text)
            self.restoreState()

def esc(text):
    """Safely escapes text for ReportLab XML parsing."""
    if not isinstance(text, str):
        text = str(text)
    return saxutils.escape(text)

def p(text, style_name='Body'):
    return Paragraph(text, styles[style_name])

def bullet(text):
    return Paragraph(f"&bull; {text}", styles['Bullet'])

def num_bullet(num, text):
    return Paragraph(f"<b>{num}.</b> {text}", styles['Bullet'])

def callout(title, text_lines, kind='note'):
    """Generates an alert / callout box (note, tip, warning, intuition)."""
    if kind == 'warning':
        bg = LIGHT_RED
        border = BORDER_RED
        title_color = RED
    elif kind == 'tip' or kind == 'green':
        bg = LIGHT_GREEN
        border = BORDER_GREEN
        title_color = GREEN
    elif kind == 'amber' or kind == 'caution':
        bg = LIGHT_AMBER
        border = BORDER_AMBER
        title_color = AMBER
    else: # note / blue
        bg = LIGHT_BLUE
        border = BORDER_BLUE
        title_color = BLUE

    content = []
    if title:
        content.append(Paragraph(f"<font color='{title_color.hexval()}'><b>{title}</b></font>", styles['CalloutHeading']))
    
    if isinstance(text_lines, str):
        text_lines = [text_lines]
    for line in text_lines:
        content.append(Paragraph(line, styles['CalloutBody']))

    t = Table([[content]], colWidths=[532])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg),
        ('BOX', (0,0), (-1,-1), 1, border),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    return KeepTogether([t, Spacer(1, 6)])

def code_block(code_text):
    """Formats code cleanly in a slate box."""
    clean_lines = [saxutils.escape(line) for line in code_text.strip().splitlines()]
    escaped_code = "<br/>".join(clean_lines)
    p_code = Paragraph(f"<font name='Courier' size='7.5' color='#0f172a'>{escaped_code}</font>", styles['Code'])
    
    t = Table([[p_code]], colWidths=[532])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f1f5f9')),
        ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    return KeepTogether([t, Spacer(1, 6)])

def divider():
    return HRFlowable(width="100%", thickness=0.5, color=GRAY_BORDER, spaceBefore=10, spaceAfter=10)
