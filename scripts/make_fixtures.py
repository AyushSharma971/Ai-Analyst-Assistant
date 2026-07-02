"""Generate synthetic submission fixtures (one per format family).

We have no real submissions yet, so this builds representative samples to prove
every Document AI path end-to-end:

  - application.pdf        digital (text-layer) PDF
  - loss_run.xlsx          Excel loss run / exposure schedule
  - narrative.docx         Word underwriting narrative (with a table)
  - scanned_acord.pdf      image-only PDF (no text layer -> exercises OCR routing)
  - submission_email.eml   email with the digital PDF attached (recursive parse)

Run:  python scripts/make_fixtures.py   ->   writes into ./samples/
"""

from __future__ import annotations

import os

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")

APP_TEXT = """ACME HEALTH SYSTEMS — D&O Insurance Application

Insured Name: Acme Health Systems, Inc.
Industry: Healthcare (Hospitals & Clinics)
Annual Revenue: $125,000,000
Employee Count: 1,200
Year Founded: 1998
Multi-Factor Authentication Enabled: Yes
Prior Claims (last 5 years): 0

The applicant operates a regional network of outpatient clinics and one acute
care facility. Coverage requested: $10M D&O, retention $250,000.
"""

NARRATIVE_PARAS = [
    "Underwriting Narrative — Acme Health Systems, Inc.",
    "Acme Health Systems is a healthcare provider headquartered in Ohio, "
    "operating since 1998 with approximately 1,200 employees.",
    "Financials: trailing twelve-month revenue of $125M, with stable EBITDA "
    "margins. No material litigation is currently pending.",
    "Risk controls: the insured has confirmed multi-factor authentication is "
    "enabled across all administrative systems.",
]

NARRATIVE_TABLE = [
    ["Attribute", "Value"],
    ["Insured Name", "Acme Health Systems, Inc."],
    ["Annual Revenue", "$125,000,000"],
    ["Employee Count", "1,200"],
    ["Prior Claims", "0"],
]

LOSS_RUN = [
    ["Policy Year", "Claim #", "Description", "Paid", "Reserved", "Status"],
    [2021, "CLM-0001", "Employment practices dispute", 0, 0, "Closed"],
    [2022, "CLM-0002", "Regulatory inquiry (no payment)", 0, 0, "Closed"],
    [2023, None, "No claims", 0, 0, "—"],
]


def _ensure_dir() -> None:
    os.makedirs(SAMPLES_DIR, exist_ok=True)


def make_digital_pdf(path: str) -> None:
    import fitz  # PyMuPDF

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), APP_TEXT, fontsize=11)
    doc.save(path)
    doc.close()


def make_scanned_pdf(path: str) -> None:
    """Image-only PDF: render text to a raster image, embed it as a full-page
    picture. No text layer -> our scanned-page detector must route it to OCR."""
    from PIL import Image, ImageDraw

    w, h = 1000, 1300
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    lines = [
        "ACORD 125 — Commercial Insurance Application (SCANNED)",
        "",
        "Named Insured: Acme Health Systems, Inc.",
        "Annual Revenue: $125,000,000",
        "Employees: 1200",
        "MFA Enabled: YES",
    ]
    y = 60
    for ln in lines:
        draw.text((60, y), ln, fill="black")
        y += 60

    import io as _io

    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    png = buf.getvalue()

    import fitz

    doc = fitz.open()
    page = doc.new_page(width=w, height=h)
    page.insert_image(fitz.Rect(0, 0, w, h), stream=png)
    doc.save(path)
    doc.close()


def make_excel(path: str) -> None:
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Loss Run"
    for row in LOSS_RUN:
        ws.append(row)
    wb.save(path)


def make_docx(path: str) -> None:
    import docx

    d = docx.Document()
    for i, p in enumerate(NARRATIVE_PARAS):
        if i == 0:
            d.add_heading(p, level=1)
        else:
            d.add_paragraph(p)
    t = d.add_table(rows=0, cols=2)
    for r in NARRATIVE_TABLE:
        cells = t.add_row().cells
        cells[0].text, cells[1].text = str(r[0]), str(r[1])
    d.save(path)


def make_rater(path: str) -> None:
    """A blank carrier rater: labels in column A, empty input cells in column B,
    and a FORMULA cell (premium = revenue * rate) to prove Safe Autofill preserves
    formulas when it writes the input cells."""
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Rater"
    ws["A1"] = "Carrier D&O Rater (BLANK TEMPLATE)"
    for cell, label in [
        ("A2", "Insured Name"),
        ("A3", "Annual Revenue"),
        ("A4", "Employee Count"),
        ("A5", "Industry"),
        ("A6", "MFA Enabled"),
        ("A7", "Prior Claims"),
    ]:
        ws[cell] = label
    # Formula that depends on an input cell (B3 = annual_revenue). Must survive.
    ws["A9"] = "Premium Estimate"
    ws["B9"] = "=B3*0.001"
    wb.save(path)


def make_email(path: str, attach_pdf: str) -> None:
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["From"] = "broker@example-brokerage.com"
    msg["To"] = "submissions@odysseygroup.com"
    msg["Subject"] = "New D&O Submission — Acme Health Systems"
    msg["Date"] = "Tue, 03 Jun 2026 09:15:00 -0400"
    msg.set_content(
        "Hi team,\n\nPlease find attached the D&O application for Acme Health "
        "Systems (revenue ~$125M, 1,200 employees). Target effective date 7/1.\n\n"
        "Thanks,\nJ. Broker"
    )
    if os.path.exists(attach_pdf):
        with open(attach_pdf, "rb") as fh:
            msg.add_attachment(
                fh.read(),
                maintype="application",
                subtype="pdf",
                filename=os.path.basename(attach_pdf),
            )
    with open(path, "wb") as fh:
        fh.write(bytes(msg))


def build_all() -> dict:
    """Generate all fixtures and return {label: path}."""
    _ensure_dir()
    paths = {
        "digital_pdf": os.path.join(SAMPLES_DIR, "application.pdf"),
        "excel": os.path.join(SAMPLES_DIR, "loss_run.xlsx"),
        "word": os.path.join(SAMPLES_DIR, "narrative.docx"),
        "scanned_pdf": os.path.join(SAMPLES_DIR, "scanned_acord.pdf"),
        "email": os.path.join(SAMPLES_DIR, "submission_email.eml"),
        "rater_template": os.path.join(SAMPLES_DIR, "rater_template.xlsx"),
    }
    make_digital_pdf(paths["digital_pdf"])
    make_excel(paths["excel"])
    make_docx(paths["word"])
    make_scanned_pdf(paths["scanned_pdf"])
    make_email(paths["email"], paths["digital_pdf"])
    make_rater(paths["rater_template"])
    return paths


if __name__ == "__main__":
    out = build_all()
    print("Wrote fixtures to", SAMPLES_DIR)
    for label, path in out.items():
        size = os.path.getsize(path) if os.path.exists(path) else 0
        print(f"  - {label:12s} {os.path.basename(path):24s} {size:>8d} bytes")
