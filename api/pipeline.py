import base64
import json
import pymupdf
import pymupdf4llm
import ollama
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from datetime import date, datetime

# Config

OCR_MODEL = "glm-ocr"
LLM_MODEL_LIGHT = "qwen3:4b"
LLM_MODEL = "qwen3:8b"
PIXMAP_RESOLUTION_FACTOR = 1.6

# Schema

class Doctor(BaseModel):
    title: str | None = Field(None, description="Academic or medical title, e.g. 'Dr.', 'Prim.', 'Univ.-Prof. Dr.'")
    first_name: str | None = Field(None, description="First name of the doctor")
    last_name: str | None = Field(None, description="Last name of the doctor")
    specialty: str | None = Field(None, description="Medical specialty, e.g. 'Allgemeinmedizin', 'Innere Medizin', 'Dermatologie'")
    practice_name: str | None = Field(None, description="Name of the medical practice if explicitly stated, not the doctor's name")
    practice_address: str | None = Field(None, description="Full address including street, number, postal code and city")
    phone: str | None = Field(None, description="Phone number of the practice")
    email: str | None = Field(None, description="Email address of the practice")
    uid: str | None = Field(None, description="Austrian VAT number, format: ATU + 8 digits, e.g. 'ATU12345678'")

class Patient(BaseModel):
    first_name: str | None = Field(None, description="First name of the patient")
    last_name: str | None = Field(None, description="Last name of the patient")
    date_of_birth: date | None = Field(None, description="Date of birth, format DD.MM.YYYY")
    social_security_number: str | None = Field(None, description="Austrian SVNR, format: '1234 150378'")

    @field_validator("date_of_birth", mode="before")
    @classmethod
    def parse_date_of_birth(cls, v):
        if v is None or isinstance(v, date):
            return v
        for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y"):
            try:
                parsed = datetime.strptime(v, fmt).date()
                if 1900 <= parsed.year <= date.today().year:
                    return parsed
            except ValueError:
                continue
        return None

class Treatment(BaseModel):
    code:        str   | None = Field(None, description="Medical service code e.g. 'Ö1', 'L100'. Use null if not present.")
    description: str   | None = Field(None, description="Description of the medical service performed")
    amount:      float | None = Field(None, description="Cost of this treatment as decimal. Never include the total sum row.")

class Invoice(BaseModel):
    invoice_date: date | None = Field(None, description="Date the invoice was issued, format DD.MM.YYYY")
    invoice_number: str | None = Field(None, description="Invoice reference number, e.g. 'RE-1234-2025'")
    document_type: str | None = Field(None, description="Typically one of: 'Rechnung', 'Honorarnote', 'Arzthonorar', 'Privatrechnung'")
    doctor: Doctor | None = Field(None, description="Details of the issuing doctor")
    patient: Patient | None = Field(None, description="Details of the patient")
    diagnosis: str | None = Field(None, description="Medical diagnosis, include ICD-10 code if present, e.g. 'Gastritis (K29)'")
    treatments: list[Treatment] | None = Field(None, description="List of treatments. Never include the total sum row as a treatment item.")
    total_amount: float | None = Field(None, description="Total invoice amount as decimal. ALWAYS take this value directly from the invoice. NEVER calculate it yourself.")
    payment_method: str | None = Field(None, description="ONLY 'cash' or 'bank transfer'. No other text, no IBAN, no account details.")
    iban: str | None = Field(None, description="IBAN if payment method is bank transfer")

    @field_validator("invoice_date", mode="before")
    @classmethod
    def parse_invoice_date(cls, v):
        if v is None or isinstance(v, date):
            return v
        for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y"):
            try:
                parsed = datetime.strptime(v, fmt).date()
                if 2000 <= parsed.year <= 2100:
                    return parsed
            except ValueError:
                continue
        return None

# Validation

SUM_KEYWORDS = ["gesamt", "gesamtbetrag", "total", "honorar gesamt", "summe", "sunne"]

def validate_invoice(invoice: Invoice) -> dict:
    flags = {}

    flags["invoice_date_missing"] = invoice.invoice_date is None
    flags["invoice_number_missing"] = invoice.invoice_number is None
    flags["diagnosis_missing"] = invoice.diagnosis is None
    flags["treatments_missing"] = not invoice.treatments
    flags["total_amount_missing"] = invoice.total_amount is None

    flags["doctor_first_name_missing"] = invoice.doctor is None or invoice.doctor.first_name is None
    flags["doctor_last_name_missing"] = invoice.doctor is None or invoice.doctor.last_name is None
    flags["doctor_specialty_missing"] = invoice.doctor is None or invoice.doctor.specialty is None
    flags["doctor_practice_address_missing"] = invoice.doctor is None or invoice.doctor.practice_address is None

    flags["patient_first_name_missing"] = invoice.patient is None or invoice.patient.first_name is None
    flags["patient_last_name_missing"] = invoice.patient is None or invoice.patient.last_name is None
    flags["patient_date_of_birth_missing"] = invoice.patient is None or invoice.patient.date_of_birth is None
    flags["patient_social_security_number_missing"] = invoice.patient is None or invoice.patient.social_security_number is None

    if invoice.treatments and invoice.total_amount:
        amounts   = [t.amount for t in invoice.treatments if t.amount]
        calc      = sum(amounts)
        last      = amounts[-1]
        last_desc = invoice.treatments[-1].description.lower() if invoice.treatments[-1].description else ""

        flags["total_mismatch"] = abs(calc - invoice.total_amount) > 0.05
        flags["last_treatment_equals_sum_of_others"] = (calc - last) == last
        flags["last_treatment_looks_like_sum"] = any(kw in last_desc for kw in SUM_KEYWORDS)
    else:
        flags["total_mismatch"] = False
        flags["last_treatment_equals_sum_of_others"] = False
        flags["last_treatment_looks_like_sum"] = False

    score = round(1 - sum(flags.values()) / len(flags), 2)
    return {"score": score, "flags": flags}

# Text extraction

TEXT_THRESHOLD = 30

def detect_pdf_type(pdf_bytes: bytes) -> str:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return "digital" if len(text) >= TEXT_THRESHOLD else "scan"

def extract_text_digital(pdf_bytes: bytes) -> str:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    md = pymupdf4llm.to_markdown(doc)
    doc.close()
    return md

def extract_text_scan(pdf_bytes: bytes) -> str:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    mat = pymupdf.Matrix(PIXMAP_RESOLUTION_FACTOR, PIXMAP_RESOLUTION_FACTOR)
    pix = page.get_pixmap(matrix=mat)
    img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
    doc.close()

    response = ollama.chat(
        model=OCR_MODEL,
        messages=[{
            "role": "user",
            "content": "Extrahiere mir den ganzen Text",
            "images": [img_b64]
        }],
        think=False,
        options={"temperature": 0}
    )
    return response.message.content

# LLM extraction

SYSTEM_PROMPT = f"""You are an expert in extracting information from Austrian private doctor's invoices (Wahlarztrechnungen).
Rules:
- If a field is not present in the invoice, use null
- NEVER calculate the total amount yourself. Always take it directly from the invoice
- NEVER include the total sum row as a treatment item
- If a treatment description looks like a misspelling of "Summe" or "Gesamt" (e.g. 'Sunne'), treat it as the total amount instead
- Date of birth format: DD.MM.YYYY as it appears on the invoice, e.g. '22.07.1965'
- Invoice date format: DD.MM.YYYY as it appears on the invoice
- payment_method: ONLY 'cash' or 'bank transfer', nothing else

Extract the following JSON structure:
{json.dumps(Invoice.model_json_schema(), indent=2)}
"""

def extract_structured_data(raw_text: str, llm_model: str = LLM_MODEL_LIGHT) -> tuple[Invoice | None, dict]:
    response = ollama.chat(
        model=llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Here is the invoice text:\n\n{raw_text}"}
        ],
        think=False,
        format="json",
        options={"temperature": 0}
    )
    print(llm_model)
    try:
        invoice = Invoice.model_validate_json(response.message.content)
    except Exception as e:
        print(f"Validation error: {e}")
        return None, {}

    validation = validate_invoice(invoice)
    return invoice, validation

# Main pipeline

def process_pdf(pdf_bytes: bytes, llm_model: str | None = None) -> tuple[Invoice | None, dict, str]:
    pdf_type = detect_pdf_type(pdf_bytes)

    if pdf_type == "digital":
        raw_text = extract_text_digital(pdf_bytes)
    else:
        raw_text = extract_text_scan(pdf_bytes)

    invoice, validation = extract_structured_data(raw_text, llm_model)
    return invoice, validation, pdf_type