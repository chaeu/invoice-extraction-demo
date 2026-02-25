import json
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from typing import Optional
from pipeline import process_pdf, LLM_MODEL_LIGHT

METADATA_FILE = Path("../invoices/metadata.json")

app = FastAPI(title="Invoice Extraction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/extract")
async def extract_invoice(
    file: UploadFile = File(...),
    model: Optional[str] = Form(default=None)
    
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    effective_model = model or LLM_MODEL_LIGHT
    pdf_bytes = await file.read()
    invoice, validation, pdf_type = process_pdf(pdf_bytes, llm_model=effective_model)

    if invoice is None:
        raise HTTPException(status_code=422, detail="Could not extract invoice data")

    ground_truth = None
    if METADATA_FILE.exists():
        with open(METADATA_FILE, encoding="utf-8") as f:
            metadata = json.load(f)
        ground_truth = next((m for m in metadata if m["file"] == file.filename), None)

    return {
        "filename": file.filename,
        "pdf_type": pdf_type,
        "model": effective_model,
        "invoice": invoice.model_dump(mode="json"),
        "validation": validation,
        "ground_truth": ground_truth,
    }

@app.get("/health")
def health():
    return {"status": "ok"}