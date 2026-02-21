"""
app/api/routes.py — FastAPI REST API

Endpoints:
  POST /upload-resume   — Upload resume file (PDF/DOCX/TXT) → store in ChromaDB
  POST /ingest-resume   — Store resume text in ChromaDB (legacy)
  POST /analyze         — Run full LangGraph pipeline
  GET  /health          — Health check
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.core.resume_store import ingest_resume
from app.core.file_parser import extract_text_from_file
from app.agents.graph import run_pipeline

app = FastAPI(
    title="AI Job Application Co-Pilot",
    description="LangGraph-powered job application assistant",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response Models ──────────────────────────────────────────────────

class ResumeRequest(BaseModel):
    resume_text: str


class AnalyzeRequest(BaseModel):
    job_description: str
    resume_text: str = ""  # Optional: if already ingested, leave empty


class AnalyzeResponse(BaseModel):
    job_title: str
    company: str
    match_score: int
    matched_skills: list
    missing_skills: list
    cover_letter: str
    email_draft: str
    interview_questions: list
    prep_tips: str


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "AI Job Co-Pilot"}


@app.post("/upload-resume")
async def upload_resume_file(file: UploadFile = File(...)):
    """
    Upload a resume file (PDF, DOCX, or TXT).
    Automatically extracts text and stores embeddings in ChromaDB.
    """
    allowed_types = {
        "application/pdf": ".pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "text/plain": ".txt",
    }

    # Validate file type by content-type or extension
    filename = file.filename or ""
    ext = filename.split(".")[-1].lower() if "." in filename else ""
    if ext not in ["pdf", "docx", "txt"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a PDF, DOCX, or TXT file."
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        resume_text = extract_text_from_file(file_bytes, filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if len(resume_text.strip()) < 50:
        raise HTTPException(
            status_code=422,
            detail="Could not extract enough text from the file. Try a different format."
        )

    result = ingest_resume(resume_text)
    return {
        "message": result,
        "filename": filename,
        "characters_extracted": len(resume_text),
        "preview": resume_text[:300] + "..." if len(resume_text) > 300 else resume_text,
    }


@app.post("/ingest-resume")
def ingest_resume_endpoint(request: ResumeRequest):
    """
    Upload and embed your resume into ChromaDB.
    Call this once. After this, all /analyze calls will use your stored resume.
    """
    if not request.resume_text.strip():
        raise HTTPException(status_code=400, detail="Resume text cannot be empty.")

    result = ingest_resume(request.resume_text)
    return {"message": result}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_job(request: AnalyzeRequest):
    """
    Run the full LangGraph pipeline on a job description.
    Returns skill match, cover letter, email, and interview prep.
    """
    if not request.job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")

    final_state = run_pipeline(
        job_description=request.job_description,
        resume_text=request.resume_text,
    )

    if final_state.get("error"):
        raise HTTPException(status_code=500, detail=final_state["error"])

    return AnalyzeResponse(
        job_title=final_state["parsed_jd"].get("job_title", ""),
        company=final_state["parsed_jd"].get("company", ""),
        match_score=final_state["match_score"],
        matched_skills=final_state["matched_skills"],
        missing_skills=final_state["missing_skills"],
        cover_letter=final_state["cover_letter"],
        email_draft=final_state["email_draft"],
        interview_questions=final_state["interview_questions"],
        prep_tips=final_state["prep_tips"],
    )
