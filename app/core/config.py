"""
app/core/config.py — App configuration and shared state schema
"""
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")


# ─── LangGraph State ──────────────────────────────────────────────────────────

class AppState(BaseModel):
    """Shared state object that flows through the entire LangGraph workflow."""

    # Inputs
    job_description: str = ""
    resume_text: str = ""

    # Intermediate results
    parsed_jd: dict = {}          # title, company, skills_required, responsibilities
    matched_skills: list = []     # skills you have that match
    missing_skills: list = []     # skill gaps
    match_score: int = 0          # 0–100 fit score

    # Generated outputs
    cover_letter: str = ""
    email_draft: str = ""
    interview_questions: list = []
    prep_tips: str = ""

    # Control
    error: Optional[str] = None
    current_step: str = "start"
