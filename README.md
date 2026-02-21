# CoPilot — AI Job Application Assistant

> Analyze any job description against your resume. Get a skill gap analysis, tailored cover letter, application email, and interview prep — in seconds.

**Live Demo →** [copilot.deoltechnify.com](https://copilot.deoltechnify.com)

---

## What It Does

Upload your resume (PDF, DOCX, or TXT), paste a job description, and CoPilot runs a 5-step AI pipeline powered by LangGraph:

| Step | What Happens |
|------|-------------|
| Parse JD | Extracts job title, company, required skills, responsibilities |
| Skill Gap | Compares JD requirements against your resume using RAG |
| Cover Letter | Generates a tailored cover letter with your real name and contact info |
| Email Draft | Writes a concise professional application email |
| Interview Prep | Generates likely questions and preparation tips |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| AI Orchestration | LangGraph (stateful multi-node graph) |
| LLM | GPT-4o-mini via LangChain |
| Vector Database | ChromaDB (resume embeddings) |
| Backend | FastAPI |
| Frontend | Vanilla HTML/CSS/JS |

---

## Project Structure

```
Job_copilot/
├── main.py                  # FastAPI entry point
├── requirements.txt
├── .env.example
├── frontend/
│   └── index.html           # Full frontend (single file)
└── app/
    ├── core/
    │   ├── config.py        # AppState + env config
    │   ├── resume_store.py  # ChromaDB ingest & retrieval
    │   └── file_parser.py   # PDF/DOCX/TXT text extraction
    ├── agents/
    │   ├── nodes.py         # 5 LangGraph node functions
    │   └── graph.py         # Graph assembly + pipeline runner
    └── api/
        └── routes.py        # FastAPI endpoints
```

---

## LangGraph Pipeline

```
[parse_jd]      → Structured extraction from raw job description
     ↓
[skill_gap]     → RAG: match resume chunks against required skills
     ↓
[cover_letter]  → Personalized letter using real contact info from resume
     ↓
[email]         → Short professional application email
     ↓
[interview_prep]→ Likely questions + role-specific prep tips
     ↓
    END
```

Each node is a plain Python function — takes state dict, does one job, returns updated state. LangGraph handles the routing.

---

## Local Setup

### 1. Clone & Install

```bash
git clone https://github.com/sarbdeol/Job_copilot.git
cd Job_copilot

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

### 3. Run

```bash
# Terminal 1 — start the API
uvicorn main:app --reload

# Open frontend/index.html in your browser
# Or serve it: python -m http.server 3000 --directory frontend
```

API docs available at `http://localhost:8000/docs`

---

## API Reference

```
GET  /health          Health check
POST /upload-resume   Upload resume file (PDF/DOCX/TXT) → embed to ChromaDB
POST /analyze         Run full LangGraph pipeline
```

---

## Self-Hosting on EC2 + Nginx

See the [deployment guide](docs/DEPLOYMENT.md) for full EC2 + Nginx + SSL setup.

Quick overview:
- EC2 Ubuntu 22.04, t3.small minimum
- Nginx serves `frontend/` as static files, proxies `/api/` to FastAPI
- Supervisor keeps FastAPI running 24/7
- Free SSL via Certbot (Let's Encrypt)

---

## Contributing

Contributions welcome. Some ideas:

- **Human-in-the-loop** — let user edit cover letter mid-pipeline using LangGraph `interrupt_before`
- **Parallel nodes** — run cover letter and email generation in parallel with `Send()`
- **Company research node** — add a web search agent to pull company context
- **Persistence** — use LangGraph `SqliteSaver` to resume interrupted pipelines
- **Conditional routing** — if match score < 40, route to "should you apply?" node
- **Multi-resume support** — namespace ChromaDB collections per user

Open an issue first for large changes. PRs welcome for bug fixes anytime.

---

## License

MIT — free to use, modify, and distribute.

---

Built by [Sarab](https://github.com/sarbdeol) · [LinkedIn](https://linkedin.com/in/your-profile)