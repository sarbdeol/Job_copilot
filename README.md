# 🚀 AI Job Application Co-Pilot

A real-world LangGraph project that helps you analyze job descriptions, identify skill gaps, generate cover letters, draft application emails, and prepare for interviews — all automatically.

## Tech Stack

| Layer | Tech |
|-------|------|
| Orchestration | **LangGraph** (stateful multi-node graph) |
| LLM | **OpenAI GPT-4o-mini** via LangChain |
| Vector DB | **ChromaDB** (resume embeddings) |
| Backend | **FastAPI** |
| Frontend | **Streamlit** |

---

## Project Structure

```
job_copilot/
├── main.py                    # FastAPI entry point
├── streamlit_app.py           # Streamlit UI
├── requirements.txt
├── .env.example
├── data/
│   └── chroma_db/             # Auto-created: resume vector store
└── app/
    ├── core/
    │   ├── config.py          # AppState (Pydantic model) + env config
    │   └── resume_store.py    # ChromaDB ingest + retrieval
    ├── agents/
    │   ├── nodes.py           # 5 LangGraph node functions
    │   └── graph.py           # Graph assembly + runner
    └── api/
        └── routes.py          # FastAPI endpoints
```

---

## LangGraph Workflow

```
[parse_jd_node]          ← Extracts job title, skills, responsibilities
      ↓
[skill_gap_node]         ← RAG: compares JD skills vs your resume
      ↓
[cover_letter_node]      ← Generates tailored cover letter
      ↓
[email_node]             ← Drafts application email
      ↓
[interview_prep_node]    ← Generates questions + prep tips
      ↓
     END
```

Each node:
- Takes the full `GraphState` dict
- Does ONE focused job (single responsibility)
- Returns updated state dict
- The graph handles routing between nodes

---

## Setup

### 1. Clone & Install

```bash
cd job_copilot
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run the API

```bash
uvicorn main:app --reload
# API running at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 4. Run the UI (in a new terminal)

```bash
streamlit run streamlit_app.py
# UI at http://localhost:8501
```

---

## Usage

1. **Paste your resume** in the sidebar → click "Save Resume to Memory"
   - This embeds your resume into ChromaDB (only needed once)

2. **Paste a job description** in the main area

3. **Click "Analyze & Generate"**

4. View results across 4 tabs:
   - 🎯 Skills Analysis (match score, gaps)
   - ✉️ Cover Letter (download ready)
   - 📧 Application Email
   - 🎤 Interview Questions + Tips

---

## API Endpoints

```
GET  /health           → Health check
POST /ingest-resume    → Store resume in ChromaDB
POST /analyze          → Run full LangGraph pipeline
```

### Test with curl

```bash
# Health check
curl http://localhost:8000/health

# Ingest resume
curl -X POST http://localhost:8000/ingest-resume \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "Your full resume here..."}'

# Analyze a job
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"job_description": "Job description here..."}'
```

---

## Key Learning Concepts in This Project

| Concept | Where in Code |
|---------|---------------|
| LangGraph StateGraph | `app/agents/graph.py` |
| LangGraph TypedDict State | `app/agents/graph.py` → `GraphState` |
| LangGraph Nodes | `app/agents/nodes.py` → each `*_node()` |
| LangChain Chains | `nodes.py` → `prompt | llm` pattern |
| RAG (Retrieval-Augmented Generation) | `resume_store.py` + `skill_gap_node` |
| ChromaDB embeddings | `app/core/resume_store.py` |
| Structured LLM output (JSON) | `parse_jd_node`, `skill_gap_node`, `interview_prep_node` |
| FastAPI + Pydantic | `app/api/routes.py` |

---

## Extend This Project

Ideas to go deeper with LangGraph:

- **Add human-in-the-loop**: Use `interrupt_before` to let user edit the cover letter mid-graph
- **Conditional routing**: If `match_score < 40`, route to a "should you apply?" warning node
- **Parallel nodes**: Run `cover_letter_node` and `email_node` in parallel with `Send()`
- **Persistence**: Use LangGraph's `SqliteSaver` to save state between sessions
- **Multi-agent**: Add a "Research Company" sub-agent that searches the web about the company
