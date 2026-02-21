"""
app/agents/nodes.py — Individual LangGraph node functions
Each node takes AppState, does ONE job, returns updated AppState dict.
"""
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from app.core.config import OPENAI_API_KEY, OPENAI_MODEL
from app.core.config import AppState
from app.core.resume_store import retrieve_resume_context


# ─── Shared LLM ───────────────────────────────────────────────────────────────

def get_llm(temperature: float = 0.3) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL,
        temperature=temperature,
    )


# ─── Node 1: Parse Job Description ────────────────────────────────────────────

def parse_jd_node(state: dict) -> dict:
    """Extracts structured info from the raw job description."""
    llm = get_llm(temperature=0.1)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a job description parser. Extract key information and return ONLY valid JSON.
Return this exact structure:
{{
  "job_title": "...",
  "company": "...",
  "skills_required": ["skill1", "skill2"],
  "responsibilities": ["resp1", "resp2"],
  "experience_years": "...",
  "location": "..."
}}"""),
        ("human", "Parse this job description:\n\n{jd}")
    ])

    chain = prompt | llm
    response = chain.invoke({"jd": state["job_description"]})

    try:
        # Strip markdown code blocks if present
        text = response.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text.strip())
    except Exception as e:
        parsed = {
            "job_title": "Unknown",
            "company": "Unknown",
            "skills_required": [],
            "responsibilities": [],
            "experience_years": "Unknown",
            "location": "Unknown",
        }

    return {**state, "parsed_jd": parsed, "current_step": "jd_parsed"}


# ─── Node 2: Skill Gap Analysis ───────────────────────────────────────────────

def skill_gap_node(state: dict) -> dict:
    """Compares JD skills against resume, returns matched/missing + score."""
    llm = get_llm(temperature=0.1)

    required_skills = state["parsed_jd"].get("skills_required", [])
    resume_context = retrieve_resume_context(
        query=f"skills experience {' '.join(required_skills[:5])}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a technical recruiter doing a skills gap analysis.
Compare the required skills against the candidate's resume and return ONLY valid JSON:
{{
  "matched_skills": ["skill1", "skill2"],
  "missing_skills": ["skill3"],
  "match_score": 85,
  "summary": "Brief 1-2 sentence assessment"
}}
match_score is 0-100 based on overall fit."""),
        ("human", """Required skills: {skills}

Candidate's resume (relevant sections):
{resume}""")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "skills": json.dumps(required_skills),
        "resume": resume_context
    })

    try:
        text = response.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text.strip())
    except Exception:
        result = {
            "matched_skills": [],
            "missing_skills": required_skills,
            "match_score": 0,
            "summary": "Could not analyze."
        }

    return {
        **state,
        "matched_skills": result.get("matched_skills", []),
        "missing_skills": result.get("missing_skills", []),
        "match_score": result.get("match_score", 0),
        "current_step": "skills_analyzed",
    }


# ─── Node 3: Generate Cover Letter ────────────────────────────────────────────

def cover_letter_node(state: dict) -> dict:
    """Generates a tailored cover letter using candidate's real name/address."""
    llm = get_llm(temperature=0.7)

    # Retrieve relevant experience sections
    resume_context = retrieve_resume_context(
        query=f"{state['parsed_jd'].get('job_title', '')} experience projects achievements"
    )

    # Targeted retrieval for contact/personal info
    contact_context = retrieve_resume_context(
        query="name address phone email contact location"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert career coach writing cover letters.

IMPORTANT: Extract the candidate's full name, address, phone, and email from the contact info provided.
Use them to format the letter properly:

[Candidate Full Name]
[Address]
[Phone] | [Email]

[Date]

[Hiring Manager / Company]

Dear Hiring Manager,

...letter body...

Sincerely,
[Candidate Full Name]

Rules:
- Keep it under 350 words
- Focus on matched skills and specific achievements
- Do NOT use generic phrases like "I am excited to apply"
- Be confident, specific, and show genuine value
- If contact info is missing, leave a placeholder like [Your Name]"""),
        ("human", """Job: {job_title} at {company}
Matched Skills: {matched_skills}
Skills to address carefully: {missing_skills}

Candidate contact info (from resume):
{contact_context}

Relevant resume experience:
{resume}

Write the cover letter:""")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "job_title": state["parsed_jd"].get("job_title", ""),
        "company": state["parsed_jd"].get("company", ""),
        "matched_skills": ", ".join(state["matched_skills"]),
        "missing_skills": ", ".join(state["missing_skills"]),
        "contact_context": contact_context,
        "resume": resume_context,
    })

    return {**state, "cover_letter": response.content, "current_step": "cover_letter_done"}


# ─── Node 4: Generate Application Email ───────────────────────────────────────

def email_node(state: dict) -> dict:
    """Generates a professional application email signed with candidate's real name."""
    llm = get_llm(temperature=0.5)

    # Retrieve candidate's name and contact info
    contact_context = retrieve_resume_context(
        query="name address phone email contact"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Write a concise, professional job application email. Under 150 words.

Format:
Subject: Application for [Job Title] — [Candidate Name]

Dear Hiring Team,

...body...

Best regards,
[Candidate Full Name]
[Phone] | [Email]

IMPORTANT: Extract and use the real candidate name, phone, and email from the contact info provided.
If not found, use placeholders like [Your Name]."""),
        ("human", """Job: {job_title} at {company}
Match Score: {score}/100
Top matched skills: {skills}

Candidate contact info:
{contact_context}

Generate the email:""")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "job_title": state["parsed_jd"].get("job_title", ""),
        "company": state["parsed_jd"].get("company", ""),
        "score": state["match_score"],
        "skills": ", ".join(state["matched_skills"][:5]),
        "contact_context": contact_context,
    })

    return {**state, "email_draft": response.content, "current_step": "email_done"}

# ─── Node 5: Interview Prep ────────────────────────────────────────────────────

def interview_prep_node(state: dict) -> dict:
    """Generates likely interview questions + prep tips."""
    llm = get_llm(temperature=0.6)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior technical interviewer.
Generate interview preparation for this role. Return ONLY valid JSON:
{{
  "technical_questions": ["q1", "q2", "q3", "q4", "q5"],
  "behavioral_questions": ["q1", "q2", "q3"],
  "prep_tips": "2-3 specific tips based on the role and skill gaps"
}}"""),
        ("human", """Role: {job_title} at {company}
Key Responsibilities: {responsibilities}
Skill Gaps to prepare for: {missing_skills}

Generate interview prep:""")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "job_title": state["parsed_jd"].get("job_title", ""),
        "company": state["parsed_jd"].get("company", ""),
        "responsibilities": json.dumps(state["parsed_jd"].get("responsibilities", [])[:4]),
        "missing_skills": ", ".join(state["missing_skills"][:5]),
    })

    try:
        text = response.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text.strip())
    except Exception:
        result = {
            "technical_questions": [],
            "behavioral_questions": [],
            "prep_tips": "Focus on your experience with the matched skills."
        }

    all_questions = (
        result.get("technical_questions", []) +
        result.get("behavioral_questions", [])
    )

    return {
        **state,
        "interview_questions": all_questions,
        "prep_tips": result.get("prep_tips", ""),
        "current_step": "complete",
    }
