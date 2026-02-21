"""
app/agents/graph.py — LangGraph workflow definition
"""
from langgraph.graph import StateGraph, END
from typing import TypedDict, Any
from app.agents.nodes import (
    parse_jd_node,
    skill_gap_node,
    cover_letter_node,
    email_node,
    interview_prep_node,
)

class GraphState(TypedDict):
    job_description: str
    resume_text: str
    parsed_jd: dict
    matched_skills: list
    missing_skills: list
    match_score: int
    cover_letter: str
    email_draft: str
    interview_questions: list
    prep_tips: str
    error: Any
    current_step: str


def build_graph() -> Any:
    graph = StateGraph(GraphState)

    # ✅ Node names must NOT match any state key names
    graph.add_node("step_parse_jd", parse_jd_node)
    graph.add_node("step_skill_gap", skill_gap_node)
    graph.add_node("step_cover_letter", cover_letter_node)
    graph.add_node("step_email", email_node)
    graph.add_node("step_interview_prep", interview_prep_node)

    graph.set_entry_point("step_parse_jd")
    graph.add_edge("step_parse_jd", "step_skill_gap")
    graph.add_edge("step_skill_gap", "step_cover_letter")
    graph.add_edge("step_cover_letter", "step_email")
    graph.add_edge("step_email", "step_interview_prep")
    graph.add_edge("step_interview_prep", END)

    return graph.compile()


def run_pipeline(job_description: str, resume_text: str) -> GraphState:
    app = build_graph()

    initial_state: GraphState = {
        "job_description": job_description,
        "resume_text": resume_text,
        "parsed_jd": {},
        "matched_skills": [],
        "missing_skills": [],
        "match_score": 0,
        "cover_letter": "",
        "email_draft": "",
        "interview_questions": [],
        "prep_tips": "",
        "error": None,
        "current_step": "start",
    }

    return app.invoke(initial_state)