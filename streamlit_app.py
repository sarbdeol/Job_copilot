"""
streamlit_app.py — Full Streamlit UI for Job Co-Pilot
Run with: streamlit run streamlit_app.py
"""
import streamlit as st
import httpx
import json

API_BASE = "http://localhost:8000"

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Job Co-Pilot",
    page_icon="🚀",
    layout="wide",
)

st.title("🚀 AI Job Application Co-Pilot")
st.caption("Powered by LangGraph + GPT-4o-mini")

# ─── Sidebar: Resume Upload ───────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Your Resume")
    st.caption("Upload your CV — PDF, DOCX, or TXT")

    uploaded_file = st.file_uploader(
        "Drop your resume here",
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, Word (.docx), Plain Text (.txt)",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        # Show file info
        file_size_kb = len(uploaded_file.getvalue()) / 1024
        st.markdown(f"**📎 {uploaded_file.name}** ({file_size_kb:.1f} KB)")

        if st.button("📥 Upload & Save to Memory", use_container_width=True, type="primary"):
            with st.spinner(f"Parsing {uploaded_file.name}..."):
                try:
                    file_bytes = uploaded_file.getvalue()
                    response = httpx.post(
                        f"{API_BASE}/upload-resume",
                        files={"file": (uploaded_file.name, file_bytes, uploaded_file.type)},
                        timeout=60,
                    )
                    if response.status_code == 200:
                        data = response.json()
                        chars = data.get("characters_extracted", 0)
                        st.success(f"✅ Resume saved! ({chars:,} characters extracted)")
                        st.session_state["resume_saved"] = True
                        st.session_state["resume_filename"] = uploaded_file.name

                        # Show preview
                        with st.expander("👁 Preview extracted text"):
                            st.text(data.get("preview", ""))
                    else:
                        error_detail = response.json().get("detail", response.text)
                        st.error(f"❌ {error_detail}")
                except httpx.ConnectError:
                    st.error("❌ API not running. Start with: `uvicorn main:app --reload`")
                except Exception as e:
                    st.error(f"Error: {e}")

    elif not st.session_state.get("resume_saved"):
        st.info("👆 Upload your resume to get started")

    if st.session_state.get("resume_saved"):
        filename = st.session_state.get("resume_filename", "your resume")
        st.success(f"✅ **{filename}** loaded in memory")
        if st.button("🗑 Clear Resume", use_container_width=True):
            st.session_state.pop("resume_saved", None)
            st.session_state.pop("resume_filename", None)
            st.rerun()

# ─── Main: Job Description Input ─────────────────────────────────────────────
st.subheader("📋 Paste Job Description")
job_description = st.text_area(
    "Job Description",
    height=250,
    placeholder="Paste the full job description here...",
    label_visibility="collapsed",
)

analyze_btn = st.button("⚡ Analyze & Generate", type="primary", use_container_width=True)

# ─── Run Analysis ─────────────────────────────────────────────────────────────
if analyze_btn:
    if not job_description.strip():
        st.error("Please paste a job description.")
    else:
        progress = st.progress(0, text="Starting pipeline...")

        with st.spinner("Running LangGraph pipeline..."):
            try:
                # Update progress visually for each step
                steps = [
                    (20, "🔍 Parsing job description..."),
                    (40, "🧠 Analyzing skill gaps..."),
                    (60, "✍️ Writing cover letter..."),
                    (80, "📧 Drafting email..."),
                    (95, "🎯 Generating interview prep..."),
                ]

                import time
                for pct, msg in steps:
                    progress.progress(pct, text=msg)
                    time.sleep(0.3)

                response = httpx.post(
                    f"{API_BASE}/analyze",
                    json={
                        "job_description": job_description,
                        "resume_text": "",  # Resume is stored in ChromaDB via /upload-resume
                    },
                    timeout=120,
                )

                progress.progress(100, text="✅ Complete!")

                if response.status_code == 200:
                    data = response.json()
                    st.session_state["result"] = data
                else:
                    st.error(f"Pipeline error: {response.text}")

            except httpx.ConnectError:
                st.error("❌ Cannot connect to API. Make sure `uvicorn main:app --reload` is running.")
            except Exception as e:
                st.error(f"Error: {e}")

# ─── Display Results ──────────────────────────────────────────────────────────
if "result" in st.session_state:
    data = st.session_state["result"]

    st.divider()

    # Header metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🏢 Role", data.get("job_title", "—"))
    with col2:
        st.metric("🏬 Company", data.get("company", "—"))
    with col3:
        score = data.get("match_score", 0)
        color = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"
        st.metric(f"{color} Match Score", f"{score}/100")

    st.divider()

    # Tabs for results
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Skills Analysis",
        "✉️ Cover Letter",
        "📧 Email",
        "🎤 Interview Prep"
    ])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("✅ You Have These")
            for skill in data.get("matched_skills", []):
                st.success(f"✓  {skill}")
        with col_b:
            st.subheader("📚 Gaps to Bridge")
            for skill in data.get("missing_skills", []):
                st.warning(f"△  {skill}")

    with tab2:
        st.subheader("Cover Letter")
        st.text_area(
            "cover_letter_output",
            value=data.get("cover_letter", ""),
            height=400,
            label_visibility="collapsed",
        )
        st.download_button(
            "📥 Download Cover Letter",
            data=data.get("cover_letter", ""),
            file_name="cover_letter.txt",
            mime="text/plain",
        )

    with tab3:
        st.subheader("Application Email")
        st.text_area(
            "email_output",
            value=data.get("email_draft", ""),
            height=250,
            label_visibility="collapsed",
        )
        st.download_button(
            "📥 Download Email",
            data=data.get("email_draft", ""),
            file_name="application_email.txt",
            mime="text/plain",
        )

    with tab4:
        st.subheader("Likely Interview Questions")
        for i, q in enumerate(data.get("interview_questions", []), 1):
            st.markdown(f"**Q{i}.** {q}")

        st.divider()
        st.subheader("💡 Prep Tips")
        st.info(data.get("prep_tips", ""))

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with LangChain • LangGraph • FastAPI • ChromaDB • Streamlit")
