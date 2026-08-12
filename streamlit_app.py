"""
Streamlit interface for the Automated Resume Parser & JD Matcher Agent.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from llm_factory import get_llm_provider_name
from matcher_agent import ResumeJDMatcherAgent

load_dotenv()

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Source+Sans+3:wght@400;500;600;700&display=swap');

:root {
  --ink: #1c2a32;
  --muted: #5b6b75;
  --paper: #f7f3ea;
  --panel: #fffdf8;
  --line: #d7cfc0;
  --accent: #0f6b5c;
  --accent-soft: #d9efe9;
}

html, body, [class*="css"] {
  font-family: "Source Sans 3", "Segoe UI", sans-serif;
  color: var(--ink);
}

.stApp {
  background:
    radial-gradient(circle at top left, #e7f2ee 0%, transparent 34%),
    linear-gradient(180deg, #f4efe4 0%, #f7f3ea 48%, #efe8da 100%);
}

.block-container {
  padding-top: 1.6rem;
  padding-bottom: 3rem;
  max-width: 920px;
}

h1, h2, h3 {
  font-family: "Fraunces", Georgia, serif !important;
  letter-spacing: -0.02em;
  color: var(--ink) !important;
}

.page-title {
  font-family: "Fraunces", Georgia, serif;
  font-size: 1.9rem;
  line-height: 1.2;
  margin: 0 0 0.35rem 0;
}

.page-sub {
  color: var(--muted);
  margin: 0 0 1.2rem 0;
  font-size: 1rem;
}

.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 1.1rem 1.15rem 0.9rem;
  margin-bottom: 1rem;
  box-shadow: 0 8px 24px rgba(28, 42, 50, 0.04);
}

.small-label {
  font-size: 0.75rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 700;
  margin-bottom: 0.2rem;
}

.gap-item, .tip-item {
  background: #fff;
  border-left: 4px solid var(--accent);
  border-radius: 10px;
  padding: 0.8rem 0.95rem;
  margin-bottom: 0.7rem;
  border-top: 1px solid var(--line);
  border-right: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
}

.gap-item.high { border-left-color: #a33b2d; }
.gap-item.medium { border-left-color: #b7791f; }
.gap-item.low { border-left-color: #2f6f4e; }

div[data-testid="stMetric"] {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 0.55rem 0.75rem;
}

.stButton > button {
  background: var(--accent) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 650 !important;
}

section[data-testid="stSidebar"] { display: none !important; }
button[kind="header"] { display: none !important; }
"""


def inject_style() -> None:
    st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)


def render_results(result: dict) -> None:
    if result.get("llm_error"):
        st.warning(
            "Part of the language enrichment failed, so this report uses the "
            f"embedding pipeline with retrieved evidence. Detail: {result['llm_error'][:160]}"
        )

    evaluation = result["evaluation"]
    meta = result.get("resume_meta") or {}

    st.markdown("### Match results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Skill match", f"{evaluation['overall_match_percentage']}%")
    m2.metric("Match band", str(evaluation["match_band"]).title())
    m3.metric("Missing skills", len(evaluation.get("missing_skills") or []))
    m4.metric("Resume pages", meta.get("page_count", "—"))

    st.markdown("#### Summary")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="small-label">Candidate</div>', unsafe_allow_html=True)
        st.write(evaluation.get("candidate_summary") or "—")
    with c2:
        st.markdown('<div class="small-label">Role</div>', unsafe_allow_html=True)
        st.write(evaluation.get("role_summary") or "—")

    tab_skills, tab_gaps, tab_tips, tab_json = st.tabs(
        ["Skill matches", "Experience gaps", "Improvement tips", "JSON"]
    )

    with tab_skills:
        skills = evaluation.get("matched_skills") or []
        if skills:
            skills_df = pd.DataFrame(
                [
                    {
                        "Skill": s.get("skill"),
                        "Score": s.get("match_score"),
                        "Present": s.get("present_in_resume"),
                        "Evidence": s.get("evidence"),
                    }
                    for s in skills
                ]
            )
            st.dataframe(skills_df, use_container_width=True, hide_index=True)

        summary_df = pd.DataFrame(
            {
                "Missing": pd.Series(evaluation.get("missing_skills") or ["None"]),
                "Partial": pd.Series(evaluation.get("partial_skills") or ["None"]),
                "Strengths": pd.Series(evaluation.get("strengths") or ["None"]),
            }
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    with tab_gaps:
        gaps = evaluation.get("experience_gaps") or []
        if not gaps:
            st.success("No major experience gaps flagged.")
        else:
            gaps_df = pd.DataFrame(
                [
                    {
                        "Area": g.get("area"),
                        "Severity": g.get("gap_severity"),
                        "JD requirement": g.get("jd_requirement"),
                        "Explanation": g.get("explanation"),
                        "Evidence": g.get("resume_evidence") or "None retrieved.",
                    }
                    for g in gaps
                ]
            )
            st.dataframe(gaps_df, use_container_width=True, hide_index=True)
            for g in gaps:
                severity = str(g.get("gap_severity") or "medium").lower()
                st.markdown(
                    f"""
<div class="gap-item {severity}">
  <div class="small-label">{severity} · {g.get('area')}</div>
  <strong>{g.get('jd_requirement')}</strong>
  <p style="margin:0.45rem 0 0.2rem 0; color:#5b6b75;">{g.get('explanation')}</p>
  <p style="margin:0; color:#5b6b75;"><em>Evidence:</em> {g.get('resume_evidence') or 'None retrieved.'}</p>
</div>
""",
                    unsafe_allow_html=True,
                )

    with tab_tips:
        tips = sorted(
            evaluation.get("improvement_tips") or [],
            key=lambda t: t.get("priority", 99),
        )
        if not tips:
            st.info("No tips generated.")
        else:
            tips_df = pd.DataFrame(
                [
                    {
                        "Priority": t.get("priority"),
                        "Category": t.get("category"),
                        "Tip": t.get("tip"),
                        "Rationale": t.get("rationale"),
                    }
                    for t in tips
                ]
            )
            st.dataframe(tips_df, use_container_width=True, hide_index=True)
            for t in tips:
                st.markdown(
                    f"""
<div class="tip-item">
  <div class="small-label">Priority {t.get('priority')} · {t.get('category')}</div>
  <strong>{t.get('tip')}</strong>
  <p style="margin:0.4rem 0 0 0; color:#5b6b75;">{t.get('rationale')}</p>
</div>
""",
                    unsafe_allow_html=True,
                )

    with tab_json:
        payload = result.get("evaluation_json") or json.dumps(evaluation, indent=2)
        st.code(payload, language="json")
        st.download_button(
            "Download evaluation JSON",
            data=payload,
            file_name="resume_jd_evaluation.json",
            mime="application/json",
        )


def main() -> None:
    st.set_page_config(
        page_title="Resume Parser & JD Matcher Agent",
        page_icon="📎",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    inject_style()

    st.markdown(
        '<h1 class="page-title">Automated Resume Parser &amp; JD Matcher Agent</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="page-sub">Upload a resume and paste a job description to generate a match report.</p>',
        unsafe_allow_html=True,
    )

    provider = get_llm_provider_name()
    if provider == "none":
        st.error(
            "LLM is not configured. Add `GROQ_API_KEY` to your local `.env` file, "
            "or to Streamlit Cloud Secrets, then reboot the app."
        )
        return

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    resume_file = st.file_uploader(
        "Resume (PDF, PNG, or JPG)",
        type=["pdf", "txt", "png", "jpg", "jpeg"],
        help="Text PDFs work best. Scanned/image resumes are OCR'd automatically.",
    )
    jd_text = st.text_area(
        "Job description",
        height=260,
        placeholder="Paste the full job description here...",
    )
    run = st.button("Run matcher agent", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if run:
        if not resume_file:
            st.error("Please upload a resume PDF (or TXT).")
        elif not jd_text.strip():
            st.error("Please paste a job description.")
        else:
            with st.spinner(
                "Analyzing resume and job description "
                "(scanned PDFs may take longer while OCR runs)..."
            ):
                try:
                    # Ensure latest parser is loaded even if Streamlit cached an old import.
                    import importlib
                    import pdf_parser as _pdf_parser
                    import matcher_agent as _matcher_agent

                    importlib.reload(_pdf_parser)
                    importlib.reload(_matcher_agent)
                    agent = _matcher_agent.ResumeJDMatcherAgent()
                    payload = agent.run(
                        file_bytes=resume_file.getvalue(),
                        filename=resume_file.name or "resume.pdf",
                        jd_text=jd_text,
                    )
                    st.session_state["result"] = payload
                except Exception as exc:  # noqa: BLE001
                    st.exception(exc)

    result = st.session_state.get("result")
    if result:
        st.divider()
        render_results(result)


if __name__ == "__main__":
    main()
