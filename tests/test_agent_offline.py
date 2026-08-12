"""Tests for heuristic JD extraction and offline agent evaluation."""

from __future__ import annotations

from matcher_agent import ResumeJDMatcherAgent, heuristic_jd_requirements


def test_heuristic_jd_extracts_core_skills():
    jd = """
    Senior Data Engineer
    Must have: Python, SQL, Apache Spark, Airflow, AWS.
    Nice to have: Kafka, Kubernetes.
    4+ years building production data pipelines.
    """
    req = heuristic_jd_requirements(jd)
    assert "Python" in req.must_have_skills or "Python" in req.keywords
    assert "SQL" in (req.must_have_skills + req.keywords)
    assert req.experience_requirements


def test_agent_offline_evaluation_high_match(monkeypatch):
    # Force no LLM path.
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.delenv("USE_OLLAMA", raising=False)

    resume = b"""Alex Rivera
Senior Data Engineer
Skills: Python, SQL, Apache Spark, Airflow, AWS, Docker, Kafka, dbt
Experience: 5 years building Spark ETL on AWS with Airflow orchestration.
"""
    jd = """
Senior Data Engineer role.
Must have Python, SQL, Apache Spark, Airflow, AWS.
Nice to have Kafka and dbt.
Requires 4+ years production data pipelines.
"""
    agent = ResumeJDMatcherAgent()
    # Ensure LLM is disabled for this test.
    agent.llm = None
    agent.provider = "none"
    out = agent.run(resume, "resume.txt", jd)
    ev = out["evaluation"]
    assert ev["overall_match_percentage"] >= 70
    assert ev["matched_skills"]
    assert isinstance(ev["improvement_tips"], list)
    # Validate pydantic-compatible dump has required keys
    for key in ("experience_gaps", "missing_skills", "match_band"):
        assert key in ev
