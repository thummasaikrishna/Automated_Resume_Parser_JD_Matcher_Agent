"""Unit tests for parser, schemas, and embedding match scoring."""

from __future__ import annotations

import json

import pytest

from agent_models import EvaluationResult, ImprovementTip, SkillMatchItem
from embeddings_matcher import cosine_similarity, match_band
from pdf_parser import parse_resume_bytes


def test_cosine_identical():
    v = [0.1, 0.2, 0.3]
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_orthogonal():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-6)


def test_match_band_thresholds():
    assert match_band(90) == "excellent"
    assert match_band(72) == "strong"
    assert match_band(60) == "moderate"
    assert match_band(45) == "weak"
    assert match_band(10) == "poor"


def test_parse_txt_resume():
    text = (
        "Jane Doe\nSoftware Engineer\nSkills: Python, FastAPI, SQL, Docker\n"
        "Experience: Built APIs and data pipelines for 4 years at Acme."
    ).encode("utf-8")
    parsed = parse_resume_bytes(text, "resume.txt")
    assert "Python" in parsed.text
    assert parsed.char_count > 40


def test_parse_rejects_short_text():
    with pytest.raises(ValueError):
        parse_resume_bytes(b"hi", "resume.txt")


def test_evaluation_result_schema_roundtrip():
    payload = EvaluationResult(
        candidate_summary="Backend engineer with Python focus.",
        role_summary="Looking for a Python API engineer.",
        overall_match_percentage=78.55,
        match_band="strong",
        matched_skills=[
            SkillMatchItem(
                skill="Python",
                jd_requirement="Python",
                present_in_resume=True,
                match_score=91.2,
                evidence="Built APIs in Python",
            )
        ],
        missing_skills=["Kubernetes"],
        partial_skills=["Docker"],
        experience_gaps=[],
        strengths=["Strong Python"],
        improvement_tips=[
            ImprovementTip(
                priority=1,
                category="skill",
                tip="Add a Kubernetes project",
                rationale="JD requires K8s",
            )
        ],
        scoring_notes="weighted embedding average",
    )
    data = json.loads(payload.model_dump_json())
    again = EvaluationResult.model_validate(data)
    assert again.overall_match_percentage == 78.55
    assert again.matched_skills[0].skill == "Python"
