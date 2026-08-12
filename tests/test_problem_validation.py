"""
Problem-statement validation checklist (runnable assertions).
"""

from __future__ import annotations

import ast
from pathlib import Path

import agent_models
import embeddings_matcher
import matcher_agent
import pdf_parser


ROOT = Path(__file__).resolve().parents[1]


def test_required_modules_exist():
    for name in [
        "pdf_parser.py",
        "embeddings_matcher.py",
        "matcher_agent.py",
        "agent_models.py",
        "llm_factory.py",
        "streamlit_app.py",
        "requirements.txt",
    ]:
        assert (ROOT / name).exists(), f"Missing {name}"


def test_tech_stack_in_requirements():
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for pkg in [
        "pymupdf",
        "pypdf",
        "langchain",
        "pydantic",
        "streamlit",
        "faiss",
        "sentence-transformers",
        "pandas",
        "python-dotenv",
        "pytest",
    ]:
        assert pkg in req, f"{pkg} missing from requirements"


def test_pydantic_evaluation_model_fields():
    fields = set(agent_models.EvaluationResult.model_fields)
    for required in [
        "overall_match_percentage",
        "matched_skills",
        "experience_gaps",
        "improvement_tips",
        "missing_skills",
    ]:
        assert required in fields


def test_pdf_parser_uses_pymupdf_or_pypdf():
    src = (ROOT / "pdf_parser.py").read_text(encoding="utf-8")
    assert "import pymupdf" in src or "import fitz" in src
    assert "pypdf" in src
    assert "RapidOCR" in src or "ocr" in src.lower()


def test_agent_uses_structured_output_and_rag():
    src = (ROOT / "matcher_agent.py").read_text(encoding="utf-8")
    assert "with_structured_output" in src
    assert "ResumeVectorIndex" in src
    assert "EvaluationResult" in src


def test_embeddings_match_scoring_present():
    src = (ROOT / "embeddings_matcher.py").read_text(encoding="utf-8")
    assert "cosine_similarity" in src
    assert "compute_match_percentage" in src
    assert "FAISS" in src


def test_streamlit_entrypoint_parses():
    tree = ast.parse((ROOT / "streamlit_app.py").read_text(encoding="utf-8"))
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "main" in names
