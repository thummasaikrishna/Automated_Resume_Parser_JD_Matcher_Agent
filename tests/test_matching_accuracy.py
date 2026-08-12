"""
Embedding match accuracy tests (no LLM required).

Uses local sentence-transformers via ResumeVectorIndex.
"""

from __future__ import annotations

import pytest

from embeddings_matcher import ResumeVectorIndex, compute_match_percentage


STRONG_RESUME = """
Alex Rivera
Senior Data Engineer

Skills: Python, SQL, Apache Spark, Airflow, dbt, AWS, Docker, PostgreSQL, Kafka

Experience:
- Designed Spark ETL pipelines processing 2TB/day on AWS EMR.
- Built Airflow DAGs and dbt models for analytics warehouse.
- Containerized services with Docker and deployed on ECS.
"""

WEAK_RESUME = """
Sam Lee
Retail Store Manager

Skills: Team leadership, inventory, POS systems, customer service, Excel

Experience:
- Managed a 20-person retail team and improved conversion rates.
- Handled scheduling, merchandising, and vendor coordination.
"""


@pytest.fixture(scope="module")
def strong_index():
    return ResumeVectorIndex(STRONG_RESUME)


@pytest.fixture(scope="module")
def weak_index():
    return ResumeVectorIndex(WEAK_RESUME)


def test_high_match_for_aligned_skills(strong_index):
    overall, sims, missing, partial = compute_match_percentage(
        must_have=["Python", "SQL", "Apache Spark", "Airflow"],
        nice_to_have=["Kafka"],
        index=strong_index,
    )
    assert overall >= 70
    assert "Python" not in missing
    assert "SQL" not in missing
    by_skill = {s.skill: s.score_0_100 for s in sims}
    assert by_skill["Python"] >= 70


def test_low_match_for_misaligned_skills(weak_index):
    overall, sims, missing, _ = compute_match_percentage(
        must_have=["Python", "Kubernetes", "Machine Learning", "TensorFlow"],
        nice_to_have=["PyTorch"],
        index=weak_index,
    )
    assert overall < 55
    assert len(missing) >= 2


def test_exact_skill_mention_boost(strong_index):
    sim = strong_index.skill_similarity("Docker")
    assert sim.present is True
    assert sim.score_0_100 >= 70
