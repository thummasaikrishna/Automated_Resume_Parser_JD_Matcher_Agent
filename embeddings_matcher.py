"""
Vector embedding skill-match scoring and resume RAG retrieval.

Uses pure cosine similarity over chunk embeddings (no FAISS),
so Streamlit Cloud installs cleanly across Python versions.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm_factory import build_embeddings


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def _normalize_skill(skill: str) -> str:
    s = re.sub(r"\s+", " ", (skill or "").strip().lower())
    s = re.sub(r"[^\w+#.\s/-]", "", s)
    return s


@dataclass
class SkillSimilarity:
    skill: str
    score_0_100: float
    best_chunk: str
    present: bool


class ResumeVectorIndex:
    """Chunk a resume and index it for RAG + skill similarity queries."""

    def __init__(self, resume_text: str, embeddings: Any | None = None):
        self.embeddings = embeddings or build_embeddings()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=120,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_text(resume_text)
        if not chunks:
            chunks = [resume_text]
        self.docs = [Document(page_content=c, metadata={"chunk_id": i}) for i, c in enumerate(chunks)]
        self.chunk_texts = [d.page_content for d in self.docs]
        self._chunk_vectors = self.embeddings.embed_documents(self.chunk_texts)

    def retrieve(self, query: str, k: int = 4) -> list[Document]:
        q_vec = self.embeddings.embed_query(query)
        scored = [
            (cosine_similarity(q_vec, vec), doc)
            for doc, vec in zip(self.docs, self._chunk_vectors)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[: max(1, k)]]

    def retrieve_context(self, query: str, k: int = 4) -> str:
        docs = self.retrieve(query, k=k)
        return "\n\n---\n\n".join(d.page_content for d in docs)

    def skill_similarity(self, skill: str, threshold: float = 0.42) -> SkillSimilarity:
        """
        Score a JD skill against the resume using max cosine similarity
        over resume chunks (embedding-based match).
        """
        skill_clean = skill.strip()
        if not skill_clean:
            return SkillSimilarity(skill=skill, score_0_100=0.0, best_chunk="", present=False)

        query = f"Experience, projects, and proficiency with {skill_clean}"
        q_vec = self.embeddings.embed_query(query)

        best_sim = 0.0
        best_chunk = ""
        for chunk, vec in zip(self.chunk_texts, self._chunk_vectors):
            sim = cosine_similarity(q_vec, vec)
            if sim > best_sim:
                best_sim = sim
                best_chunk = chunk

        norm_skill = _normalize_skill(skill_clean)
        resume_norm = _normalize_skill(" ".join(self.chunk_texts))
        if norm_skill and norm_skill in resume_norm:
            best_sim = max(best_sim, 0.88)
        else:
            tokens = [t for t in norm_skill.split() if len(t) > 2]
            if tokens and all(t in resume_norm for t in tokens):
                best_sim = max(best_sim, 0.72)

        score = round(best_sim * 100, 2)
        present = best_sim >= threshold
        snippet = re.sub(r"\s+", " ", best_chunk).strip()
        if len(snippet) > 280:
            snippet = snippet[:277] + "..."
        return SkillSimilarity(
            skill=skill_clean,
            score_0_100=score,
            best_chunk=snippet,
            present=present,
        )


def compute_match_percentage(
    must_have: Iterable[str],
    nice_to_have: Iterable[str],
    index: ResumeVectorIndex,
    must_weight: float = 0.75,
    nice_weight: float = 0.25,
    present_threshold: float = 0.42,
) -> tuple[float, list[SkillSimilarity], list[str], list[str]]:
    """
    Weighted embedding match:
      overall = must_weight * avg(must) + nice_weight * avg(nice)
    """
    must = [s.strip() for s in must_have if s and s.strip()]
    nice = [s.strip() for s in nice_to_have if s and s.strip()]

    must_sims = [index.skill_similarity(s, threshold=present_threshold) for s in must]
    nice_sims = [index.skill_similarity(s, threshold=present_threshold) for s in nice]

    def avg(scores: list[SkillSimilarity]) -> float:
        if not scores:
            return 100.0 if not must else 0.0
        return sum(s.score_0_100 for s in scores) / len(scores)

    if must and nice:
        overall = must_weight * avg(must_sims) + nice_weight * avg(nice_sims)
    elif must:
        overall = avg(must_sims)
    elif nice:
        overall = avg(nice_sims)
    else:
        overall = 0.0

    all_sims = must_sims + nice_sims
    missing: list[str] = []
    partial: list[str] = []
    for s in must_sims:
        if s.score_0_100 < present_threshold * 100:
            missing.append(s.skill)
        elif s.score_0_100 < 70:
            partial.append(s.skill)

    return round(overall, 2), all_sims, missing, partial


def match_band(score: float) -> str:
    if score >= 85:
        return "excellent"
    if score >= 70:
        return "strong"
    if score >= 55:
        return "moderate"
    if score >= 40:
        return "weak"
    return "poor"
