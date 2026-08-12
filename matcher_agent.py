"""
Resume Parser & JD Matcher Agent.

Pipeline:
1. Parse PDF resume (pypdf)
2. Extract JD requirements (Groq structured output, heuristic fallback)
3. Build resume vector index (embeddings) for Document RAG
4. Compute skill-match % via embedding similarity (authoritative)
5. Produce EvaluationResult (Pydantic) via Groq + RAG, or deterministic fallback
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from agent_models import (
    EvaluationResult,
    ExperienceGap,
    ImprovementTip,
    JDRequirements,
    ResumeProfile,
    SkillMatchItem,
)
from embeddings_matcher import (
    ResumeVectorIndex,
    SkillSimilarity,
    compute_match_percentage,
    match_band,
)
from llm_factory import build_chat_llm, get_llm_provider_name
from pdf_parser import ParsedResume, parse_resume_bytes

# Broad skill lexicon for high-recall heuristic JD parsing when LLM is unavailable.
SKILL_LEXICON = [
    "Python", "Java", "JavaScript", "TypeScript", "Go", "Golang", "C++", "C#", "Ruby", "PHP",
    "SQL", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
    "Apache Spark", "Spark", "Hadoop", "Kafka", "Airflow", "dbt", "Snowflake", "BigQuery",
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "CI/CD",
    "FastAPI", "Django", "Flask", "React", "Node.js", "GraphQL",
    "Machine Learning", "Deep Learning", "NLP", "TensorFlow", "PyTorch", "scikit-learn",
    "Power BI", "Tableau", "Excel", "Pandas", "NumPy",
    "REST", "Microservices", "Linux", "Git",
]


JD_EXTRACT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract structured hiring requirements from a job description. "
            "Return ONLY fields defined by the schema. Be specific and deduplicate skills. "
            "Prefer concrete technologies, tools, and competencies over vague soft phrases.",
        ),
        ("human", "Job Description:\n\n{jd_text}"),
    ]
)

RESUME_EXTRACT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract a structured candidate profile from resume evidence. "
            "Use ONLY the provided resume context. Do not invent employers or skills. "
            "If unknown, leave fields empty/null.",
        ),
        ("human", "Resume context (retrieved excerpts + full text sample):\n\n{resume_context}"),
    ]
)

EVAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert technical recruiter and career coach. "
            "Produce a rigorous structured evaluation grounded ONLY in the provided evidence. "
            "Do not invent resume facts. Be specific in improvement tips. "
            "overall_match_percentage MUST equal the provided embedding_match_percentage. "
            "matched_skills should reflect the provided embedding skill scores. "
            "Identify experience gaps (years, domain depth, leadership, scale, certifications) clearly.",
        ),
        (
            "human",
            """Embedding match percentage (authoritative): {embedding_match_percentage}
Match band: {band}
Missing skills (embedding): {missing_skills}
Partial skills (embedding): {partial_skills}

JD requirements JSON:
{jd_json}

Resume profile JSON:
{resume_json}

Skill embedding scores JSON:
{skill_scores_json}

Retrieved resume evidence for gaps/tips:
{rag_context}

Return the full structured evaluation.
""",
        ),
    ]
)


def _structured(llm: Any, schema: type):
    try:
        return llm.with_structured_output(schema)
    except Exception:
        return llm.with_structured_output(schema, method="json_mode")


def heuristic_jd_requirements(jd_text: str) -> JDRequirements:
    """Deterministic JD parsing used when LLM is unavailable or fails."""
    text = jd_text or ""
    lower = text.lower()
    found: list[str] = []
    for skill in sorted(SKILL_LEXICON, key=len, reverse=True):
        pattern = r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])"
        if re.search(pattern, lower):
            if skill == "Spark" and "Apache Spark" in found:
                continue
            if skill == "Golang" and "Go" in found:
                continue
            found.append(skill)

    # Split must vs nice using section cues.
    nice: list[str] = []
    must = list(found)
    nice_match = re.search(
        r"(nice to have|preferred|bonus|plus)([\s\S]{0,400})",
        lower,
    )
    if nice_match:
        window = nice_match.group(0)
        moved = []
        for skill in found:
            if skill.lower() in window:
                moved.append(skill)
        if moved:
            nice = moved
            must = [s for s in found if s not in nice]

    exp = []
    for m in re.finditer(r"(\d+\+?\s*\+?\s*years?[^\n.]{0,80})", text, flags=re.I):
        exp.append(m.group(1).strip())

    title = "Unknown Role"
    first = text.strip().splitlines()[0].strip() if text.strip() else ""
    if 3 < len(first) < 80:
        title = first

    return JDRequirements(
        role_title=title,
        must_have_skills=must or found,
        nice_to_have_skills=nice,
        experience_requirements=exp[:5],
        responsibilities=[],
        keywords=found,
    )


def heuristic_resume_profile(resume_text: str) -> ResumeProfile:
    lines = [ln.strip() for ln in resume_text.splitlines() if ln.strip()]
    name = lines[0] if lines else None
    headline = lines[1] if len(lines) > 1 else None
    skills = []
    lower = resume_text.lower()
    for skill in SKILL_LEXICON:
        if re.search(r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])", lower):
            skills.append(skill)
    return ResumeProfile(
        full_name=name,
        headline=headline,
        skills=skills[:30],
        tools_and_technologies=skills[:30],
        experience_highlights=lines[2:8],
    )


def build_deterministic_evaluation(
    *,
    jd_req: JDRequirements,
    resume_profile: ResumeProfile,
    overall: float,
    band: str,
    skill_sims: list[SkillSimilarity],
    missing: list[str],
    partial: list[str],
    index: ResumeVectorIndex,
) -> EvaluationResult:
    matched_skills = [
        SkillMatchItem(
            skill=s.skill,
            jd_requirement=s.skill,
            present_in_resume=s.present,
            match_score=s.score_0_100,
            evidence=s.best_chunk,
        )
        for s in skill_sims
    ]
    strengths = [s.skill for s in skill_sims if s.score_0_100 >= 75][:8]
    tips = [
        ImprovementTip(
            priority=i + 1,
            category="skill",
            tip=f"Add measurable evidence for '{skill}' (project bullet, metric, or certification).",
            rationale=f"Embedding similarity for '{skill}' is below the presence threshold against the JD.",
        )
        for i, skill in enumerate(missing[:5])
    ]
    for i, skill in enumerate(partial[:3], start=len(tips) + 1):
        tips.append(
            ImprovementTip(
                priority=i,
                category="presentation",
                tip=f"Strengthen wording around '{skill}' with tools used, scale, and outcomes.",
                rationale="Partial embedding match suggests the skill is implied but not strongly evidenced.",
            )
        )

    gaps: list[ExperienceGap] = []
    for req in jd_req.experience_requirements[:4]:
        evidence = index.retrieve_context(req, k=1)
        sim = index.skill_similarity(req)
        severity = "high" if sim.score_0_100 < 45 else "medium" if sim.score_0_100 < 70 else "low"
        if severity != "low":
            gaps.append(
                ExperienceGap(
                    area="Experience requirement",
                    jd_requirement=req,
                    resume_evidence=evidence[:240] if evidence else None,
                    gap_severity=severity,  # type: ignore[arg-type]
                    explanation=(
                        "JD experience requirement is weakly aligned with retrieved resume evidence."
                        if severity != "low"
                        else "Requirement appears supported."
                    ),
                )
            )
    for skill in missing[:3]:
        gaps.append(
            ExperienceGap(
                area=f"Skill depth: {skill}",
                jd_requirement=f"Demonstrated experience with {skill}",
                resume_evidence=None,
                gap_severity="high",
                explanation=f"No strong resume evidence for required skill '{skill}'.",
            )
        )

    return EvaluationResult(
        candidate_summary=(
            f"{resume_profile.full_name or 'Candidate'}"
            + (f" — {resume_profile.headline}" if resume_profile.headline else "")
            + ". Profile inferred from parsed resume text."
        ),
        role_summary=f"Target role: {jd_req.role_title}. Focus skills: {', '.join((jd_req.must_have_skills or [])[:6])}.",
        overall_match_percentage=overall,
        match_band=band,  # type: ignore[arg-type]
        matched_skills=matched_skills,
        missing_skills=missing,
        partial_skills=partial,
        experience_gaps=gaps,
        strengths=strengths,
        improvement_tips=tips,
        scoring_notes=(
            "overall_match_percentage is a weighted average of embedding cosine similarities "
            "between JD skills and resume chunks (must-have 75% / nice-to-have 25%), "
            "with lexical boosts for exact mentions. LLM enrichment unavailable or failed; "
            "deterministic RAG-backed evaluation used."
        ),
    )


class ResumeJDMatcherAgent:
    def __init__(self):
        self.provider = get_llm_provider_name()
        self.llm = None
        self.llm_error: str | None = None
        if self.provider != "none":
            try:
                self.llm = build_chat_llm(temperature=0.0)
            except Exception as exc:  # noqa: BLE001
                self.llm_error = str(exc)
                self.llm = None

    def extract_jd_requirements(self, jd_text: str) -> JDRequirements:
        if self.llm is not None:
            try:
                chain = JD_EXTRACT_PROMPT | _structured(self.llm, JDRequirements)
                result = chain.invoke({"jd_text": jd_text.strip()})
                jd = result if isinstance(result, JDRequirements) else JDRequirements.model_validate(result)
                if jd.must_have_skills or jd.keywords:
                    return jd
            except Exception as exc:  # noqa: BLE001
                self.llm_error = str(exc)
        return heuristic_jd_requirements(jd_text)

    def extract_resume_profile(self, index: ResumeVectorIndex, resume_text: str) -> ResumeProfile:
        if self.llm is not None:
            try:
                queries = [
                    "candidate name title summary objective",
                    "skills technologies tools programming languages",
                    "work experience roles achievements years",
                    "education certifications projects",
                ]
                parts = [index.retrieve_context(q, k=3) for q in queries]
                sample = resume_text[:3500]
                context = "FULL RESUME SAMPLE:\n" + sample + "\n\nRETRIEVED EXCERPTS:\n" + "\n\n".join(parts)
                chain = RESUME_EXTRACT_PROMPT | _structured(self.llm, ResumeProfile)
                result = chain.invoke({"resume_context": context[:12000]})
                return result if isinstance(result, ResumeProfile) else ResumeProfile.model_validate(result)
            except Exception as exc:  # noqa: BLE001
                self.llm_error = str(exc)
        return heuristic_resume_profile(resume_text)

    def evaluate(self, resume: ParsedResume, jd_text: str) -> EvaluationResult:
        if not jd_text or len(jd_text.strip()) < 40:
            raise ValueError("Job description is too short. Paste a fuller JD.")

        jd_req = self.extract_jd_requirements(jd_text)
        index = ResumeVectorIndex(resume.text)

        must = jd_req.must_have_skills or jd_req.keywords
        nice = jd_req.nice_to_have_skills
        if not must:
            # Last resort: score lexicon skills mentioned in JD.
            jd_req = heuristic_jd_requirements(jd_text)
            must = jd_req.must_have_skills or jd_req.keywords
            nice = jd_req.nice_to_have_skills

        overall, skill_sims, missing, partial = compute_match_percentage(
            must_have=must,
            nice_to_have=nice,
            index=index,
        )
        band = match_band(overall)
        resume_profile = self.extract_resume_profile(index, resume.text)

        # Authoritative embedding fields always applied after any LLM pass.
        def _apply_scores(result: EvaluationResult) -> EvaluationResult:
            result.overall_match_percentage = overall
            result.match_band = band  # type: ignore[assignment]
            result.matched_skills = [
                SkillMatchItem(
                    skill=s.skill,
                    jd_requirement=s.skill,
                    present_in_resume=s.present,
                    match_score=s.score_0_100,
                    evidence=s.best_chunk,
                )
                for s in skill_sims
            ]
            result.missing_skills = missing
            result.partial_skills = partial
            if not result.scoring_notes:
                result.scoring_notes = (
                    "overall_match_percentage is a weighted average of embedding cosine "
                    "similarities between JD skills and resume chunks "
                    "(must-have 75% / nice-to-have 25%), with lexical boosts for exact mentions."
                )
            if not result.improvement_tips and missing:
                result.improvement_tips = [
                    ImprovementTip(
                        priority=i + 1,
                        category="skill",
                        tip=f"Add concrete evidence for '{skill}'.",
                        rationale=f"'{skill}' is required by the JD but weakly evidenced.",
                    )
                    for i, skill in enumerate(missing[:5])
                ]
            if not result.improvement_tips:
                result.improvement_tips = [
                    ImprovementTip(
                        priority=1,
                        category="presentation",
                        tip="Quantify impact in top bullets (latency, cost, revenue, scale).",
                        rationale="Even strong matches improve when achievements are measurable.",
                    ),
                    ImprovementTip(
                        priority=2,
                        category="keyword",
                        tip="Mirror exact JD skill phrasing in a Skills section for ATS and recruiters.",
                        rationale="Lexical alignment improves both human and embedding recall.",
                    ),
                ]
            return result

        if self.llm is not None:
            try:
                gap_queries = list(jd_req.experience_requirements) + missing[:8] + [
                    "years of experience leadership management",
                    "cloud production scale architecture",
                ]
                rag_bits = [f"Query: {q}\n{index.retrieve_context(str(q), k=2)}" for q in gap_queries[:10]]
                skill_scores_json = [
                    {
                        "skill": s.skill,
                        "match_score": s.score_0_100,
                        "present_in_resume": s.present,
                        "evidence": s.best_chunk,
                    }
                    for s in skill_sims
                ]
                chain = EVAL_PROMPT | _structured(self.llm, EvaluationResult)
                raw = chain.invoke(
                    {
                        "embedding_match_percentage": overall,
                        "band": band,
                        "missing_skills": missing,
                        "partial_skills": partial,
                        "jd_json": jd_req.model_dump_json(indent=2),
                        "resume_json": resume_profile.model_dump_json(indent=2),
                        "skill_scores_json": skill_scores_json,
                        "rag_context": "\n\n".join(rag_bits)[:10000],
                    }
                )
                result = raw if isinstance(raw, EvaluationResult) else EvaluationResult.model_validate(raw)
                return _apply_scores(result)
            except Exception as exc:  # noqa: BLE001
                self.llm_error = str(exc)

        return build_deterministic_evaluation(
            jd_req=jd_req,
            resume_profile=resume_profile,
            overall=overall,
            band=band,
            skill_sims=skill_sims,
            missing=missing,
            partial=partial,
            index=index,
        )

    def run(self, file_bytes: bytes, filename: str, jd_text: str) -> dict[str, Any]:
        parsed = parse_resume_bytes(file_bytes, filename)
        evaluation = self.evaluate(parsed, jd_text)
        return {
            "provider": self.provider,
            "llm_error": self.llm_error,
            "resume_meta": {
                "filename": parsed.filename,
                "page_count": parsed.page_count,
                "char_count": parsed.char_count,
                "parser": parsed.parser,
            },
            "evaluation": evaluation.model_dump(),
            "evaluation_json": evaluation.model_dump_json(indent=2),
        }
