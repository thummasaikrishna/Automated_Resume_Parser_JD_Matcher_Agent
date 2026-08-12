"""
Pydantic schemas for structured resume ↔ JD evaluation output.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SkillMatchItem(BaseModel):
    """Per-skill embedding match between JD requirement and resume evidence."""

    skill: str = Field(..., description="Normalized skill or requirement name")
    jd_requirement: str = Field(..., description="How the JD states this requirement")
    present_in_resume: bool = Field(..., description="Whether resume shows credible evidence")
    match_score: float = Field(..., ge=0, le=100, description="Embedding similarity 0-100")
    evidence: str = Field("", description="Short resume snippet supporting the match")

    @field_validator("match_score")
    @classmethod
    def round_score(cls, v: float) -> float:
        return round(float(v), 2)


class ExperienceGap(BaseModel):
    """A concrete experience or seniority gap vs the JD."""

    area: str = Field(..., description="Gap theme, e.g. leadership, cloud ops, years of experience")
    jd_requirement: str = Field(..., description="What the JD asks for")
    resume_evidence: str | None = Field(None, description="Closest resume evidence, if any")
    gap_severity: Literal["high", "medium", "low"] = Field(..., description="How critical the gap is")
    explanation: str = Field(..., description="Why this is a gap")


class ImprovementTip(BaseModel):
    """Actionable, tailored advice to improve match quality."""

    priority: int = Field(..., ge=1, le=10, description="1 = highest priority")
    category: Literal["skill", "experience", "presentation", "keyword", "project"] = Field(
        ..., description="Tip category"
    )
    tip: str = Field(..., description="Concrete improvement action")
    rationale: str = Field(..., description="Why this tip improves JD match")


class EvaluationResult(BaseModel):
    """
    Structured JSON evaluation returned by the Resume Parser & JD Matcher Agent.
    """

    candidate_summary: str = Field(..., description="1-3 sentence profile summary from the resume")
    role_summary: str = Field(..., description="1-2 sentence summary of the target role from the JD")
    overall_match_percentage: float = Field(
        ..., ge=0, le=100, description="Weighted embedding-based skill match percentage"
    )
    match_band: Literal["excellent", "strong", "moderate", "weak", "poor"] = Field(
        ..., description="Human-readable match band"
    )
    matched_skills: list[SkillMatchItem] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    partial_skills: list[str] = Field(default_factory=list)
    experience_gaps: list[ExperienceGap] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    improvement_tips: list[ImprovementTip] = Field(default_factory=list)
    scoring_notes: str = Field(
        "",
        description="Brief explanation of how the match percentage was derived",
    )

    @field_validator("overall_match_percentage")
    @classmethod
    def round_overall(cls, v: float) -> float:
        return round(float(v), 2)


class JDRequirements(BaseModel):
    """LLM-extracted structured requirements from a job description."""

    role_title: str = Field("Unknown Role")
    must_have_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    experience_requirements: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class ResumeProfile(BaseModel):
    """LLM-extracted structured profile from resume text (RAG-grounded)."""

    full_name: str | None = None
    headline: str | None = None
    years_experience_estimate: float | None = Field(
        None, description="Estimated total years of experience if inferable"
    )
    skills: list[str] = Field(default_factory=list)
    tools_and_technologies: list[str] = Field(default_factory=list)
    experience_highlights: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
