from matcher_agent import ResumeJDMatcherAgent

resume = b"""Alex Rivera
Senior Data Engineer

Skills: Python, SQL, Apache Spark, Airflow, dbt, AWS, Docker, PostgreSQL, Kafka

Experience:
- Designed Spark ETL pipelines processing 2TB/day on AWS EMR (5 years).
- Built Airflow DAGs and dbt models for analytics warehouse.
- Containerized services with Docker and deployed on ECS.
Education: B.S. Computer Science
"""

jd = """
We are hiring a Senior Data Engineer.
Must have: Python, SQL, Apache Spark, Airflow, AWS.
Nice to have: Kafka, dbt, Kubernetes.
Requirements: 4+ years building production data pipelines, experience with cloud data platforms.
"""

agent = ResumeJDMatcherAgent()
out = agent.run(resume, "resume.txt", jd)
ev = out["evaluation"]
print("provider:", out["provider"])
print("match:", ev["overall_match_percentage"], ev["match_band"])
print("missing:", ev["missing_skills"])
print("gaps:", len(ev["experience_gaps"]))
print("tips:", len(ev["improvement_tips"]))
print("skills_scored:", len(ev["matched_skills"]))
assert 0 <= ev["overall_match_percentage"] <= 100
assert ev["matched_skills"], "expected skill scores"
assert "evaluation_json" in out
# Strong resume should score reasonably high for this JD
assert ev["overall_match_percentage"] >= 60, ev["overall_match_percentage"]
print("E2E_OK")
