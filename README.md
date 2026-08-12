# Automated Resume Parser & JD Matcher Agent

Upload a **PDF / image resume** and a **job description** to get:

- skill-match **percentage** (vector embeddings + cosine similarity)
- **experience gaps** with retrieved evidence
- **structured JSON** (Pydantic) and improvement tips

Live app entrypoint: `streamlit_app.py`

## Tech stack

| Layer | Technology | Purpose |
| --- | --- | --- |
| Language | **Python 3.12** | Core development |
| PDF Parsing | **PyMuPDF** / **pypdf** + RapidOCR | Extract resume text (including scans) |
| LLM Framework | **LangChain** | Parsing, embeddings, prompts |
| LLM | **Groq** | Resume/JD analysis & recommendations |
| Embeddings | **HuggingFace** MiniLM | Semantic skill matching |
| Retrieval | **Cosine similarity** | Rank resume chunks (no FAISS) |
| Structured Output | **Pydantic** | Validate evaluation JSON |
| UI | **Streamlit** | Web interface |
| Similarity | **Cosine Similarity** | Match score |
| Data Handling | **Pandas** | Result tables |
| Environment | **python-dotenv** / Streamlit Secrets | API key management |
| Testing | **pytest** | Component tests |
| Version Control | **Git + GitHub** | Portfolio / Cloud deploy |

## Local setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `.env` (never commit this file):

```env
GROQ_API_KEY=your_groq_key_here
GROQ_MODEL=llama-3.3-70b-versatile
USE_LOCAL_EMBEDDINGS=true
```

Run:

```powershell
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud

See [DEPLOY.md](DEPLOY.md). **Critical:** in app **Advanced settings**, set **Python to 3.12** (Cloud may ignore `runtime.txt`), then reboot.

There is **no** `packages.txt` — `opencv-python-headless` works without apt packages. Do **not** add `libglib2.0-0` (breaks on Debian Trixie).

## Project layout

```
streamlit_app.py       # UI
matcher_agent.py       # Agent pipeline
embeddings_matcher.py  # Pure cosine RAG + skill scoring
pdf_parser.py          # PyMuPDF / pypdf / OCR
agent_models.py        # Pydantic schemas
llm_factory.py         # Groq + embeddings (+ Streamlit secrets)
runtime.txt            # python-3.12 hint for Cloud
.python-version        # 3.12
.streamlit/            # Theme + secrets example
tests/                 # pytest suite
```
