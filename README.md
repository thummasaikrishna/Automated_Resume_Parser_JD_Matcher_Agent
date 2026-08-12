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
| Vector Store | **FAISS** | Store/search resume chunks |
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

1. Push this repo to GitHub (`Automated_Resume_Parser_JD_Matcher_Agent`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select the repo, branch `main`, main file `streamlit_app.py`.
4. Under **Advanced settings → Secrets**, paste:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
GROQ_MODEL = "llama-3.3-70b-versatile"
USE_LOCAL_EMBEDDINGS = "true"
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

5. Deploy. First boot may take several minutes (embeddings + OCR models).

`packages.txt` installs Linux libs needed by OpenCV/OCR on Streamlit Cloud.

## Project layout

```
streamlit_app.py       # UI
matcher_agent.py       # Agent pipeline
embeddings_matcher.py  # FAISS + cosine scoring
pdf_parser.py          # PyMuPDF / pypdf / OCR
agent_models.py        # Pydantic schemas
llm_factory.py         # Groq + embeddings (+ Streamlit secrets)
packages.txt           # Streamlit Cloud system packages
.streamlit/            # Theme + secrets example
tests/                 # pytest suite
```
