# Deploy to Streamlit Community Cloud

Repo: https://github.com/thummasaikrishna/Automated_Resume_Parser_JD_Matcher_Agent

## Steps

1. Open https://share.streamlit.io and sign in with GitHub (`thummasaikrishna`).
2. Click **Create app** / **New app**.
3. Set:
   - **Repository:** `thummasaikrishna/Automated_Resume_Parser_JD_Matcher_Agent`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
   - **App URL** (optional): `automated-resume-parser-jd-matcher-agent`
4. Open **Advanced settings → Secrets** and paste (use your real Groq key from local `.env`):

```toml
GROQ_API_KEY = "paste_your_key_here"
GROQ_MODEL = "llama-3.3-70b-versatile"
USE_LOCAL_EMBEDDINGS = "true"
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

5. Click **Deploy**.
6. First build can take 5–15 minutes (torch / sentence-transformers / OCR models).

Expected public URL pattern:

`https://automated-resume-parser-jd-matcher-agent.streamlit.app`

(or the custom subdomain Streamlit assigns)
