# Deploy to Streamlit Community Cloud

Repo: https://github.com/thummasaikrishna/Automated_Resume_Parser_JD_Matcher_Agent

## Critical: Python version

In the Streamlit app **Advanced settings**, set **Python version to 3.12**.

Do **not** leave the default on 3.13/3.14 — `rapidocr` / some ML wheels will fail.

`runtime.txt` and `.python-version` are also set to 3.12 in the repo.

## packages.txt

Do **not** restore `packages.txt` with `libglib2.0-0` (fails on Debian Trixie). Prefer no `packages.txt` while `opencv-python-headless` works without apt packages.

## Steps

1. Open https://share.streamlit.io and sign in with GitHub.
2. App settings for `resumematcher2026` (or New app):
   - **Repository:** `thummasaikrishna/Automated_Resume_Parser_JD_Matcher_Agent`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
   - **Python version:** `3.12`
3. **Secrets:**

```toml
GROQ_API_KEY = "paste_your_key_here"
GROQ_MODEL = "llama-3.3-70b-versatile"
USE_LOCAL_EMBEDDINGS = "true"
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

4. **Reboot** / **Redeploy** the app.
5. First build can take several minutes (sentence-transformers / OCR).

Public URL: `https://resumematcher2026.streamlit.app`
