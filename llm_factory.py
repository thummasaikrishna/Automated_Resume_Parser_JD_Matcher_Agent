"""
LLM + embeddings factory.

Groq is the primary chat model.
Supports local `.env` and Streamlit Cloud secrets.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _secret(name: str, default: str | None = None) -> str | None:
    """Read from Streamlit secrets first, then environment variables."""
    try:
        import streamlit as st

        if hasattr(st, "secrets") and name in st.secrets:
            value = st.secrets.get(name)
            if value is not None and str(value).strip():
                return str(value).strip()
    except Exception:
        pass
    value = os.getenv(name, default)
    if value is None:
        return default
    value = str(value).strip()
    return value or default


def get_llm_provider_name() -> str:
    """Groq is the default provider for this agent."""
    if _secret("GROQ_API_KEY"):
        return "groq"
    if _secret("OPENAI_API_KEY"):
        return "openai"
    if _secret("OLLAMA_BASE_URL") or (_secret("USE_OLLAMA", "") or "").lower() in {
        "1",
        "true",
        "yes",
    }:
        return "ollama"
    return "none"


def build_chat_llm(temperature: float = 0.0) -> Any:
    provider = get_llm_provider_name()

    if provider == "groq":
        from langchain_groq import ChatGroq

        model = _secret("GROQ_MODEL", "llama-3.3-70b-versatile") or "llama-3.3-70b-versatile"
        return ChatGroq(
            model=model,
            temperature=temperature,
            groq_api_key=_secret("GROQ_API_KEY"),
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        model = _secret("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
        return ChatOpenAI(model=model, temperature=temperature)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        model = _secret("OLLAMA_MODEL", "llama3.1") or "llama3.1"
        base_url = _secret("OLLAMA_BASE_URL", "http://127.0.0.1:11434") or "http://127.0.0.1:11434"
        return ChatOllama(model=model, temperature=temperature, base_url=base_url)

    raise RuntimeError(
        "No LLM configured. Set GROQ_API_KEY in `.env` locally or in Streamlit Cloud Secrets."
    )


def build_embeddings() -> Any:
    """
    Local HuggingFace embeddings by default.
    Set USE_LOCAL_EMBEDDINGS=false with OPENAI_API_KEY to use OpenAI embeddings.
    """
    use_local = (_secret("USE_LOCAL_EMBEDDINGS", "true") or "true").lower() in {
        "1",
        "true",
        "yes",
        "",
    }
    if not use_local and _secret("OPENAI_API_KEY"):
        from langchain_openai import OpenAIEmbeddings

        model = _secret("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small") or "text-embedding-3-small"
        return OpenAIEmbeddings(model=model)

    from langchain_huggingface import HuggingFaceEmbeddings

    model_name = (
        _secret("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        or "sentence-transformers/all-MiniLM-L6-v2"
    )
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )
