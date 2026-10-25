"""Free-tier-capable model choices, not a substitute for project billing controls."""
import logging
import os

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
# Verified against Google's pricing page on 2026-10-25. Recheck on model changes.
GEMINI_MODELS = {DEFAULT_GEMINI_MODEL, "gemini-3.1-flash-lite"}
logger = logging.getLogger(__name__)


def gemini_model():
    requested = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
    if requested not in GEMINI_MODELS:
        logger.warning("Using the approved low-latency Gemini default instead of the configured model.")
        return DEFAULT_GEMINI_MODEL
    return requested


def allow_groq_fallback():
    return os.getenv("RAG_ALLOW_GROQ_FALLBACK", "false").strip().lower() == "true"


def failure_category(exc):
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if not isinstance(status, int):
        status = None
    name = type(exc).__name__.lower()
    if isinstance(exc, TimeoutError) or "timeout" in name or status in {408, 504}:
        category = "timeout"
    elif status == 429:
        category = "rate_limit"
    elif status in {401, 403}:
        category = "access_denied"
    elif status == 404:
        category = "model_unavailable"
    elif status == 400:
        category = "invalid_request"
    elif status is not None and status >= 500:
        category = "service_unavailable"
    elif "connect" in name:
        category = "connection"
    else:
        category = "unavailable"
    return category, status
