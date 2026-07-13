"""JSON screening API over ``ScreeningSystem.predict``.

Usage:
    uvicorn src.api:app --reload            # dev server on http://127.0.0.1:8000
    # POST /screen  {"text": "..."}  ->  a stable public verdict payload
    # GET  /health                    ->  {"status": "ok", "models_loaded": bool}

This is the productization seam: the same screener that backs the Streamlit demo
exposed as an endpoint a browser extension or a batch client can call. Two
design choices keep it honest and maintainable:

* **A stable public contract, decoupled from internals.** :func:`build_response`
  reshapes the rich internal ``predict`` dict into a small, documented schema, so
  the API doesn't leak (or freeze) internal field names. It is pure — testable
  with a synthetic prediction, no models or server needed.
* **Validation before the model.** :func:`validate_request` bounds the input
  (present, non-empty, under ``API_MAX_TEXT_CHARS``) so a bad request is a clean
  400, not a model crash. Also pure.

Only :func:`create_app` imports FastAPI and loads the models (once, at startup).
"""
from __future__ import annotations

from . import config


def validate_request(payload) -> tuple[str, list[str]]:
    """Return ``(text, errors)``. ``errors`` empty means the request is valid."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return "", ["body must be a JSON object"]
    text = payload.get("text")
    if not isinstance(text, str):
        return "", ["'text' is required and must be a string"]
    stripped = text.strip()
    if len(stripped) < config.API_MIN_TEXT_CHARS:
        errors.append("'text' must not be empty")
    if len(text) > config.API_MAX_TEXT_CHARS:
        errors.append(f"'text' exceeds {config.API_MAX_TEXT_CHARS} characters")
    return text, errors


def build_response(result: dict) -> dict:
    """Shape a ``ScreeningSystem.predict`` result into the public API schema.

    Pure: pass any dict shaped like predict()'s output (missing optional keys are
    tolerated) and get back the stable contract the clients depend on.
    """
    manip = result.get("manipulation") or {}
    ai_style = result.get("ai_style") or {}
    live = result.get("live") or None
    expl = result.get("explanation") or {}
    neural = result.get("explanation_neural") or {}
    rag = result.get("explanation_rag") or {}

    def _tokens(block, key):
        return [c.get("token") for c in (block.get(key) or [])][:6]

    return {
        "verdict": result.get("verdict"),
        "fake_probability": result.get("fake_probability"),
        "confidence": result.get("confidence"),
        "needs_review": bool(result.get("needs_review")),
        "evidence_backed": bool(result.get("evidence_backed")),
        "reason": result.get("reason"),
        "signals": {
            "manipulation": {
                "count": manip.get("count", 0),
                "high": bool(manip.get("high")),
                "techniques": [t.get("label") for t in (manip.get("techniques") or [])],
            },
            "fabricated_authority": {
                "count": ai_style.get("count", 0),
                "high": bool(ai_style.get("high")),
            },
            "live_factcheck": (
                {"verdict": live.get("verdict"), "source": live.get("source")}
                if isinstance(live, dict) and live.get("verdict") else None
            ),
        },
        "explanation": {
            "svm_top_fake": _tokens(expl, "fake_pushing"),
            "svm_top_real": _tokens(expl, "real_pushing"),
            "rnn_top_fake": _tokens(neural, "fake_pushing"),
            "rnn_top_real": _tokens(neural, "real_pushing"),
            "evidence": rag.get("driving") or [],
        },
        "disclaimer": "Screening aid, not a verdict — a prompt for human "
                      "verification. See needs_review and the evidence.",
    }


def create_app():
    """Build the FastAPI app. Loads the screening system once at startup."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    from .predict import ScreeningSystem

    app = FastAPI(title="Fake-News Screening API", version="1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.API_ALLOWED_ORIGINS,
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
    )

    state: dict = {"system": None}

    @app.on_event("startup")
    def _load():
        # Live retrieval off by default: an API call should be fast and
        # deterministic; enable per-deployment if you want live evidence.
        state["system"] = ScreeningSystem(with_live=False)

    class ScreenRequest(BaseModel):
        text: str

    @app.get("/health")
    def health():
        return {"status": "ok", "models_loaded": state["system"] is not None}

    @app.post("/screen")
    def screen(req: ScreenRequest):
        text, errors = validate_request(req.model_dump())
        if errors:
            return JSONResponse(status_code=400, content={"errors": errors})
        result = state["system"].predict(text)
        return build_response(result)

    return app


# Importing FastAPI lazily means ``uvicorn src.api:app`` still needs it present;
# module import stays cheap for the pure helpers and their tests.
def __getattr__(name):
    if name == "app":
        return create_app()
    raise AttributeError(name)
