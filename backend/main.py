"""
MatPad — Matrix Operations Web Application
FastAPI backend

Run from the matpad/ project root:
    uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .operations import CATALOG
from .parser import parse_and_evaluate


# ─── Custom JSON encoding for numpy types ────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.complexfloating):
            return {"re": float(obj.real), "im": float(obj.imag)}
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _json_safe(obj: Any) -> Any:
    """Round-trip through the custom encoder to make any value JSON-safe."""
    return json.loads(json.dumps(obj, cls=_NumpyEncoder))


# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="MatPad", description="Matrix Operations Web Application")


# ─── API ──────────────────────────────────────────────────────────────────────

@app.get("/api/operations")
def get_operations():
    """Return the operations catalog for the capabilities panel and autocomplete."""
    return [
        {
            "name": e.name,
            "operator": e.operator,
            "description": e.description,
        }
        for e in CATALOG
    ]


class ComputeRequest(BaseModel):
    matrices: Dict[str, Any]   # {name: [[row, …], …]}
    expression: str


@app.post("/api/compute")
def compute(req: ComputeRequest):
    """Parse the expression and evaluate it against the supplied matrices."""
    if not req.expression.strip():
        return JSONResponse({"results": [], "error": None})

    # Validate and normalise each matrix
    matrices: Dict[str, list] = {}
    for name, data in req.matrices.items():
        try:
            arr = np.array(data, dtype=float)
        except (ValueError, TypeError) as exc:
            return JSONResponse(
                {"results": [], "error": f"Invalid data for matrix '{name}': {exc}"}
            )
        if arr.ndim != 2:
            return JSONResponse(
                {"results": [], "error": f"Matrix '{name}' must be 2-dimensional (got shape {arr.shape})"}
            )
        matrices[name] = arr.tolist()

    results = parse_and_evaluate(req.expression, matrices)
    return JSONResponse(_json_safe({"results": results, "error": None}))


# ─── Serve frontend ───────────────────────────────────────────────────────────

_FRONTEND_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "frontend")
)

if os.path.isdir(_FRONTEND_DIR):
    # Mount static assets (CSS, JS) under /static to avoid catching /api/*
    _static_css = os.path.join(_FRONTEND_DIR, "css")
    _static_js  = os.path.join(_FRONTEND_DIR, "js")

    if os.path.isdir(_static_css):
        app.mount("/css", StaticFiles(directory=_static_css), name="css")
    if os.path.isdir(_static_js):
        app.mount("/js", StaticFiles(directory=_static_js), name="js")

    @app.get("/favicon.svg", include_in_schema=False)
    def serve_favicon():
        favicon = os.path.join(_FRONTEND_DIR, "favicon.svg")
        if os.path.isfile(favicon):
            return FileResponse(favicon, media_type="image/svg+xml")
        return JSONResponse({"error": "Not found"}, status_code=404)

    @app.get("/", include_in_schema=False)
    @app.get("/{path:path}", include_in_schema=False)
    def serve_frontend(path: str = ""):
        index = os.path.join(_FRONTEND_DIR, "index.html")
        if os.path.isfile(index):
            return FileResponse(index)
        return JSONResponse({"error": "Frontend not found"}, status_code=404)
