"""
server/app.py — OpenEnv server entry point.
Re-exports the FastAPI app from the root app.py.
"""
import sys
import os

# Ensure root is on path so env/models/tasks are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: F401 — re-export for openenv validate
