"""
server/app.py — OpenEnv server entry point.
"""
import sys
import os

# Ensure root is on path so env/models/tasks are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from app import app  # noqa: F401 — re-export FastAPI app


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
