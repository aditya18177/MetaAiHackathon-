"""
server/app.py — OpenEnv-compliant server entry point.
Uses openenv-core's create_fastapi_app to expose reset/step/state endpoints
and adds /tasks + /grade/{task_id} for grader validation.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.responses import JSONResponse

try:
    from openenv.core.env_server import create_fastapi_app
    _USE_OPENENV_CORE = True
except ImportError:
    _USE_OPENENV_CORE = False

from env import DataWranglerEnv
from models import DataWranglerAction, ResetRequest

# ── Per-task isolated env instances ──────────────────────────────────────────
_envs = {
    "easy":          DataWranglerEnv(),
    "medium":        DataWranglerEnv(),
    "hard":          DataWranglerEnv(),
    "easy_sample":   DataWranglerEnv(),
    "medium_sample": DataWranglerEnv(),
    "hard_sample":   DataWranglerEnv(),
}

# Initialise all envs so df is never None when grader is called
for _tid, _e in _envs.items():
    _e.reset(task_id=_tid)

_shared_env = _envs["easy"]

TASK_METADATA = [
    {"id": "easy",          "difficulty": "easy",   "has_grader": True},
    {"id": "medium",        "difficulty": "medium", "has_grader": True},
    {"id": "hard",          "difficulty": "hard",   "has_grader": True},
    {"id": "easy_sample",   "difficulty": "easy",   "has_grader": True},
    {"id": "medium_sample", "difficulty": "medium", "has_grader": True},
    {"id": "hard_sample",   "difficulty": "hard",   "has_grader": True},
]

# ── Build app ─────────────────────────────────────────────────────────────────
if _USE_OPENENV_CORE:
    app = create_fastapi_app(_shared_env)
else:
    app = FastAPI(title="DataWrangler-Env")


@app.get("/")
def health():
    return {"status": "ok", "env": "DataWrangler-Env"}


@app.get("/tasks")
def list_tasks():
    return JSONResponse(content={"tasks": TASK_METADATA})


@app.post("/reset")
def reset(body: ResetRequest = None):
    req = body or ResetRequest()
    global _shared_env
    _shared_env = _envs.get(req.task_id, _envs["easy"])
    obs = _shared_env.reset(task_id=req.task_id)
    return JSONResponse(content=obs.dict())


@app.post("/step")
async def step(action: DataWranglerAction):
    obs, reward, done, info = _shared_env.step(action)
    return JSONResponse(content={
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "info": info
    })


@app.get("/state")
def state():
    return JSONResponse(content=_shared_env.state().dict())


@app.post("/grade/{task_id}")
def grade(task_id: str):
    """Grader endpoint — returns score strictly in (0, 1)."""
    if task_id not in _envs:
        return JSONResponse(status_code=404, content={"error": f"Task '{task_id}' not found."})
    score = _envs[task_id].get_score()
    return JSONResponse(content={
        "task_id": task_id,
        "score": score,
        "score_range": {"min": 0.0, "max": 1.0, "exclusive": True},
    })


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
