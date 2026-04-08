import os
import json
import subprocess
import threading
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import gradio as gr
import uvicorn
from env import DataWranglerEnv
from models import DataWranglerAction

logging.basicConfig(level=logging.INFO, format='%(message)s')

# --- FastAPI app for OpenEnv HTTP endpoints ---
api = FastAPI()
_env = DataWranglerEnv()


@api.get("/")
def health():
    """HF Space ping — must return 200."""
    return {"status": "ok", "env": "DataWrangler-Env"}


@api.post("/reset")
def reset(request_body: dict = None):
    task_id = (request_body or {}).get("task_id", "easy")
    obs = _env.reset(task_id=task_id)
    return JSONResponse(content=obs.dict())


@api.post("/step")
async def step(request: Request):
    body = await request.json()
    action = DataWranglerAction(**body)
    obs, reward, done, info = _env.step(action)
    return JSONResponse(content={
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "info": info
    })


@api.get("/state")
def state():
    return JSONResponse(content=_env.state())


# --- Gradio UI ---
def run_inference_ui(hf_token: str, model_name: str, api_base: str):
    if not hf_token.strip():
        return "Error: HF_TOKEN is required."

    env = {
        **os.environ,
        "HF_TOKEN": hf_token.strip(),
        "MODEL_NAME": model_name.strip(),
        "API_BASE_URL": api_base.strip(),
    }

    try:
        result = subprocess.run(
            ["python", "inference.py"],
            capture_output=True,
            text=True,
            timeout=1200,  # 20 min limit
            env=env
        )
        output = result.stdout
        if result.stderr:
            output += "\n[STDERR]\n" + result.stderr
        return output
    except subprocess.TimeoutExpired:
        return "[ERROR] Inference timed out after 20 minutes."
    except Exception as e:
        return f"[ERROR] {str(e)}"


with gr.Blocks(title="DataWrangler-Env") as demo:
    gr.Markdown("# 🧹 DataWrangler-Env\nAI agent benchmark for data cleaning tasks (OpenEnv compliant).")
    gr.Markdown("Provide your credentials below and click **Run Inference** to evaluate the agent across all 3 tasks.")

    with gr.Row():
        hf_token = gr.Textbox(label="HF Token", type="password", placeholder="hf_...")
        model_name = gr.Textbox(label="Model Name", value="meta-llama/Llama-3-70B-Instruct")
        api_base = gr.Textbox(label="API Base URL", value="https://api-inference.huggingface.co/v1/")

    run_btn = gr.Button("Run Inference", variant="primary")
    output = gr.Textbox(label="Structured Logs + Results", lines=35, interactive=False)

    run_btn.click(fn=run_inference_ui, inputs=[hf_token, model_name, api_base], outputs=output)


# --- Mount Gradio into FastAPI and launch ---
app = gr.mount_gradio_app(api, demo, path="/ui")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
