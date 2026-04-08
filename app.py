import os
import json
import logging
import gradio as gr
from openai import OpenAI
from env import DataWranglerEnv
from models import DataWranglerAction

logging.basicConfig(level=logging.INFO, format='%(message)s')


def run_inference(hf_token: str, model_name: str, api_base: str):
    if not hf_token.strip():
        return "Error: HF_TOKEN is required."

    client = OpenAI(base_url=api_base.strip(), api_key=hf_token.strip())
    env = DataWranglerEnv()

    tasks = ["easy", "medium", "hard"]
    logs = []
    total_cumulative_reward = 0.0
    final_results = []

    for task_id in tasks:
        obs = env.reset(task_id=task_id)
        done = False
        step_count = 0
        task_reward = 0.0

        while not done and step_count < 10:
            step_count += 1
            prompt = f"""
You are a data cleaning agent. Your task is: {obs.task_goal}

Current Data Snapshot (First 5 rows):
{json.dumps(obs.head, indent=2)}

Column Info:
{json.dumps([col.dict() for col in obs.columns], indent=2)}

Last Action Status: {obs.last_action_status}
{f"Last Error: {obs.last_error}" if obs.last_error else ""}

Select one action:
- drop_nulls(column_name)
- fill_nulls(column_name, fill_value)
- cast_type(column_name, new_type)
- drop_column(column_name)
- rename_column(old_name, new_name)
- submit()

Respond ONLY with JSON: {{"command": {{"action_type": "...", ...}}}}
"""
            try:
                response = client.chat.completions.create(
                    model=model_name.strip(),
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                action_data = json.loads(response.choices[0].message.content)
                action = DataWranglerAction(**action_data)
                obs, reward, done, info = env.step(action)
                task_reward += reward
                total_cumulative_reward += reward

                log_entry = {
                    "task_id": task_id,
                    "step": step_count,
                    "action": action.command.action_type,
                    "reward": reward,
                    "done": done,
                }
                logs.append(json.dumps(log_entry))
            except Exception as e:
                logs.append(f"[ERROR] Task: {task_id}, Step: {step_count}, Error: {str(e)}")
                break

        final_score = env.get_score()
        final_results.append({
            "task_id": task_id,
            "final_score": final_score,
            "total_steps": step_count,
            "task_reward": round(task_reward, 4),
        })

    summary = {
        "final_results": final_results,
        "total_cumulative_reward": round(total_cumulative_reward, 4),
    }
    output = "\n".join(logs) + "\n\n[SUMMARY]\n" + json.dumps(summary, indent=2)
    return output


with gr.Blocks(title="DataWrangler-Env") as demo:
    gr.Markdown("# 🧹 DataWrangler-Env\nAn AI agent benchmark for data cleaning tasks.")

    with gr.Row():
        hf_token = gr.Textbox(label="HF Token", type="password", placeholder="hf_...")
        model_name = gr.Textbox(label="Model Name", value="meta-llama/Llama-3-70B-Instruct")
        api_base = gr.Textbox(label="API Base URL", value="https://api-inference.huggingface.co/v1/")

    run_btn = gr.Button("Run Inference", variant="primary")
    output = gr.Textbox(label="Results", lines=30, interactive=False)

    run_btn.click(fn=run_inference, inputs=[hf_token, model_name, api_base], outputs=output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
