import os
import sys
import json
import logging
from openai import OpenAI
from env import DataWranglerEnv
from models import DataWranglerAction

# Configure logging — plain format so [START]/[STEP]/[END] tags are clean
logging.basicConfig(level=logging.INFO, format='%(message)s')


def run_inference():
    # Required env vars per hackathon spec
    api_key = os.getenv("HF_TOKEN")
    api_base = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
    model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3-70B-Instruct")

    if not api_key:
        print("Error: HF_TOKEN must be set in environment variables.")
        sys.exit(1)

    client = OpenAI(base_url=api_base, api_key=api_key)
    env = DataWranglerEnv()

    tasks = ["easy", "medium", "hard"]
    final_results = []
    total_cumulative_reward = 0.0

    print("[START]")

    for task_id in tasks:
        obs = env.reset(task_id=task_id)
        done = False
        step_count = 0
        task_reward = 0.0

        while not done and step_count < 10:
            step_count += 1

            prompt = f"""You are a data cleaning agent. Your task is: {obs.task_goal}

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

Respond ONLY with JSON: {{"command": {{"action_type": "...", ...}}}}"""

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )

                action_data = json.loads(response.choices[0].message.content)
                action = DataWranglerAction(**action_data)

                obs, reward, done, info = env.step(action)

                # Clamp reward to [0.0, 1.0] range for grader compliance
                reward = max(-1.0, min(1.0, reward))
                task_reward += reward
                total_cumulative_reward += reward

                # Strictly formatted [STEP] log — field names and order must not change
                log_step = {
                    "task_id": task_id,
                    "step_number": step_count,
                    "action_taken": action.command.action_type,
                    "reward_received": reward,
                    "is_done": done,
                    "info": info
                }
                print(f"[STEP] {json.dumps(log_step)}")

            except Exception as e:
                print(f"[ERROR] Task: {task_id}, Step: {step_count}, Error: {str(e)}")
                break

        final_score = env.get_score()
        # Clamp final score to [0.0, 1.0]
        final_score = max(0.0, min(1.0, final_score))

        final_results.append({
            "task_id": task_id,
            "final_score": final_score,
            "total_steps": step_count,
            "task_reward": round(task_reward, 4)
        })

    end_report = {
        "final_results": final_results,
        "total_cumulative_reward": round(total_cumulative_reward, 4)
    }
    print(f"[END] {json.dumps(end_report)}")


if __name__ == "__main__":
    run_inference()
