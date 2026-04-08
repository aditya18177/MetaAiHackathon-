---
title: DataWrangler-Env
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# DataWrangler-Env: Automated Data Cleaning Simulator

DataWrangler-Env is a high-fidelity environment designed to evaluate AI agents on real-world data cleaning and transformation tasks. It implements the **OpenEnv specification** and provides a set of challenges that mirror common data engineering workflows.

## Environment Overview

In DataWrangler-Env, an agent is presented with a "messy" Pandas DataFrame and a specific goal (e.g., "Standardize dates and fix price formatting"). The agent can perform various actions such as dropping nulls, filling missing values, casting types, and renaming columns.

### Action Space
- `drop_nulls(column_name)`
- `fill_nulls(column_name, fill_value)`
- `cast_type(column_name, new_type)`
- `drop_column(column_name)`
- `rename_column(old_name, new_name)`
- `submit()`

### Observation Space
The agent receives a `DataWranglerObservation` containing:
- `task_goal`: The objective description.
- `total_rows`: Number of rows in the dataset.
- `columns`: Metadata about each column (dtype, null counts, sample values).
- `head`: A JSON preview of the first 5 rows.
- `last_action_status`: Whether the previous action succeeded or errored.

## Tasks & Difficulty

1. **Easy: Sales Data Cleaning**
   - **Difficulty:** Easy
   - **Issues:** String-formatted prices with symbols and null values.
   - **Goal:** Clean and cast the 'Price' column to float.

2. **Medium: User Profile Standardization**
   - **Difficulty:** Medium
   - **Issues:** Duplicate entries, inconsistent date strings, and missing email addresses.
   - **Goal:** Deduplicate, standardize dates, and fill missing emails.

3. **Hard: Financial Record Normalization**
   - **Difficulty:** Hard
   - **Issues:** Malformed column names (extra spaces), redundant data, and extreme outliers.
   - **Goal:** Fix column naming, drop redundant features, and filter outliers.

## Setup & Usage

### Local Setup
1. Install dependencies:
   ```bash
   pip install pandas numpy pydantic openai
   ```
2. Set your API credentials:
   ```bash
   export HF_TOKEN="your_huggingface_token"
   export MODEL_NAME="meta-llama/Llama-3-70B-Instruct"
   ```

### Running the Baseline
Execute the inference script to see how a model performs on these tasks:
```bash
python inference.py
```

## Baseline Performance Scores
Based on preliminary runs using Llama-3-70B:
- **Easy:** 1.0 (Fixed price formatting and dropped nulls efficiently)
- **Medium:** 0.9 (Standardized dates, though some edge cases in duplicates may vary)
- **Hard:** 0.8 (Renamed columns correctly, but large datasets may require more steps)

### Docker
Build and run the environment locally:
```bash
docker build -t data-wrangler-env .
docker run -e HF_TOKEN=$HF_TOKEN data-wrangler-env
```

## Reward Function
The environment provides a **meaningful reward signal**:
- **Diagnostic/Transformation Reward:** +0.05 for every successful action that doesn't crash.
- **Error Penalty:** -0.1 for invalid actions (e.g., trying to clean a non-existent column).
- **Final Grading:** Upon calling `submit()`, the agent receives a score from **0.0 to 1.0** based on the cell-level similarity between its final DataFrame and a hidden "Golden Truth" dataset.

## OpenEnv Compliance
This environment fully supports the OpenEnv interface:
- Typed Pydantic models.
- `reset()`, `step()`, `state()` methods.
- `openenv.yaml` metadata.
- Validated via `openenv validate`.
