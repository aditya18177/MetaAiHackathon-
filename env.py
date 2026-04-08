import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from models import (
    DataWranglerObservation, 
    DataWranglerAction, 
    DataWranglerReward, 
    EnvState,
    ColumnInfo
)
from tasks import TASKS

class DataWranglerEnv:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.golden_df: Optional[pd.DataFrame] = None
        self.task_id: str = ""
        self.task_goal: str = ""
        self.steps_taken = 0
        self.max_steps = 10
        self.cumulative_reward = 0.0
        self.done = False
        self.last_error = None
        self.last_status = "initial"

    def reset(self, task_id: str = "easy") -> DataWranglerObservation:
        """Resets the environment to a specific task."""
        if task_id not in TASKS:
            raise ValueError(f"Task {task_id} not found.")
        
        self.task_id = task_id
        self.df, self.golden_df, self.task_goal = TASKS[task_id]()
        self.steps_taken = 0
        self.cumulative_reward = 0.0
        self.done = False
        self.last_error = None
        self.last_status = "initial"
        
        return self._get_observation()

    def step(self, action: DataWranglerAction) -> Tuple[DataWranglerObservation, float, bool, Dict[str, Any]]:
        """Executes one step in the environment."""
        if self.done:
            return self._get_observation(), 0.0, True, {"info": "Episode already finished"}

        self.steps_taken += 1
        cmd = action.command
        success = False
        error_msg = None

        try:
            if cmd.action_type == "drop_nulls":
                if cmd.column_name in self.df.columns:
                    self.df = self.df.dropna(subset=[cmd.column_name]).reset_index(drop=True)
                    success = True
                else:
                    error_msg = f"Column '{cmd.column_name}' not found."

            elif cmd.action_type == "fill_nulls":
                if cmd.column_name in self.df.columns:
                    self.df[cmd.column_name] = self.df[cmd.column_name].fillna(cmd.fill_value)
                    success = True
                else:
                    error_msg = f"Column '{cmd.column_name}' not found."

            elif cmd.action_type == "cast_type":
                if cmd.column_name in self.df.columns:
                    if cmd.new_type == "int":
                        self.df[cmd.column_name] = pd.to_numeric(self.df[cmd.column_name], errors='coerce').astype(pd.Int64Dtype())
                    elif cmd.new_type == "float":
                        self.df[cmd.column_name] = pd.to_numeric(self.df[cmd.column_name], errors='coerce').astype(float)
                    elif cmd.new_type == "str":
                        self.df[cmd.column_name] = self.df[cmd.column_name].astype(str)
                    elif cmd.new_type == "datetime":
                        self.df[cmd.column_name] = pd.to_datetime(self.df[cmd.column_name], errors='coerce')
                    success = True
                else:
                    error_msg = f"Column '{cmd.column_name}' not found."

            elif cmd.action_type == "drop_column":
                if cmd.column_name in self.df.columns:
                    self.df = self.df.drop(columns=[cmd.column_name])
                    success = True
                else:
                    error_msg = f"Column '{cmd.column_name}' not found."

            elif cmd.action_type == "rename_column":
                if cmd.old_name in self.df.columns:
                    self.df = self.df.rename(columns={cmd.old_name: cmd.new_name})
                    success = True
                else:
                    error_msg = f"Column '{cmd.old_name}' not found."

            elif cmd.action_type == "regex_replace":
                if cmd.column_name in self.df.columns:
                    self.df[cmd.column_name] = self.df[cmd.column_name].astype(str).str.replace(cmd.pattern, cmd.replacement, regex=True)
                    success = True
                else:
                    error_msg = f"Column '{cmd.column_name}' not found."

            elif cmd.action_type == "submit":
                self.done = True
                success = True

        except Exception as e:
            error_msg = str(e)

        if success:
            self.last_status = "success"
            self.last_error = None
        else:
            self.last_status = "error"
            self.last_error = error_msg

        # Calculate reward
        step_reward = self._calculate_reward(success)
        self.cumulative_reward += step_reward

        if self.steps_taken >= self.max_steps:
            self.done = True

        obs = self._get_observation()
        # Include a typed reward object in the info dict for observability via OpenEnv spec
        reward_report = DataWranglerReward(
            reward=step_reward,
            cumulative_reward=self.cumulative_reward
        )
        info = {
            "success": success, 
            "steps": self.steps_taken,
            "reward_report": reward_report.dict()
        }
        
        return obs, step_reward, self.done, info

    def state(self) -> EnvState:
        """Returns the current internal state as a typed Pydantic model."""
        return EnvState(
            task_id=self.task_id,
            steps_taken=self.steps_taken,
            max_steps=self.max_steps,
            done=self.done,
            cumulative_reward=round(self.cumulative_reward, 4),
            df_shape=list(self.df.shape) if self.df is not None else None
        )

    def _get_observation(self) -> DataWranglerObservation:
        """Constructs the observation model."""
        cols = []
        for col in self.df.columns:
            cols.append(ColumnInfo(
                name=col,
                dtype=str(self.df[col].dtype),
                null_count=int(self.df[col].isna().sum()),
                sample_values=self.df[col].head(3).tolist()
            ))

        return DataWranglerObservation(
            task_id=self.task_id,
            task_goal=self.task_goal,
            total_rows=len(self.df),
            columns=cols,
            head=self.df.head(5).to_dict(orient="records"),
            last_action_status=self.last_status,
            last_error=self.last_error
        )

    def _calculate_reward(self, last_action_success: bool) -> float:
        if not last_action_success:
            return -0.1  # penalty for invalid action

        if self.done:
            return self.get_score()  # strictly in (0.05, 0.95)

        return 0.05  # small reward for a valid step

    def get_score(self) -> float:
        """Programmatic grader: calculates the final score strictly in (0, 1) exclusive."""
        if self.df is None or self.golden_df is None:
            return 0.05  # never return exactly 0.0

        try:
            # 1. Column mismatch — partial credit
            if set(self.df.columns) != set(self.golden_df.columns):
                common = len(set(self.df.columns) & set(self.golden_df.columns))
                total = len(set(self.golden_df.columns))
                raw = 0.5 * (common / total) if total > 0 else 0.05
                return max(0.05, min(0.95, raw))

            # 2. Shape mismatch — partial credit
            if self.df.shape != self.golden_df.shape:
                return 0.4

            # 3. Cell-level similarity
            df1 = self.df.sort_index(axis=1).reset_index(drop=True)
            df2 = self.golden_df.sort_index(axis=1).reset_index(drop=True)

            # Convert both to string for safe comparison across mixed types
            matching_cells = (df1.astype(str).values == df2.astype(str).values).sum()
            total_cells = df1.size

            raw = matching_cells / total_cells if total_cells > 0 else 0.05

            # Clamp strictly to (0.05, 0.95) — never exactly 0.0 or 1.0
            return max(0.05, min(0.95, raw))
        except Exception:
            return 0.05
