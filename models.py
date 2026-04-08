from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal, Union

# --- Observation Model ---

class ColumnInfo(BaseModel):
    name: str
    dtype: str
    null_count: int
    sample_values: List[Any]

class DataWranglerObservation(BaseModel):
    task_id: str = Field(description="The unique identifier for the current task.")
    task_goal: str = Field(description="The objective for the current task.")
    total_rows: int
    columns: List[ColumnInfo]
    head: List[Dict[str, Any]] = Field(description="First 5 rows of the current dataset.")
    last_action_status: str = Field(default="initial", description="Status of the last action (success/error).")
    last_error: Optional[str] = Field(default=None, description="Error message if the previous action failed.")

# --- Action Models ---

class DropNullsAction(BaseModel):
    action_type: Literal["drop_nulls"] = "drop_nulls"
    column_name: str

class FillNullsAction(BaseModel):
    action_type: Literal["fill_nulls"] = "fill_nulls"
    column_name: str
    fill_value: Any

class CastTypeAction(BaseModel):
    action_type: Literal["cast_type"] = "cast_type"
    column_name: str
    new_type: Literal["int", "float", "str", "datetime"]

class DropColumnAction(BaseModel):
    action_type: Literal["drop_column"] = "drop_column"
    column_name: str

class RenameColumnAction(BaseModel):
    action_type: Literal["rename_column"] = "rename_column"
    old_name: str
    new_name: str

class RegexReplaceAction(BaseModel):
    action_type: Literal["regex_replace"] = "regex_replace"
    column_name: str
    pattern: str
    replacement: str

class SubmitAction(BaseModel):
    action_type: Literal["submit"] = "submit"

class DataWranglerAction(BaseModel):
    command: Union[
        DropNullsAction, 
        FillNullsAction, 
        CastTypeAction, 
        DropColumnAction, 
        RenameColumnAction,
        RegexReplaceAction,
        SubmitAction
    ]

# --- Reward Model ---

class DataWranglerReward(BaseModel):
    reward: float = Field(ge=-1.0, le=1.0, description="The reward for the last step.")
    cumulative_reward: float = Field(description="Total reward earned in the episode.")
