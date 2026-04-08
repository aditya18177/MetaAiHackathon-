import pandas as pd
import numpy as np
from typing import Tuple


def create_easy_task() -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Easy: Sales data — strip currency symbols, drop nulls, cast Price to float."""
    messy_df = pd.DataFrame({
        "ID": [1, 2, 3, 4, 5],
        "Product": ["A", "B", "C", "D", "E"],
        "Price": ["$1,200", None, "$950", "$3,400.50", "invalid"]
    })
    golden_df = pd.DataFrame({
        "ID": [1, 3, 4],
        "Product": ["A", "C", "D"],
        "Price": [1200.0, 950.0, 3400.50]
    }).reset_index(drop=True)
    goal = "Drop rows where 'Price' is missing or invalid, and convert 'Price' to float."
    return messy_df, golden_df, goal


def create_medium_task() -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Medium: User profiles — deduplicate, standardize dates, fill missing emails."""
    messy_df = pd.DataFrame({
        "UserID": [101, 102, 101, 103, 104],
        "Name": ["Alice", "Bob", "Alice", "Charlie", "David"],
        "JoinDate": ["2023-01-01", "01/02/2023", "2023-01-01", "2023-03-15", "15-04-2023"],
        "Email": ["alice@ex.com", None, "alice@ex.com", "charlie@ex.com", "david@ex.com"]
    })
    golden_df = pd.DataFrame({
        "UserID": [101, 102, 103, 104],
        "Name": ["Alice", "Bob", "Charlie", "David"],
        "JoinDate": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-03-15", "2023-04-15"]),
        "Email": ["alice@ex.com", "N/A", "charlie@ex.com", "david@ex.com"]
    })
    goal = "Remove duplicate users, standardize 'JoinDate' to datetime, fill missing 'Email' with 'N/A'."
    return messy_df, golden_df, goal


def create_hard_task() -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Hard: Financial records — fix column names, drop redundant col, remove outliers."""
    messy_df = pd.DataFrame({
        "ID": [1, 2, 3, 4],
        "  Amount ": [500, 15000, 250, 8000],
        "Category": ["Food", "Misc", "Rent", "Fees"],
        "Notes": ["Lunch", "Error", "Month 1", "Admin"]
    })
    golden_df = pd.DataFrame({
        "ID": [1, 3, 4],
        "Amount": [500.0, 250.0, 8000.0],
        "Notes": ["Lunch", "Month 1", "Admin"]
    }).reset_index(drop=True)
    goal = "Rename '  Amount ' to 'Amount', drop 'Category', remove rows where Amount > 10000."
    return messy_df, golden_df, goal


# ── Sample tasks (deterministic partial scores for grader validation) ──────────

def create_easy_sample_task() -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Easy sample: Inventory data — fill missing stock values with 0."""
    messy_df = pd.DataFrame({
        "ItemID": [1, 2, 3, 4],
        "Name": ["Widget", "Gadget", "Doohickey", "Thingamajig"],
        "Stock": [10, None, 5, None]
    })
    golden_df = pd.DataFrame({
        "ItemID": [1, 2, 3, 4],
        "Name": ["Widget", "Gadget", "Doohickey", "Thingamajig"],
        "Stock": [10.0, 0.0, 5.0, 0.0]
    })
    goal = "Fill missing values in 'Stock' column with 0."
    return messy_df, golden_df, goal


def create_medium_sample_task() -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Medium sample: Employee records — rename column, drop unused col, cast salary."""
    messy_df = pd.DataFrame({
        "emp_id": [1, 2, 3],
        "full name": ["Alice Smith", "Bob Jones", "Carol White"],
        "salary_str": ["50000", "62000", "71000"],
        "dept_code": ["HR", "ENG", "MKT"]
    })
    golden_df = pd.DataFrame({
        "emp_id": [1, 2, 3],
        "full name": ["Alice Smith", "Bob Jones", "Carol White"],
        "salary": [50000.0, 62000.0, 71000.0]
    })
    goal = "Rename 'salary_str' to 'salary', cast it to float, and drop the 'dept_code' column."
    return messy_df, golden_df, goal


def create_hard_sample_task() -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Hard sample: Sensor logs — drop nulls, remove outliers, cast timestamp."""
    messy_df = pd.DataFrame({
        "sensor_id": [1, 2, 3, 4, 5],
        "reading": [23.5, None, 9999.0, 22.1, 24.8],
        "recorded_at": ["2024-01-01 10:00", "2024-01-01 10:05",
                        "2024-01-01 10:10", "2024-01-01 10:15", "2024-01-01 10:20"]
    })
    golden_df = pd.DataFrame({
        "sensor_id": [1, 4, 5],
        "reading": [23.5, 22.1, 24.8],
        "recorded_at": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 10:15", "2024-01-01 10:20"])
    }).reset_index(drop=True)
    goal = "Drop rows with null 'reading', remove rows where 'reading' > 100, cast 'recorded_at' to datetime."
    return messy_df, golden_df, goal


TASKS = {
    "easy": create_easy_task,
    "medium": create_medium_task,
    "hard": create_hard_task,
    "easy_sample": create_easy_sample_task,
    "medium_sample": create_medium_sample_task,
    "hard_sample": create_hard_sample_task,
}
