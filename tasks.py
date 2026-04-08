import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

def create_easy_task() -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Task: Clean a simple sales dataset.
    Issues: 'Price' column has '$' and commas, and contains nulls.
    Goal: Remove nulls from 'Price' and convert it to float.
    """
    messy_df = pd.DataFrame({
        "ID": [1, 2, 3, 4, 5],
        "Product": ["A", "B", "C", "D", "E"],
        "Price": ["$1,200", None, "$950", "$3,400.50", "invalid"]
    })
    
    # Golden truth: Drop rows with invalid or null prices, cast to float
    golden_df = pd.DataFrame({
        "ID": [1, 3, 4],
        "Product": ["A", "C", "D"],
        "Price": [1200.0, 950.0, 3400.50]
    }).reset_index(drop=True)
    
    goal = "Drop rows where 'Price' is missing or invalid, and convert 'Price' to float numbers."
    return messy_df, golden_df, goal

def create_medium_task() -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Task: Manage customer data.
    Issues: Duplicate rows, inconsistent 'JoinDate' formats, missing 'Email'.
    Goal: Remove duplicates, standardize 'JoinDate' to YYYY-MM-DD, fill missing 'Email' with 'N/A'.
    """
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
    
    goal = "Remove duplicate users, standardize 'JoinDate' to datetime objects, and fill missing 'Email' with 'N/A'."
    return messy_df, golden_df, goal

def create_hard_task() -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Task: Financial records normalization.
    Issues: Extra spaces in column names, redundant 'Category' column with inconsistent naming, outliers in 'Amount'.
    Goal: Rename '  Amount ' to 'Amount', drop 'Category' column, remove rows where 'Amount' > 10000.
    """
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
    
    goal = "Rename column '  Amount ' to 'Amount', drop the 'Category' column, and remove rows where 'Amount' is greater than 10,000."
    return messy_df, golden_df, goal

TASKS = {
    "easy": create_easy_task,
    "medium": create_medium_task,
    "hard": create_hard_task
}
