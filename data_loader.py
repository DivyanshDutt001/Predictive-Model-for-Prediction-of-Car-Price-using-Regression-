
import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """Read the CSV file and return a DataFrame."""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded  :  {df.shape[0]} rows  x  {df.shape[1]} columns")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Print column types, missing values, and the first few rows."""

    print("\n--- Column Data Types ---")
    print(df.dtypes.to_string())

    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() else "  No missing values found.")

    print("\n--- Numeric Summary ---")
    print(df.describe().round(2).to_string())

    print("\n--- First 5 Rows ---")
    print(df.head().to_string())
