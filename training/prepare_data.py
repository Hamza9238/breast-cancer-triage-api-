"""
Prepare the breast cancer dataset from sklearn and export it as CSV.

Validates that there are no null values and exactly 30 feature columns.
"""

import os
import sys
import pandas as pd
from sklearn.datasets import load_breast_cancer


def main():
    # Load the breast cancer dataset from sklearn
    data = load_breast_cancer()

    # Build a DataFrame with feature columns + target
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    # --- Validation ---
    # Check no null values
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        print(f"ERROR: Found {null_count} null values in the dataset.")
        sys.exit(1)

    # Check feature count is exactly 30 (excluding target)
    feature_count = len(df.columns) - 1  # subtract the target column
    if feature_count != 30:
        print(f"ERROR: Expected 30 features, found {feature_count}.")
        sys.exit(1)

    print(f"Dataset validation passed: {len(df)} samples, {feature_count} features, 0 nulls.")

    # Ensure output directory exists
    output_path = os.path.join("data", "raw", "breast_cancer.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Export to CSV
    df.to_csv(output_path, index=False)
    print(f"Dataset exported to {output_path}")


if __name__ == "__main__":
    main()
