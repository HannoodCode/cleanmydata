import pandas as pd

def clean_data(df):
    print("\n--- Cleaning Data ---")

    # Drop duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    print(f"Duplicate rows dropped: {duplicates_removed}\n")

    # Handle missing values
    print("Missing values filled:")
    missing_filled = 0
    for col in df.columns:
        missing_count = df[col].isna().sum()

        if missing_count == 0:
            continue
    
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    
        missing_filled += missing_count
        print(f"  • {col}: {missing_count}")

    print(f"\n{duplicates_removed} Duplicates removed.")
    print(f"{missing_filled} Missing values filled.")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("--- Data Cleaning Complete ---")

    return df