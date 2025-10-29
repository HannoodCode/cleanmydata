import argparse
import os
from src.clean import clean_data
from src.utils import load_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CleanMyData - Clean a messy dataset")
    parser.add_argument("path", help="Path to dataset (.csv or .xls/.xlsx)")
    parser.add_argument("--output", default=None, help="Output file name (default: original_cleaned.csv)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed cleaning logs")
    args = parser.parse_args()

    df = load_data(args.path)

    if args.verbose:
        print("\n--- Original Data Preview ---")
        print(df.head())
        print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    cleaned_df = clean_data(df, verbose=args.verbose)

    if cleaned_df.empty:
        print("\nNo data cleaned — dataset is empty or invalid.")
    else:
        if args.output:
            output_path = os.path.join("data", args.output)
        else:
            filename = os.path.basename(args.path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join("data", f"{name}_cleaned{ext}")

        cleaned_df.to_csv(output_path, index=False)
        print(f"\nCleaned data saved as '{output_path}'")