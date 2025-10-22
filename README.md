Simple CLI tool to clean datasets.

- Removes duplicate rows
- Normalizes column names (lowercase, underscores, removes special characters, deduplicates)
- Cleans text columns (trims whitespace, optional lowercasing, replaces `'nan'`, `'none'`, `'null'` with missing values)
- Optional categorical text normalization via mapping
- Converts numeric-like strings to proper numeric type (handles currencies, commas)
- Converts datetime-like columns to `datetime` type (keyword-based detection + fallback inference)
- Fills missing values:
  - Numeric columns → mean
  - Categorical/text columns → mode or `'Unknown'`
- Handles outliers:
  - `cap` (clip values to IQR bounds) or `remove` (drop outlier rows)
- Supports CSV, XLS, and XLSX file formats
- Saves cleaned data to the `data/` directory

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

A sample dataset (`cafe_sales.csv`) is included in the `data/` folder for testing purposes.
[Source (Kaggle)](https://www.kaggle.com/datasets/ahmedmohamed2003/cafe-sales-dirty-data-for-cleaning-training/data)

## Usage
```bash
# Clean and save with default naming
python main.py data/cafe_sales.csv  # Saves as cafe_sales_cleaned.csv

# Specify output file name
python main.py data/cafe_sales.csv --output cleaned_data.csv

python main.py --help
```