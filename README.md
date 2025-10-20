Simple CLI tool to clean datasets.

- Removes duplicate rows
- Fills missing values (mean for numeric columns, mode for categorical columns)
- Supports CSV, XLS, and XLSX file formats
- Saves cleaned data to the `data/` directory

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

A sample dataset (`cafe_sales.csv`) is included in the `data/` folder for testing purposes.[Source](https://www.kaggle.com/datasets/ahmedmohamed2003/cafe-sales-dirty-data-for-cleaning-training/data)

## Usage
```bash
python main.py data/cafe_sales.csv  # Saves as cafe_sales_cleaned.csv
python main.py data/cafe_sales.csv --output cleaned_data.csv
python main.py --help
```