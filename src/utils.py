import pandas as pd

def load_data(filepath):
    try:
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        elif filepath.endswith(('.xls', '.xlsx')):
            return pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format. Current supported formats: .csv, .xls, .xlsx")
        
    except FileNotFoundError:
        print(f"Error: The file at {filepath} was not found.")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        return pd.DataFrame()
    except pd.errors.ParserError:
        print("Error: There was a parsing error while reading the file.")
        return pd.DataFrame()