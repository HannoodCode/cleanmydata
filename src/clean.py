import pandas as pd
import numpy as np
import re

def remove_duplicates(df, subset=None, keep='first', verbose=True):
    initial_rows = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep)
    removed = initial_rows - len(df)
    if verbose:
        print(f"  • Duplicate rows removed: {removed}")
    return df

def fill_missing_values(df, verbose=True):
    missing_filled = 0
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count == 0:
            continue
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        missing_filled += missing_count
        if verbose:
            print(f"  • {col}: {missing_count} missing values filled")
    return df

def normalize_column_names(df, verbose=True):
    old_cols = df.columns.tolist()

    # Clean and normalize names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace('[^a-z0-9_]+', '_', regex=True)
        .str.replace('_+', '_', regex=True)
        .str.strip('_')
    )

    # Deduplicate column names if needed
    seen = {}
    new_cols = []
    for col in df.columns:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
    df.columns = new_cols

    if verbose:
        print("  • Column names standardized")
        if old_cols != list(df.columns):
            for before, after in zip(old_cols, df.columns):
                if before != after:
                    print(f"    - {before} → {after}")

    return df

def normalize_categorical_text(df, mapping=None, verbose=True):
    """mapping: dict of {column_name: {old_value: new_value}}"""

    if mapping is None:
        if verbose:
            print("  • No categorical mappings provided.")
        return df
    
    for col, col_map in mapping.items():
        if col in df.columns:
            df[col] = df[col].replace(col_map)
            if verbose:
                print(f"  • Normalized {col} with mapping")
    return df

def clean_text_columns(df, lowercase=True, verbose=True, categorical_mapping=None):
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].astype("string")
        df[col] = df[col].replace(['nan', 'none', 'null'], pd.NA)
        df[col] = df[col].str.strip().replace(r'\s+', ' ', regex=True)
        if lowercase:
            df[col] = df[col].str.lower()

    if categorical_mapping:
        df = normalize_categorical_text(df, mapping=categorical_mapping, verbose=verbose)

    if verbose:
        print("  • Text columns cleaned")
    return df

def handle_outliers(df, method='cap', verbose=True):
    num_cols = df.select_dtypes(include=[np.number]).columns
    outliers_removed = 0
    outliers_capped = {}

    for col in num_cols:
        if df[col].dropna().nunique() < 2:
            # Skip constant or empty columns
            continue

        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

        if method == 'remove':
            before = len(df)
            df = df[(df[col] >= lower) & (df[col] <= upper)]
            removed_count = before - len(df)
            if removed_count > 0:
                outliers_removed += removed_count

        elif method == 'cap':
            before_outliers = df[(df[col] < lower) | (df[col] > upper)][col].count()
            if before_outliers > 0:
                df[col] = df[col].clip(lower, upper)
                outliers_capped[col] = before_outliers

    # ---------- REPORT ----------
    if verbose:
        if method == 'remove' and outliers_removed > 0:
            print(f"  • Outliers removed: {outliers_removed}")
        elif method == 'cap' and outliers_capped:
            print("  • Outliers capped:")
            for col, count in outliers_capped.items():
                print(f"    - {col}: {count} values capped")
        else:
            print("  • No outliers detected.")

    return df

# ---------- COMPILED CONSTANTS ----------
CURRENCY_PATTERN = re.compile(
    r'(\b(?:USD|EUR|JPY|GBP|INR|AUD|CAD|CHF|CNY|HKD|SGD|MXN|NZD|SEK|NOK|DKK|KRW|RUB|BRL|ZAR|THB|TRY|ARS|EGP|PLN|BGN|HUF|COP|ILS|SAR|BHD|KWD|AED|DZD|NGN|PHP|PKR|BDT|VND|IDR|MYR|CLP|CZK)\b|'
    r'\$|€|¥|£|₹|A\$|C\$|NZ\$|S\$|kr|₩|₽|R\$|R|฿|₺|zł|лв|Ft|₪|﷼|د.إ|₦|₱|₨|₫|Rp)'
)

DATETIME_KEYWORDS = ['date', 'time', 'timestamp', 'created', 'modified', 'updated', 'dt']
# ---------- ------------------ ----------

def standardize_formats(df, verbose=True):
    converted = []

    for col in df.columns:
        if df[col].dtype != 'object':
            continue

        col_lower = col.lower()
        series = df[col]

        # ---------- DATETIME DETECTION ----------
        if any(k in col_lower for k in DATETIME_KEYWORDS):
            try:
                temp = pd.to_datetime(series, errors='coerce')
                success_ratio = temp.notna().mean()
                if success_ratio > 0.8:
                    df[col] = temp
                    converted.append(f"{col} → datetime ({success_ratio:.0%} parsed, keyword match)")
                    continue
            except Exception:
                pass

        # ---------- DATETIME FALLBACK ----------
        try:
            temp = pd.to_datetime(series, errors='coerce')
            if temp.notna().mean() > 0.9:
                df[col] = temp
                converted.append(f"{col} → datetime (fallback inference)")
                continue
        except Exception:
            pass

        # ---------- NUMERIC DETECTION ----------
        if series.astype(str).str.contains(r'\d', na=False).any():
            cleaned = series.astype(str)
            cleaned = cleaned.str.replace(CURRENCY_PATTERN, '', regex=True)
            cleaned = cleaned.str.replace(r'[,\s]', '', regex=True)
            temp = pd.to_numeric(cleaned, errors='coerce')

            success_ratio = temp.notna().mean()
            if success_ratio > 0.8:
                df[col] = temp
                converted.append(f"{col} → numeric ({success_ratio:.0%} valid)")

    # ---------- REPORT ----------
    if verbose:
        if converted:
            print("  • Formats standardized:")
            for item in converted:
                print(f"    - {item}")
        else:
            print("  • No columns converted.")

    return df

def clean_data(df, outliers='cap', normalize_cols=True, clean_text=True, categorical_mapping=None, verbose=True):
    
    if verbose:
        print("\n--- Cleaning Data ---")

    # 1. Remove duplicates
    df = remove_duplicates(df, verbose=verbose)

    # 2. Normalize column names
    if normalize_cols:
        df = normalize_column_names(df, verbose=verbose)

    # 3. Clean text columns
    if clean_text:
        df = clean_text_columns(df, lowercase=True, verbose=verbose, categorical_mapping=categorical_mapping)

    # 4. Standardize numeric and datetime formats
    df = standardize_formats(df, verbose=verbose)

    # 5. Fill missing values
    df = fill_missing_values(df, verbose=verbose)

    # 6. Handle outliers
    if outliers:
        df = handle_outliers(df, method=outliers, verbose=verbose)

    if verbose:
        print(f"\nRows: {df.shape[0]}, Columns: {df.shape[1]}")
        print("--- Data Cleaning Complete ---\n")

    return df