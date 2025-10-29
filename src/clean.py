import pandas as pd
import numpy as np
import unicodedata
import shutil
import time
import re

def clean_data(
    df,
    *,
    outliers='cap',
    normalize_cols=True,
    clean_text=True,
    categorical_mapping=None,
    auto_outlier_detect=True,
    verbose=True
):
    """
    Master cleaning pipeline: sequentially applies cleaning operations to the input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataset.
    outliers : {'cap', 'remove', None}, default='cap'
        Outlier handling strategy.
    normalize_cols : bool, default=True
        Normalize column names (strip, lowercase, etc.).
    clean_text : bool, default=True
        Clean and standardize text fields.
    categorical_mapping : dict, optional
        Mapping dictionary for category normalization.
    auto_outlier_detect : bool, default=True
        Automatically decide IQR vs Z-score for outlier handling.
    verbose : bool, default=True
        Print cleaning progress and summary information.
    """

    start = time.time()

    if df is None or df.empty:
        if verbose:
            print("⚠️ No data provided or DataFrame is empty.")
        return df

    if verbose:
        print("\n--- Cleaning Data ---")

    # ---------- 1️. Remove duplicates ----------
    df = remove_duplicates(df, verbose=verbose)

    # ---------- 2️. Normalize column names ----------
    if normalize_cols:
        df = normalize_column_names(df, verbose=verbose)

    # ---------- 3️. Clean text & categorical values ----------
    if clean_text:
        df = clean_text_columns(
            df,
            lowercase=True,
            verbose=verbose,
            categorical_mapping=categorical_mapping
        )

    # ---------- 4. Standardize data formats ----------
    df = standardize_formats(df, verbose=verbose)

    # ---------- 5. Handle outliers ----------
    if outliers:
        df = handle_outliers(
            df,
            method=outliers,
            auto_detect=auto_outlier_detect,
            verbose=verbose
        )

    # ---------- 6. Fill missing values ----------
    df = fill_missing_values(df, verbose=verbose)

    # ---------- Summary ----------
    if verbose:
        print(f"\nRows: {df.shape[0]}, Columns: {df.shape[1]}")
        print(f"--- Data Cleaning Complete in {time.time() - start:.2f}s ---\n")

    return df

def remove_duplicates(
    df,
    subset=None,
    keep='first',
    verbose=True,
    normalize_text=False,
    return_report=False
):
    if df.empty:
        if verbose:
            print("  • No duplicates removed (empty dataset).")
        return (df, pd.DataFrame()) if return_report else df

    if subset:
        missing_cols = [c for c in subset if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Subset columns not found: {missing_cols}")

    if normalize_text:
        text_cols = df.select_dtypes(include='object').columns
        df[text_cols] = df[text_cols].apply(lambda col: col.str.strip().str.lower())

    initial_rows = len(df)
    dup_mask = df.duplicated(subset=subset, keep=keep)
    removed_rows = df[dup_mask]
    df_cleaned = df[~dup_mask]
    removed = len(removed_rows)

    if verbose:
        pct = (removed / initial_rows) * 100 if initial_rows else 0
        subset_info = f" (subset={subset})" if subset else ""

        if removed > 0:
            print(f"  • Duplicate rows removed: {removed} ({pct:.1f}%){subset_info}")
            print(f"    Example of removed rows (showing {min(3, removed)} of {removed}):")
            print(removed_rows.head(3).to_string(index=False))
        else:
            print("  • No duplicate rows found.")

    if return_report:
        return df_cleaned, removed_rows
    return df_cleaned

def normalize_column_names(df, verbose=True):
    old_cols = df.columns.tolist()

    # ---------- Normalize ----------
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r'[^0-9a-zA-Z_]+', '_', regex=True)
        .str.replace(r'_+', '_', regex=True)
        .str.strip('_')
    )

    # ---------- Deduplicate safely ----------
    seen = {}
    new_cols = []
    for col in df.columns:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_col = f"{col}_{seen[col]}"
            while new_col in seen:  # ensure no collision
                seen[col] += 1
                new_col = f"{col}_{seen[col]}"
            seen[new_col] = 0
            new_cols.append(new_col)
    df.columns = new_cols

    # ---------- Reporting ----------
    if verbose:
        print("  • Column names standardized:")
        renamed_pairs = [(o, n) for o, n in zip(old_cols, df.columns) if o != n]

        if renamed_pairs:
            term_width = shutil.get_terminal_size((80, 20)).columns
            max_len = min(max(len(o) for o, _ in renamed_pairs) + 2, term_width // 2)

            for o, n in renamed_pairs:
                print(f"    - {o:<{max_len}} → {n}")
            print(f"    ({len(renamed_pairs)} columns renamed)")
        else:
            print("    No renaming needed.")

    return df

def clean_text_columns(df, lowercase=True, verbose=True, categorical_mapping=None):
    def normalize_unicode(s):
        if isinstance(s, str):
            return unicodedata.normalize("NFKC", s)
        return s

    for col in df.select_dtypes(include=["object", "string"]):
        df[col] = (
            df[col]
            .astype("string")
            .replace(['nan', 'none', 'null'], pd.NA, regex=False)
            .str.strip()
            .replace(r'\s+', ' ', regex=True)
        )
        df[col] = df[col].map(normalize_unicode)
        if lowercase:
            df[col] = df[col].str.lower()

    if categorical_mapping:
        df = normalize_categorical_text(df, mapping=categorical_mapping, verbose=verbose)

    if verbose:
        text_cols = df.select_dtypes(include="object").columns
        print(f"  • Text columns cleaned ({len(text_cols)} columns)")
        print("    - Stripped whitespace, standardized spacing, and normalized casing.")

    return df

def normalize_categorical_text(df, mapping=None, verbose=True):
    """
      • mapping format: {column_name: {old_value: new_value}}
      • Safely handles unmapped values
      • Reports summary of mapped/unmapped categories
    """
    if mapping is None:
        if verbose:
            print("  • No categorical mappings provided.")
        return df

    for col, col_map in mapping.items():
        if col not in df.columns:
            if verbose:
                print(f"  • Skipped '{col}' (column not found).")
            continue

        # Preserve categorical dtype if applicable
        if pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].cat.add_categories(list(col_map.values())).replace(col_map, regex=False)
        else:
            df[col] = df[col].replace(col_map, regex=False)

        if verbose:
            unique_values = set(df[col].dropna().unique())
            mapped_keys = set(col_map.keys())
            unmapped = unique_values - mapped_keys
            print(f"  • Normalized '{col}' ({len(mapped_keys)} mapped)")
            if len(col_map) <= 5:
                for k, v in col_map.items():
                    print(f"    - {k} → {v}")
            if unmapped:
                preview = ', '.join(map(str, list(unmapped)[:3]))
                more = '' if len(unmapped) <= 3 else '...'
                print(f"    ⚠️ Unmapped values remain: {len(unmapped)} unique ({preview}{more})")


    return df

# ---------- COMPILED CONSTANTS ----------
CURRENCY_PATTERN = re.compile(
    r'(\b(?:USD|EUR|JPY|GBP|INR|AUD|CAD|CHF|CNY|HKD|SGD|MXN|NZD|SEK|NOK|DKK|KRW|RUB|BRL|ZAR|THB|TRY|ARS|EGP|PLN|BGN|HUF|COP|ILS|SAR|BHD|KWD|AED|DZD|NGN|PHP|PKR|BDT|VND|IDR|MYR|CLP|CZK)\b|'
    r'\$|€|¥|£|₹|A\$|C\$|NZ\$|S\$|kr|₩|₽|R\$|R|฿|₺|zł|лв|Ft|₪|﷼|د.إ|₦|₱|₨|₫|Rp)'
)

DATETIME_KEYWORDS = ['date', 'time', 'timestamp', 'created', 'modified', 'updated', 'dt']
NUMERIC_KEYWORDS = ['price', 'amount', 'total', 'cost', 'score', 'rate', 'balance', 'qty', 'quantity']
# ---------- ------------------ ----------

def standardize_formats(df, verbose=True):
    converted_dt = []
    converted_num = []

    for col in df.columns:
        if df[col].dtype not in ['object', 'string']:
            continue

        col_lower = col.lower()
        series = df[col]

        # ---------- DATETIME DETECTION ----------
        if any(k in col_lower for k in DATETIME_KEYWORDS):
            temp = pd.to_datetime(series, errors='coerce', format='mixed')
            success_ratio = temp.notna().mean()
            if success_ratio > 0.8:
                df[col] = temp
                converted_dt.append(f"{col} → datetime ({success_ratio:.0%} successfully parsed, keyword match)")
                continue

        # ---------- DATETIME FALLBACK ----------
        temp = pd.to_datetime(series, errors='coerce', format='mixed')
        if temp.notna().mean() > 0.9:
            df[col] = temp
            converted_dt.append(f"{col} → datetime (fallback inference)")
            continue

        # ---------- NUMERIC DETECTION ----------
        if any(k in col_lower for k in NUMERIC_KEYWORDS) or series.astype(str).str.contains(r'\d', na=False).any():
            cleaned = series.astype(str)
            cleaned = cleaned.str.replace(CURRENCY_PATTERN, '', regex=True)
            cleaned = cleaned.str.replace(r'\(([\d.,]+)\)', r'-\1', regex=True)  # accounting negatives
            cleaned = cleaned.str.replace(r'[,\s]', '', regex=True)
            temp = pd.to_numeric(cleaned, errors='coerce')
            success_ratio = temp.notna().mean()

            if success_ratio > 0.8:
                df[col] = temp
                converted_num.append(f"{col} → numeric ({success_ratio:.0%} valid)")

    # ---------- REPORT ----------
    if verbose:
        if converted_dt or converted_num:
            print("  • Formats standardized:")
            if converted_dt:
                print("      Datetime columns:")
                for item in converted_dt:
                    print(f"      - {item}")
            if converted_num:
                print("    Numeric columns:")
                for item in converted_num:
                    print(f"      - {item}")
        else:
            print("  • No columns converted.")

    return df

def fill_missing_values(df, verbose=True, numeric_strategy='auto', datetime_strategy='median'):
    """
    Fill missing values intelligently by column dtype:
      • Numeric     → mean/median (auto-detects skewness if numeric_strategy='auto')
      • Datetime    → median/mode/ffill (auto-detects pattern if datetime_strategy='auto')
      • Boolean     → mode (or False)
      • Category    → mode (adds 'Unknown' if missing from categories)
      • Object/Text → mode (or 'Unknown')

    numeric_strategy : {'auto', 'mean', 'median'}, optional
        Strategy for numeric columns:
          - 'auto'   → uses median if |skew| > 0.75, else mean

    datetime_strategy : {'median', 'auto'}, optional
        Strategy for datetime columns:
          - 'median' → always median timestamp (safe default baseline)
          - 'auto'   → chooses between ffill / mode / median depending on pattern
    """
    missing_filled = 0

    for col in df.columns:
        series = df[col]
        missing_count = series.isna().sum()
        if missing_count == 0:
            continue

        # Handle columns that are entirely NaN
        if series.dropna().empty:
            if np.issubdtype(series.dtype, np.number):
                fill_val, method_used = 0, "constant 0 (empty col)"
            elif np.issubdtype(series.dtype, np.datetime64):
                fill_val, method_used = pd.Timestamp("1970-01-01"), "default date (empty col)"
            elif series.dtype == 'bool':
                fill_val, method_used = False, "False (empty col)"
            else:
                fill_val, method_used = "Unknown", "'Unknown' (empty col)"
            df[col] = series.fillna(fill_val)
            missing_filled += missing_count
            if verbose:
                print(f"  • {col:<25} → {missing_count} missing filled [{method_used}]")
            continue

        # ---------- NUMERIC ----------
        if np.issubdtype(series.dtype, np.number):
            skew_val = series.skew(skipna=True)

            # Describe skewness qualitatively
            if abs(skew_val) < 0.5:
                skew_text = "(roughly symmetric)"
            elif abs(skew_val) < 1.5:
                skew_text = "(moderately skewed)"
            else:
                skew_text = "(highly skewed)"

            if numeric_strategy == 'median':
                fill_val = series.median()
                method_used = "median (forced)"
            elif numeric_strategy == 'mean':
                fill_val = series.mean()
                method_used = "mean (forced)"
            else:  # auto
                if abs(skew_val) > 0.75:
                    fill_val = series.median()
                    method_used = f"median (auto, skew={skew_val:.2f} {skew_text})"
                else:
                    fill_val = series.mean()
                    method_used = f"mean (auto, skew={skew_val:.2f} {skew_text})"
            df[col] = series.fillna(fill_val)

        # ---------- DATETIME ----------
        elif np.issubdtype(series.dtype, np.datetime64):
            if datetime_strategy == 'auto':
                valid_series = series.dropna().sort_values()
                if len(valid_series) > 2:
                    time_diffs = valid_series.diff().dropna()
                    regularity = (
                        (time_diffs.std() / time_diffs.mean()).total_seconds()
                        if isinstance(time_diffs.mean(), pd.Timedelta)
                        else np.inf
                    )
                    dup_ratio = 1 - valid_series.nunique() / len(valid_series)

                    if regularity < 0.2:
                        df[col] = series.fillna(method='ffill')
                        method_used = "ffill (auto: regular intervals)"
                    elif dup_ratio > 0.3:
                        mode_val = series.mode()
                        fill_val = mode_val[0] if not mode_val.empty else series.median()
                        df[col] = series.fillna(fill_val)
                        method_used = "mode (auto: repetitive dates)"
                    else:
                        fill_val = series.median()
                        df[col] = series.fillna(fill_val)
                        method_used = "median (auto: irregular)"
                else:
                    fill_val = series.median()
                    df[col] = series.fillna(fill_val)
                    method_used = "median (fallback)"
            else:
                fill_val = series.median()
                df[col] = series.fillna(fill_val)
                method_used = "median (default)"

        # ---------- BOOLEAN ----------
        elif series.dtype == 'bool':
            mode_val = series.mode()
            fill_val = mode_val[0] if not mode_val.empty else False
            df[col] = series.fillna(fill_val)
            method_used = "mode (bool)"

        # ---------- CATEGORY ----------
        elif pd.api.types.is_categorical_dtype(series):
            mode_val = series.mode()
            fill_val = mode_val[0] if not mode_val.empty else "Unknown"
            if fill_val not in series.cat.categories:
                series = series.cat.add_categories([fill_val])
            df[col] = series.fillna(fill_val)
            method_used = "mode/category"

        # ---------- OBJECT / STRING ----------
        else:
            mode_val = series.mode()
            fill_val = mode_val[0] if not mode_val.empty else "Unknown"
            df[col] = series.fillna(fill_val)
            method_used = "mode/string"

        missing_filled += missing_count

        if verbose:
            print(f"  • {col:<25} → {missing_count} missing filled [{method_used}]")

    if verbose and missing_filled > 0:
        print(f"  • Total missing values filled: {missing_filled}")

    return df

def handle_outliers(df, method='cap', auto_detect=True, verbose=True):
    """
    Handles outliers in numeric columns using IQR or Z-score detection.
    Can auto-switch based on column skewness.

    Args:
        method (str): 'cap' (default) to cap outliers, or 'remove' to drop them
        auto_detect (bool): If True, automatically decides between IQR and Z-score per column

    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers_removed = 0
    outliers_capped = {}

    for col in numeric_cols:
        series = df[col].dropna()

        # Skip empty or constant columns
        if series.nunique() < 2:
            continue

        # ---------- AUTO-DETECTION ----------
        skew = series.skew()
        method_used = "IQR" if (auto_detect and abs(skew) > 0.5) else "Z-score" if auto_detect else "IQR"

        # ---------- OUTLIER BOUNDS ----------
        if method_used == "IQR":
            Q1, Q3 = series.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            if IQR == 0:
                continue
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        else:  # Z-score
            mean, std = series.mean(), series.std()
            if std == 0:
                continue
            lower, upper = mean - 3 * std, mean + 3 * std

        # ---------- OUTLIER DETECTION ----------
        mask = (df[col] < lower) | (df[col] > upper)
        outlier_count = mask.sum()
        if outlier_count == 0:
            continue

        low_outliers = (df[col] < lower).sum()
        high_outliers = (df[col] > upper).sum()

        # ---------- OUTLIER HANDLING ----------
        if method == 'remove':
            before = len(df)
            df = df.loc[~mask]
            removed_count = before - len(df)
            outliers_removed += removed_count
        elif method == 'cap':
            df.loc[df[col] < lower, col] = lower
            df.loc[df[col] > upper, col] = upper
            outliers_capped[col] = outlier_count

        # ---------- SKEW INTERPRETATION ----------
        skew_desc = (
            "roughly symmetric" if abs(skew) < 0.3 else
            "moderately skewed" if abs(skew) < 1 else
            "highly skewed"
        )

        # ---------- LOGGING ----------
        if verbose:
            print(f"  • {col:<25} → {outlier_count} outliers handled [{method_used}, skew={skew:.2f} ({skew_desc})]")
            print(f"    - low={low_outliers}, high={high_outliers}")

    # ---------- SUMMARY ----------
    if verbose:
        if method == 'remove' and outliers_removed > 0:
            print(f"  • Total outliers removed: {outliers_removed}")
        elif method == 'cap' and outliers_capped:
            total_capped = sum(outliers_capped.values())
            print(f"  • Outliers capped across {len(outliers_capped)} columns ({total_capped} total).")
        else:
            print("  • No significant outliers detected.")

    return df
