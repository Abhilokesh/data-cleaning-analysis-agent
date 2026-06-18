import re
import pandas as pd
import numpy as np
import json
from typing import Optional, Tuple


def pre_analyze(df: pd.DataFrame) -> str:
    """Programmatically detect data quality issues and return a plain-English issue list.

    This runs before the agent so the instruction is specific, not vague.
    """
    issues = []

    # 1. Always include all column names — LLM must judge semantic quality
    #    (abbreviations, misspellings, and special chars all need LLM reasoning)
    issues.append(
        f"Review every column name and rename any that are abbreviated, misspelled, "
        f"or not fully descriptive. Current column names: {list(df.columns)}"
    )

    # 2. String columns whose values look like datetimes
    dt_cols = []
    for col in df.select_dtypes(include='object').columns:
        sample = df[col].dropna().head(30).astype(str)
        if sample.str.match(r'^\d{4}-\d{2}-\d{2}').mean() > 0.7:
            dt_cols.append(col)
    if dt_cols:
        issues.append(
            f"Columns storing datetime values as strings — convert to datetime dtype: {dt_cols}"
        )

    # 3. Duplicate rows
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        issues.append(f"{dup_count} duplicate rows — call remove_duplicates")

    # 4. Missing values
    null_info = {
        col: int(df[col].isnull().sum())
        for col in df.columns if df[col].isnull().sum() > 0
    }
    if null_info:
        issues.append(f"Columns with missing values: {null_info}")

    # 5. Case inconsistencies in string columns
    case_cols = []
    for col in df.select_dtypes(include='object').columns:
        non_null = df[col].dropna()
        if len(non_null) > 0 and non_null.str.lower().nunique() < non_null.nunique():
            case_cols.append(col)
    if case_cols:
        issues.append(f"Case inconsistencies (same value in different cases) in: {case_cols}")

    # 6. Numeric outliers
    outlier_cols = []
    for col in df.select_dtypes(include='number').columns:
        if pd.api.types.is_bool_dtype(df[col]):
            continue
        s = df[col].dropna().astype(float)
        if len(s) == 0:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        count = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        if count > 0:
            outlier_cols.append(f"{col} ({count} outliers)")
    if outlier_cols:
        issues.append(f"Numeric outliers detected in: {outlier_cols}")

    if not issues:
        return "No obvious issues detected by automated analysis."

    return "\n".join(f"- {issue}" for issue in issues)


def compute_statistics(df: pd.DataFrame) -> str:
    """Compute real statistics on the full dataframe — nulls, distributions, outliers."""
    stats = {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": {}
    }

    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        col_stats = {
            "dtype": str(df[col].dtype),
            "null_count": null_count,
            "null_pct": round(null_count / len(df) * 100, 2) if len(df) > 0 else 0,
            "unique_count": int(df[col].nunique(dropna=True)),
        }

        if pd.api.types.is_bool_dtype(df[col]):
            # treat boolean columns as categorical — numpy can't do arithmetic on bool arrays
            value_counts = df[col].value_counts(dropna=False).head(15)
            col_stats["value_counts"] = {str(k): int(v) for k, v in value_counts.items()}
            col_stats["sample_unique_values"] = [str(v) for v in df[col].dropna().unique()[:10]]
        elif pd.api.types.is_numeric_dtype(df[col]):
            # cast to float64 so all numpy arithmetic (std, mean, subtraction) is safe
            non_null = df[col].dropna().astype(float)
            if len(non_null) > 0:
                q1 = float(non_null.quantile(0.25))
                q3 = float(non_null.quantile(0.75))
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_count = int(((non_null < lower) | (non_null > upper)).sum())
                col_stats.update({
                    "min": round(float(non_null.min()), 4),
                    "max": round(float(non_null.max()), 4),
                    "mean": round(float(non_null.mean()), 4),
                    "median": round(float(non_null.median()), 4),
                    "std": round(float(non_null.std()), 4),
                    "outlier_count": outlier_count,
                    "iqr_bounds": {"lower": round(lower, 4), "upper": round(upper, 4)},
                })
            else:
                col_stats["note"] = "all values are null"
        else:
            value_counts = df[col].value_counts(dropna=False).head(15)
            col_stats["value_counts"] = {str(k): int(v) for k, v in value_counts.items()}
            col_stats["sample_unique_values"] = [
                str(v) for v in df[col].dropna().unique()[:10]
            ]

        stats["columns"][col] = col_stats

    return json.dumps(stats, indent=2)


def rename_columns(df: pd.DataFrame, mapping: dict) -> Tuple[pd.DataFrame, str]:
    invalid = [k for k in mapping if k not in df.columns]
    if invalid:
        return df, f"Error: columns not found: {invalid}. Available: {list(df.columns)}"
    df = df.rename(columns=mapping)
    return df, f"Renamed {len(mapping)} columns: {mapping}"


def fill_missing(
    df: pd.DataFrame,
    column: str,
    strategy: str,
    fill_value: Optional[str] = None
) -> Tuple[pd.DataFrame, str]:
    if column not in df.columns:
        return df, f"Error: column '{column}' not found. Available: {list(df.columns)}"

    null_count = int(df[column].isnull().sum())
    if null_count == 0:
        return df, f"No missing values in '{column}' — skipped."

    if strategy == "mean":
        val = float(df[column].astype(float).mean())
        if pd.api.types.is_integer_dtype(df[column]):
            val = int(round(val))
        df[column] = df[column].fillna(val)
        return df, f"Filled {null_count} nulls in '{column}' with mean={val}"
    elif strategy == "median":
        val = float(df[column].astype(float).median())
        if pd.api.types.is_integer_dtype(df[column]):
            val = int(round(val))
        df[column] = df[column].fillna(val)
        return df, f"Filled {null_count} nulls in '{column}' with median={val}"
    elif strategy == "mode":
        val = df[column].mode()
        if len(val) == 0:
            return df, f"Error: cannot compute mode for '{column}' (all nulls?)"
        df[column] = df[column].fillna(val[0])
        return df, f"Filled {null_count} nulls in '{column}' with mode='{val[0]}'"
    elif strategy == "constant":
        if fill_value is None:
            return df, "Error: strategy='constant' requires a fill_value"
        # Cast constant to match column dtype so fillna doesn't reject it
        try:
            if pd.api.types.is_integer_dtype(df[column]):
                fill_value = int(float(fill_value))
            elif pd.api.types.is_float_dtype(df[column]):
                fill_value = float(fill_value)
        except (ValueError, TypeError):
            pass
        df[column] = df[column].fillna(fill_value)
        return df, f"Filled {null_count} nulls in '{column}' with constant='{fill_value}'"
    elif strategy == "forward_fill":
        df[column] = df[column].ffill()
        return df, f"Forward-filled {null_count} nulls in '{column}'"
    else:
        return df, f"Error: unknown strategy '{strategy}'. Choose from: mean, median, mode, constant, forward_fill"


def standardize_values(
    df: pd.DataFrame,
    column: str,
    mapping: dict
) -> Tuple[pd.DataFrame, str]:
    if column not in df.columns:
        return df, f"Error: column '{column}' not found"
    before = df[column].copy()
    df[column] = df[column].apply(
        lambda x: mapping.get(str(x), x) if pd.notna(x) else x
    )
    changed = int((before.astype(str) != df[column].astype(str)).sum())
    return df, f"Standardized {changed} values in '{column}' using mapping {mapping}"


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    return df, f"Removed {removed} duplicate rows. Remaining: {len(df)}"


def convert_dtype(df: pd.DataFrame, column: str, dtype: str) -> Tuple[pd.DataFrame, str]:
    if column not in df.columns:
        return df, f"Error: column '{column}' not found"
    try:
        if dtype in ("datetime", "date"):
            df[column] = pd.to_datetime(df[column], errors="coerce")
        elif dtype == "int":
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
        elif dtype == "float":
            df[column] = pd.to_numeric(df[column], errors="coerce")
        elif dtype == "str":
            df[column] = df[column].astype(str).replace("nan", np.nan)
        else:
            return df, f"Error: unknown dtype '{dtype}'. Choose: datetime, int, float, str"
        return df, f"Converted '{column}' to {dtype}"
    except Exception as e:
        return df, f"Error converting '{column}': {e}"


def standardize_text(df: pd.DataFrame, column: str, case: str) -> Tuple[pd.DataFrame, str]:
    if column not in df.columns:
        return df, f"Error: column '{column}' not found"
    if df[column].dtype != object:
        return df, f"'{column}' is not a string column — skipped"
    if case == "upper":
        df[column] = df[column].str.upper()
    elif case == "lower":
        df[column] = df[column].str.lower()
    elif case == "title":
        df[column] = df[column].str.title()
    elif case == "strip":
        df[column] = df[column].str.strip()
    else:
        return df, f"Error: unknown case '{case}'. Choose: upper, lower, title, strip"
    return df, f"Applied '{case}' to column '{column}'"


def handle_outliers(
    df: pd.DataFrame,
    column: str,
    method: str,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
) -> Tuple[pd.DataFrame, str]:
    if column not in df.columns:
        return df, f"Error: column '{column}' not found"
    if pd.api.types.is_bool_dtype(df[column]) or not pd.api.types.is_numeric_dtype(df[column]):
        return df, f"Error: '{column}' is boolean or non-numeric — outlier handling not applicable"

    non_null = df[column].dropna().astype(float)
    if lower is None:
        q1 = float(non_null.quantile(0.25))
        iqr = float(non_null.quantile(0.75)) - q1
        lower = q1 - 1.5 * iqr
    if upper is None:
        q3 = float(non_null.quantile(0.75))
        iqr = q3 - float(non_null.quantile(0.25))
        upper = q3 + 1.5 * iqr

    outlier_mask = (df[column] < lower) | (df[column] > upper)
    count = int(outlier_mask.sum())

    if method == "clip":
        df[column] = df[column].clip(lower=lower, upper=upper)
        return df, f"Clipped {count} outliers in '{column}' to [{round(lower,4)}, {round(upper,4)}]"
    elif method == "fill_median":
        median = float(non_null.median())
        if pd.api.types.is_integer_dtype(df[column]):
            median = int(round(median))
        df.loc[outlier_mask, column] = median
        return df, f"Replaced {count} outliers in '{column}' with median={median}"
    else:
        return df, f"Error: unknown method '{method}'. Choose: clip, fill_median"


def strip_whitespace(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, str]:
    if column not in df.columns:
        return df, f"Error: column '{column}' not found"
    if df[column].dtype != object:
        return df, f"'{column}' is not a string column — skipped"
    df[column] = df[column].str.strip()
    return df, f"Stripped whitespace from '{column}'"
