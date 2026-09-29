"""Data layer: load each CSV, describe it, and precompute its statistics.

Pure pandas, no LLM calls. Nothing here is specific to one dataset: behaviour
is driven by column dtypes, so adding a dataset means adding a DATASET_CONFIG entry.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
MAX_CATEGORIES = 10  # skip group-bys on columns with many unique values (e.g. names)

# The human-written parts. Descriptions matter because column names alone
# don't say what a "Spending Score" is or that search values are relative interest.
DATASET_CONFIG = {
    "shopping_habits": {
        "file": "shopping_habits.csv",
        "description": (
            "Demographics of individual retail customers (age, gender, annual income) "
            "and a 1-100 score for how much each one shops / spends. One row per customer."
        ),
        "drop_columns": ["Customer ID"],  # PII: never sent to the LLM
        "date_columns": [],
        # Optional: turn a numeric column into groups people ask about.
        "buckets": {"Age": {"bins": [0, 34, 54, 120], "labels": ["under 35", "35-54", "55+"]}},
    },
    "weekly_searches": {
        "file": "weekly_searches_for_programming_languages.csv",
        "description": (
            "Weekly search-engine interest for three programming languages "
            "(Python, Java, C++). One row per week."
        ),
        "drop_columns": [],
        "date_columns": ["Week"],
        "buckets": {},
    },
}


def build_profile(name: str, description: str, df: pd.DataFrame) -> dict:
    """Describe a dataset compactly: this is what the router sees instead of the data."""
    columns = {}
    for col in df.columns:
        series = df[col]
        info = {"dtype": str(series.dtype)}
        if pd.api.types.is_datetime64_any_dtype(series):
            info["date_range"] = [str(series.min().date()), str(series.max().date())]
        elif pd.api.types.is_numeric_dtype(series):
            info["range"] = [float(series.min()), float(series.max())]
        else:
            info["categories"] = sorted(series.unique().tolist())
        columns[col] = info

    return {
        "name": name,
        "description": description,
        "row_count": len(df),
        "columns": columns,
        "sample_rows": df.head(3).astype(str).to_dict("records"),
    }


def compute_stats(df: pd.DataFrame, buckets: dict) -> str:
    """Precompute the numbers a question is likely to need, for any dataset."""
    numeric = df.select_dtypes("number").columns.tolist()
    dates = df.select_dtypes("datetime").columns.tolist()
    categorical = [col for col in df.columns if col not in numeric + dates]

    parts = [f"Overall (n={len(df)}):\n{df[numeric].describe().round(1).to_string()}"]

    # Group averages: by each categorical column, and by each configured bucket.
    groupings = {col: df[col] for col in categorical if df[col].nunique() <= MAX_CATEGORIES}
    for col, bucket in buckets.items():
        groupings[f"{col} bucket"] = pd.cut(df[col], bins=bucket["bins"], labels=bucket["labels"])
    for label, group_keys in groupings.items():
        grouped = df[numeric].groupby(group_keys, observed=True)
        table = grouped.mean().round(1)
        table.insert(0, "Count", grouped.size())
        parts.append(f"Averages by {label}:\n{table.to_string()}")

    if len(numeric) > 1:
        parts.append(f"Pearson correlations:\n{df[numeric].corr().round(2).to_string()}")

    # Time series: for each date column, where each numeric column peaked and how it moved.
    for date_col in dates:
        ordered = df.sort_values(date_col)
        lines = [f"Over time (by {date_col}):"]
        for col in numeric:
            highest = ordered.loc[ordered[col].idxmax()]
            lowest = ordered.loc[ordered[col].idxmin()]
            # Trend = first 4 rows vs last 4 rows. Cruder than a regression,
            # but easy to verify by hand.
            first, last = ordered[col].head(4).mean(), ordered[col].tail(4).mean()
            lines.append(
                f"{col}: highest {highest[date_col].date()} ({highest[col]}), "
                f"lowest {lowest[date_col].date()} ({lowest[col]}), "
                f"first-4 avg {first:.1f} vs last-4 avg {last:.1f} (change {last - first:+.1f})"
            )
        parts.append("\n".join(lines))

    return "\n\n".join(parts)


def load_datasets() -> dict:
    """Load and clean every CSV. Returns {name: {"df", "profile", "stats"}}."""
    datasets = {}
    for name, config in DATASET_CONFIG.items():
        df = pd.read_csv(DATA_DIR / config["file"])
        df = df.drop(columns=config["drop_columns"])
        for col in config["date_columns"]:
            df[col] = pd.to_datetime(df[col], format="%m/%d/%Y")
        datasets[name] = {
            "df": df,
            "profile": build_profile(name, config["description"], df),
            # Computed once here, not per question, so answering adds no pandas work.
            "stats": compute_stats(df, config["buckets"]),
        }
    return datasets


DATASETS = load_datasets()
