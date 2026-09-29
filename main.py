"""Context selection: route a question to the most relevant dataset, then answer with it.

Usage: python main.py "Which language is searched the most?"
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import anthropic
import pandas as pd
from dotenv import load_dotenv

import prompts

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # hide per-request noise
log = logging.getLogger("router")

# Haiku: routing is a simple classification, so a small fast model is enough.
# It also accepts `temperature`; newer models (e.g. claude-sonnet-5-5) reject it.
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
DATA_DIR = Path(__file__).parent / "data"

# Descriptions are human-written: column names alone don't say what a
# "Spending Score" is or that search values are relative interest.
DATASET_CONFIG = {
    "shopping_habits": {
        "file": "shopping_habits.csv",
        "description": (
            "Demographics of individual retail customers (age, gender, annual income) "
            "and a 1-100 score for how much each one shops / spends. One row per customer."
        ),
        "drop_columns": ["Customer ID"],  # PII: never sent to the LLM
        "date_columns": [],
    },
    "weekly_searches": {
        "file": "weekly_searches_for_programming_languages.csv",
        "description": (
            "Weekly search-engine interest for three programming languages "
            "(Python, Java, C++). One row per week."
        ),
        "drop_columns": [],
        "date_columns": ["Week"],
    },
}


def call_llm(system: str, user: str, temperature: float) -> str:
    """Send one system + user message to the model and return its reply as a string."""
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    # A reply is a list of content blocks; keep only the text ones.
    return "".join(block.text for block in response.content if block.type == "text")


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


def load_datasets() -> dict:
    """Load every CSV, clean it, and build its profile. Returns {name: {"df", "profile"}}."""
    datasets = {}
    for name, config in DATASET_CONFIG.items():
        df = pd.read_csv(DATA_DIR / config["file"])
        df = df.drop(columns=config["drop_columns"])
        for col in config["date_columns"]:
            df[col] = pd.to_datetime(df[col], format="%m/%d/%Y")
        profile = build_profile(name, config["description"], df)
        datasets[name] = {"df": df, "profile": profile}
    return datasets


def shopping_stats(df: pd.DataFrame) -> str:
    """Precompute the numbers a shopping question is likely to need."""
    numeric = ["Age", "Annual Income", "Spending Score"]

    by_gender = df.groupby("Gender")[numeric].mean().round(1)
    by_gender.insert(0, "Count", df.groupby("Gender").size())

    age_bucket = pd.cut(df["Age"], bins=[0, 34, 54, 120], labels=["under 35", "35-54", "55+"])
    by_age = df.groupby(age_bucket, observed=True)[["Annual Income", "Spending Score"]].mean().round(1)
    by_age.insert(0, "Count", df.groupby(age_bucket, observed=True).size())

    correlation = df["Annual Income"].corr(df["Spending Score"])

    return (
        f"Overall (n={len(df)}):\n{df[numeric].describe().round(1).to_string()}\n\n"
        f"Averages by gender:\n{by_gender.to_string()}\n\n"
        f"Averages by age bucket:\n{by_age.to_string()}\n\n"
        f"Pearson correlation, Annual Income vs Spending Score: {correlation:.2f}"
    )


def searches_stats(df: pd.DataFrame) -> str:
    """Precompute the numbers a search-trend question is likely to need."""
    lines = [f"Weeks covered: {len(df)}"]
    for language in [col for col in df.columns if col != "Week"]:
        values = df[language]
        highest = df.loc[values.idxmax()]
        lowest = df.loc[values.idxmin()]
        # Trend = first 4 weeks vs last 4 weeks. Cruder than a regression,
        # but easy to verify by hand.
        first, last = values.head(4).mean(), values.tail(4).mean()
        lines.append(
            f"{language}: mean {values.mean():.1f}, min {values.min()}, max {values.max()}, "
            f"highest week {highest['Week'].date()} ({highest[language]}), "
            f"lowest week {lowest['Week'].date()} ({lowest[language]}), "
            f"first-4-week avg {first:.1f} vs last-4-week avg {last:.1f} (change {last - first:+.1f})"
        )
    return "\n".join(lines)


DATASETS = load_datasets()
STATS_FUNCTIONS = {"shopping_habits": shopping_stats, "weekly_searches": searches_stats}


def parse_selection(raw: str) -> Optional[dict]:
    """Turn the router's reply into {"dataset", "reasoning"}, or None if it's unusable."""
    # Take everything between the first { and last } so code fences or
    # stray prose around the JSON don't break parsing.
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None

    dataset = str(parsed.get("dataset", "")).strip().lower()
    if dataset != "none" and dataset not in DATASETS:
        return None
    return {"dataset": dataset, "reasoning": str(parsed.get("reasoning", ""))}


def select_dataset(question: str) -> dict:
    """Ask the LLM which dataset (or "none") can answer the question."""
    profiles = "\n\n".join(json.dumps(d["profile"], indent=2) for d in DATASETS.values())
    user = prompts.ROUTER_USER.format(
        profiles=profiles, names=", ".join(DATASETS), question=question
    )

    selection = None
    for attempt in range(2):  # one try + one retry
        raw = call_llm(prompts.ROUTER_SYSTEM, user, temperature=0)
        selection = parse_selection(raw)
        if selection is not None:
            break
        log.warning("Unusable router reply (attempt %d): %r", attempt + 1, raw)
        user += prompts.ROUTER_RETRY

    if selection is None:
        # Failing closed: better to answer nothing than answer from the wrong data.
        selection = {"dataset": "none", "reasoning": "The router reply could not be parsed."}

    log.info("question=%r dataset=%s reasoning=%r", question, selection["dataset"], selection["reasoning"])
    return selection


def answer_question(question: str, selection: Optional[dict] = None) -> str:
    """Route the question, then answer it from the selected dataset.

    Pass `selection` to reuse a routing decision that was already made.
    """
    if selection is None:
        selection = select_dataset(question)
    name = selection["dataset"]
    if name == "none":
        return prompts.NO_DATASET_MESSAGE.format(reasoning=selection["reasoning"])

    df = DATASETS[name]["df"]
    user = prompts.ANSWER_USER.format(
        profile=json.dumps(DATASETS[name]["profile"], indent=2),
        stats=STATS_FUNCTIONS[name](df),
        csv=df.to_csv(index=False),  # 30 rows, so the whole table fits in the prompt
        question=question,
    )
    return call_llm(prompts.ANSWER_SYSTEM, user, temperature=0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python main.py "your question"')
    print(answer_question(" ".join(sys.argv[1:])))
