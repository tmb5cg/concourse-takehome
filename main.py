"""LLM layer: route a question to the most relevant dataset, then answer with it.

Usage: python main.py "Which language is searched the most?"
"""

import json
import logging
import os
import sys
from typing import Optional

import anthropic
from dotenv import load_dotenv

import prompts
from profiling import DATASETS

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # hide per-request noise
log = logging.getLogger("router")

# Haiku: routing is a simple classification, so a small fast model is enough.
# It also accepts `temperature`; newer models (e.g. claude-sonnet-5-5) reject it.
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")


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
    # The router sees profiles only, never the data, so the prompt stays small.
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

    dataset = DATASETS[name]
    user = prompts.ANSWER_USER.format(
        profile=json.dumps(dataset["profile"], indent=2),
        stats=dataset["stats"],
        csv=dataset["df"].to_csv(index=False),  # 30 rows, so the whole table fits in the prompt
        question=question,
    )
    return call_llm(prompts.ANSWER_SYSTEM, user, temperature=0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python main.py "your question"')
    print(answer_question(" ".join(sys.argv[1:])))
