"""LLM layer: route a question to the most relevant dataset, then answer with it.

Usage: python main.py "Which language is searched the most?"   (add -v to log every step)
"""

import argparse
import json
import logging
import os
import time
from typing import Optional

import anthropic
from dotenv import load_dotenv

import prompts
from profiling import DATASETS

load_dotenv()

# Logs go to stderr, so stdout carries only the answer.
logging.basicConfig(level=logging.WARNING, format="%(message)s")
# Our loggers all sit under "app", so -v can turn on our detail
# without also turning on the SDK's debug output.
logging.getLogger("app").setLevel(logging.INFO)

# Indentation shows nesting: answer calls router, and both call llm.
INDENT = {"answer": "", "router": "  ", "llm": "    "}


def trace(layer: str, message: str, level: int = logging.DEBUG) -> None:
    """Log one line tagged with the layer that is speaking. DEBUG lines show only with -v."""
    logging.getLogger(f"app.{layer}").log(level, "%-9s%s%s", f"[{layer}]", INDENT[layer], message)


# Haiku: routing is a simple classification, so a small fast model is enough.
# It also accepts `temperature`; newer models (e.g. claude-sonnet-5-5) reject it.
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")


def call_llm(system: str, user: str, temperature: float) -> str:
    """Send one system + user message to the model and return its reply as a string."""
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    started = time.time()
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    trace("llm", f"{MODEL} took {time.time() - started:.1f}s, "
                 f"{response.usage.input_tokens} tokens in, {response.usage.output_tokens} out")
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

    trace("router", f"sending {len(DATASETS)} profiles ({len(user)} chars)")

    selection = None
    for attempt in range(2):  # one try + one retry
        raw = call_llm(prompts.ROUTER_SYSTEM, user, temperature=0)
        trace("router", f"raw reply: {raw!r}")
        selection = parse_selection(raw)
        if selection is not None:
            break
        trace("router", f"unusable reply on attempt {attempt + 1}", logging.WARNING)
        user += prompts.ROUTER_RETRY

    if selection is None:
        # Failing closed: better to answer nothing than answer from the wrong data.
        selection = {"dataset": "none", "reasoning": "The router reply could not be parsed."}

    # INFO, not DEBUG: the routing decision is always logged.
    trace("router", f"selected {selection['dataset']}: {selection['reasoning']}", logging.INFO)
    return selection


def answer_question(question: str, selection: Optional[dict] = None) -> str:
    """Route the question, then answer it from the selected dataset.

    Pass `selection` to reuse a routing decision that was already made.
    """
    started = time.time()
    trace("answer", f"question: {question!r}")
    if selection is None:
        selection = select_dataset(question)
    name = selection["dataset"]
    if name == "none":
        trace("answer", "no dataset selected, returning fixed message (no LLM call)")
        # List what we do have, so the user knows what they can ask about.
        available = "\n".join(f"- {n}: {d['profile']['description']}" for n, d in DATASETS.items())
        return prompts.NO_DATASET_MESSAGE.format(reasoning=selection["reasoning"], datasets=available)

    dataset = DATASETS[name]
    user = prompts.ANSWER_USER.format(
        profile=json.dumps(dataset["profile"], indent=2),
        stats=dataset["stats"],
        csv=dataset["df"].to_csv(index=False),  # 30 rows, so the whole table fits in the prompt
        question=question,
    )
    trace("answer", f"built context from profile + stats + {len(dataset['df'])} rows ({len(user)} chars)")
    answer = call_llm(prompts.ANSWER_SYSTEM, user, temperature=0)
    trace("answer", f"done in {time.time() - started:.1f}s")
    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Answer a question from the most relevant dataset.")
    parser.add_argument("question")
    parser.add_argument("-v", "--verbose", action="store_true", help="log every step of the pipeline")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("app").setLevel(logging.DEBUG)
    print(answer_question(args.question))
