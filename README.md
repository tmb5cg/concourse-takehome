# Context selection for LLM question answering

Take-home for Concourse. Given a user question, pick which dataset (if any) can answer it, then answer it with an LLM using that dataset as context.

## How to run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then add your ANTHROPIC_API_KEY

python main.py "Which age group spends the most?"
python examples.py              # prints routing accuracy, rewrites examples.md
```

`examples.md` holds the output of a real run (10/10 routed correctly). `PROMPTS.md` holds the prompts used to build this and the prompts the code sends at runtime.

## Design

```
question -> select_dataset (LLM, temp 0) -> "none" -> fixed message
                                         -> dataset -> profile + stats + CSV -> answer (LLM)
```

- **Dataset profiles.** At load time each CSV becomes a profile: a human-written description, columns with dtypes, value ranges / categories / date range, and 3 sample rows. The router sees profiles, not data, so its prompt stays small however large the datasets are.
- **LLM router with a "none" option.** Routing is zero-shot classification over the profiles. Keyword matching breaks on wording like "big earners", and a trained classifier needs retraining for every new dataset. "none" covers off-topic questions and questions needing both datasets, which share no join key. The reply is JSON with a `reasoning` field; on a bad reply it retries once, then fails closed to "none".
- **Stats computed in code.** LLMs are unreliable at arithmetic, so pandas computes the numbers (group averages, correlation, min/max weeks, trend) and the prompt tells the model to quote them. Every number in an answer traces back to a line of code.
- **PII drop.** `Customer ID` is removed at load, before anything reaches the LLM.
- **Logging.** Every routing decision is logged with its reasoning.

## Known limitations

- Stats are a fixed set per dataset. Questions outside them (e.g. a median for one gender) make the model read raw rows, which is less reliable.
- The model can still do small arithmetic despite the instruction (one example answer says "about 4 points"). The prompt reduces this; it doesn't enforce it.
- No confidence score: a self-reported LLM confidence isn't calibrated, so only the reasoning is returned.
- Accuracy is measured on 10 hand-written questions, which is a smoke test, not an evaluation.
- The default model is Claude Haiku 4.5 because newer models reject the `temperature` parameter.
- Single-turn only, and one dataset per question.

## Next steps at scale

- **Many datasets:** embed the profiles, retrieve the top-k with ANN search (e.g. HNSW) plus metadata filters, then have the LLM rerank those k.
- **Large datasets:** stop pasting data. Give the model a text-to-SQL or DuckDB/Postgres tool and let it query.
- **Multi-dataset questions:** detect join keys from the profiles (matching column names, or value overlap for differently named columns) and route to a set of datasets.
- **Ingestion:** build and cache profiles when a dataset is ingested, not on every start.
- **Evaluation:** a larger labeled question set, run on every prompt change.
