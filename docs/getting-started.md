# Getting started

## Install

```bash
python3 -m venv venv
./venv/bin/pip install -e .
cp .env.example .env      # add AZURE_ENDPOINT and AZURE_API_KEY
```

## Run an analysis (no API calls)

Every model response is cached in `data/`, so all analysis is offline.

```bash
./venv/bin/python -m open_ended_aggregation.analysis.compare_qampari
./venv/bin/python -m open_ended_aggregation.analysis.compare_quest
./venv/bin/python -m open_ended_aggregation.analysis.subset_sweep
./venv/bin/python -m open_ended_aggregation.analysis.verify_qampari
```

Pass a specific generation file to pin a sample size:

```bash
./venv/bin/python -m open_ended_aggregation.analysis.compare_qampari data/qampari_gen.jsonl.n198.bak
```

`data/` is gitignored (~250 MB, ~$130 of Azure spend) and **not backed up anywhere**.
Archive it before doing anything destructive.

## Generate new data

Generation is **resumable and idempotent**: `done_keys()` skips any `(qid, model)` pair
already on disk, so re-running only fills gaps.

```bash
./venv/bin/python -m open_ended_aggregation.benchmarks.quest 400
./venv/bin/python -m open_ended_aggregation.generation.qampari_800 800
```

Run generation in **passes**, not one long process. A long-lived generator degrades to
~50% silent drop rate under sustained load while a fresh one succeeds ~100% on identical
inputs; the drivers in `generation/` loop until a pass adds nothing. See
[troubleshooting.md](troubleshooting.md).

## Add a benchmark

Copy `benchmarks/quest.py`. A benchmark module owns five things and shares none of them:

1. `load_items(n)` → `[{qid, question, gold}]`
2. `PROMPT`
3. `parse_list(txt)` → atomic items
4. `score_set(pred, gold)` → metrics
5. `GEN_PATH` and `POOL`

Do not factor these into a shared base. The datasets have different answer shapes and
different metric properties, and coupling them is what produced the earlier
"ASQA cannot replicate" confusion.
