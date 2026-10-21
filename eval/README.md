# Retrieval evaluation

This folder measures how well DocuMind **finds** the right passage for a
question. It never asks the LLM for an answer: it only checks what retrieval
returns. That keeps it free to run, repeatable, and focused on the part that
decides whether a good answer is possible at all.

```
eval/
  corpus/          the documents that get ingested (all of them, every run)
  questions.yaml   the questions, and where each answer lives
  dataset.py       loads and validates questions.yaml
  metrics.py       hit@k, MRR, rejection rates (pure functions)
  run.py           python -m eval.run: one evaluation, one report
  sweep.py         python -m eval.sweep: the gate at many RAG_MAX_DISTANCE values
  experiments.py   python -m eval.experiments: the comparison matrix
  reports/         JSON reports (not committed)
```

## The corpus

Everything in `corpus/` was written for this project and describes a made-up
company, Larkspur Home. It is released under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/): no rights
reserved.

### Adding a document

1. Only add documents you have the right to commit and share: your own
   writing, public-domain texts, or openly licensed ones (CC0, CC BY, ...)
   whose licence allows redistribution. **Never** commit internal company
   files, customer data, paid reports, or anything copied from a website
   without a licence. If a document is not CC0, say which licence applies
   and where it came from in the list below.
2. Save it in `corpus/` as `.md`, `.txt`, `.pdf` or `.docx` (the formats the
   app accepts). The file name is how questions refer to it, so pick a
   stable, descriptive one.
3. Add questions for it (next section). A document nobody asks about still
   helps: it is a distractor that retrieval must not pick.
4. Run `python -m eval.dataset` to check the file loads (see below).

Changing a document's text changes its chunks, so the next run embeds those
chunks once (the cache makes unchanged chunks free).

| File | Source | Licence |
| --- | --- | --- |
| `returns-and-refunds.md` | written for DocuMind | CC0 |
| `warranty-and-repairs.md` | written for DocuMind | CC0 |
| `shipping-and-delivery.md` | written for DocuMind | CC0 |
| `ops-runbook.md` | written for DocuMind | CC0 |
| `product-care.md` | written for DocuMind | CC0 |
| `loyalty-programme.md` | written for DocuMind | CC0 |
| `support-handbook.txt` | written for DocuMind | CC0 |

## The questions

Each entry in `questions.yaml`:

```yaml
- id: refund-timing            # unique; lowercase letters, digits, dashes
  type: paraphrase             # answerable | keyword | paraphrase | short | unanswerable
  question: When do I get my money back after a return?
  expected:                    # omitted for unanswerable questions
    document: returns-and-refunds.md
    must_contain: the refund is paid within 14 days
    page: 3                    # optional, for PDFs
```

A retrieved chunk counts as **the right one** when it comes from `document`,
contains `must_contain` (ignoring case and line breaks), and is on `page`
if one is given.

Why a passage and not a chunk id: chunk ids are new every time a document is
processed, and every chunk-size experiment produces different chunks. A
passage stays valid across all of them.

Tips for good questions:

- **`must_contain`**: a short phrase (under 200 characters) that appears
  only where the answer is. Shorter is safer: a long passage can be split
  between two chunks at small chunk sizes and then match neither.
- **Types** let the report show *where* retrieval is weak. `keyword` covers
  rare exact terms (error codes, part numbers) that embeddings tend to blur;
  `paraphrase` uses different words from the document, which keyword search
  can't match; `short` is what people type into a search box.
- **`unanswerable`** questions are just as important: they check that the
  relevance gate says "nothing found" instead of passing junk to the LLM.
  Mix obviously off-topic ones with plausible ones the corpus happens not to
  cover.

### Checking the file

```bash
uv run python -m eval.dataset
```

Prints a summary, or the question and reason for the first problem: unknown
field, a document missing from `corpus/`, a `must_contain` that doesn't
appear in the document's extracted text, a duplicate id, and so on.

## Running it

Needs the database and Redis from `docker compose up -d db redis` (the
Celery worker can keep running: the runner processes every document it
creates immediately, so the worker never sees one pending). Run from the
project root:

```bash
uv run python -m eval.run                          # settings from .env
uv run python -m eval.run --mode hybrid --max-distance 0.6
uv run python -m eval.sweep --mode hybrid          # thresholds 0.40 .. 0.90
uv run python -m eval.experiments                  # the whole matrix
```

Options (all default to the app's settings): `--mode vector|keyword|hybrid`,
`--chunk-tokens`, `--overlap`, `--max-distance`, `--keyword-slack`,
`--keyword-match strict|loose`, `--text-search-config`, `--top-k`.

What happens:

- A user `eval@documind.local` (random password; it can't log in) owns one
  collection per chunk size, e.g. "eval: 500 tokens, 60 overlap". It is
  separate from every real user's data and safe to delete.
- Each corpus file is uploaded and processed with the **real** pipeline.
  Unchanged documents already processed with the same model are skipped.
- The questions are embedded in one batch, then searched exactly as `/ask`
  would search. **The LLM is never called.**
- Embeddings go through the cache table `embedding_cache`. The first run on
  an empty database embeds the corpus and questions (Voyage, a few requests,
  within the shared rate limit); every later run with the same text makes
  **zero** provider calls. The report's last line says how many were made.

## Reading the report

```
type            n  hit@1  hit@3  hit@5  falseRej  rejected
all            21    71%    88%    94%       6%      100%
keyword         3    67%   100%   100%       0%       n/a
...
MRR: 0.812
Failed questions: 4
  blinking-lamp            paraphrase   rank 4
      returns-and-refunds.md   d=0.512 v#1 k#-
```

- **hit@k**: of the answerable questions, how often the right chunk was in
  the first k results. hit@5 low = the passage isn't being found at all;
  hit@5 high but hit@1 low = found, but ranked below other chunks.
- **MRR**: the average of 1/rank of the right chunk (0 if missing). Unlike
  hit@k it rewards rank 1 over rank 3.
- **falseRej** (false rejection rate): answerable questions the relevance
  gate turned away, so `/ask` would have said "I couldn't find that". Lower
  is better.
- **rejected** (rejection rate): unanswerable questions the gate turned
  away. Higher is better: otherwise unrelated chunks reach the LLM.
- **Failed questions**: answerable ones whose right chunk wasn't first (with
  its rank, or "not in top 20"), or that were rejected; unanswerable ones the
  gate let through. Under each, the top results: `*` marks the right chunk,
  `d` its vector distance, `v#`/`k#` its rank in the vector and keyword
  lists (`-` = not in that list).

The JSON report in `eval/reports/` holds the same numbers, the full
configuration (search mode, chunk size and overlap, embedding model,
threshold, keyword options) and every question's top results, so two
reports can be compared later.

The sweep prints, per question, the distance of the closest chunk the gate
looks at, then the two rejection rates at every threshold, and the range of
thresholds (if any) that gets both right on this set.

Results and chosen defaults: [docs/EVALUATION.md](../docs/EVALUATION.md).
