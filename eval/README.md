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
