# Retrieval evaluation: results and chosen defaults

Measured 2026-10-21 with `voyage-4-lite` (1024 dimensions) on the evaluation set in
[`eval/`](../eval/README.md): 7 self-written documents about a made-up company
(47 chunks at 500 tokens) and 21 questions:

- 6 answerable;
- 3 keyword (error codes, part numbers);
- 5 paraphrase;
- 3 short;
- 4 unanswerable.

Only retrieval is measured: no LLM was called.

**Cost.** The whole matrix below made 10 Voyage requests (about 8,000 tokens; 72
texts embedded, 282 served from the embedding cache) and 0 Gemini requests. The
waits came from the 3-requests-per-minute limit, so it took a few minutes (about 4 of
them waiting). Every
later run, sweep and experiment made 0 provider requests.

Reproduce:

```bash
uv run python -m eval.experiments
uv run python -m eval.sweep --mode vector --keyword-slack 0 --start 0.5 --stop 0.8
uv run python -m eval.sweep --mode hybrid --keyword-slack 0.1 --start 0.5 --stop 0.8
```

## 1. Search modes, keyword matching, text search config, chunk size

All at the old threshold `RAG_MAX_DISTANCE=0.72`, with no keyword slack. Hit rates
and false rejections are over the 17 answerable questions; "Rejected" is over the 4
unanswerable ones. MRR shows n/a until `mrr()` is implemented.

| Experiment | Chunks | hit@1 | hit@3 | hit@5 | MRR | False rej. | Rejected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vector | 47 | 76% | 94% | 100% | n/a | 6% | 25% |
| keyword strict | 47 | 24% | 24% | 24% | n/a | 82% | 100% |
| keyword loose | 47 | 47% | 76% | 82% | n/a | 6% | 50% |
| hybrid strict | 47 | 76% | 94% | 100% | n/a | 6% | 25% |
| hybrid loose | 47 | 82% | 82% | 88% | n/a | 6% | 25% |
| hybrid loose, `simple` | 47 | 76% | 82% | 82% | n/a | 6% | 50% |
| vector, 300 tokens | 51 | 82% | 94% | 100% | n/a | 6% | 25% |
| hybrid loose, 300 | 51 | 82% | 82% | 88% | n/a | 6% | 25% |
| vector, 800 tokens | 46 | 82% | 94% | 100% | n/a | 6% | 25% |
| hybrid loose, 800 | 46 | 76% | 82% | 88% | n/a | 6% | 25% |

With 17 answerable questions, one question is 6 percentage points. Differences of
one question are noise, not findings.

What the numbers say:

- **Vector search does most of the work.** Every answerable question's right chunk
  is in its top 5. The misses at rank 1 are mostly paraphrases ranked 2nd–4th, usually
  behind a neighbouring chunk of the *same* document: the right document, a near-miss
  chunk.
- **Keyword-only search fails on natural questions.** In strict mode a chunk must
  contain every word of the question, so it found only 4 of 17. It is
  excellent at what it is for: "ERR-4471" and "ZX-9" were rank 1. Loose (any word)
  finds more, but ranks chunks full of common words ("lamp", "customer") above the
  answer.
- **Hybrid strict ranks exactly like vector on this set.** The strict keyword list
  is nearly empty for natural questions, so RRF keeps the vector order. Where
  it does match, on rare terms, it agrees with or rescues the vector result. The
  rescue case (a rare-term chunk ranked low by vectors) is tested in
  `tests/test_hybrid.py`; this corpus didn't happen to contain one.
- **Hybrid loose is worse where it matters.** It gains one hit@1 but loses at hit@5
  (88% against 100%): paraphrase questions fall to rank 9–12 because the loose
  keyword list votes for wordy, off-topic chunks. A chunk that isn't in the top 5
  can't be cited; one at rank 2 still can. So **strict** stays the default.
- **`english` beats `simple`.** `simple` keeps stop words, so "What is part ZX-9?"
  needs "what" and "is" in the chunk too. The keyword match disappears, and with it
  the gate's keyword path (below). `english` also stems, so "refunded" finds
  "refund".
- **Chunk size barely matters here.** Every Markdown section is shorter than 300
  tokens, and chunks never cross a heading, so only the plain-text handbook is
  chunked differently (51 / 47 / 46 chunks). 300 and 800 each gained one hit@1 on
  the vector side: within noise. **500 stays.** A corpus of long, heading-free
  documents is needed to measure this properly (see weak spots).

## 2. The relevance threshold

`python -m eval.sweep` retrieves once, then re-applies the gate at each threshold.
Below is the closest chunk the gate sees for every question (vector order, top 8),
then the two rates:

```
  A 0.372 refund-timing          A 0.546 blinking-lamp
  A 0.375 points-per-euro        A 0.546 short-customs
  A 0.376 battery-warranty       A 0.559 smoking-lamp
  A 0.398 points-expiry          U 0.559 no-phone-number
  A 0.414 free-shipping-thr.     A 0.639 sl-77
  A 0.499 not-home               A 0.644 short-exchanges   (all-terms keyword #1)
  A 0.509 goodwill-limit         A 0.647 err-4471          (all-terms keyword #1)
  A 0.513 short-chat-hours       U 0.648 no-parental-leave
  A 0.514 old-lamp-broken        U 0.680 no-revenue
  A 0.519 change-of-mind         U 0.734 no-sourdough
                                 A 0.769 part-zx9          (all-terms keyword #1)
```

| max distance | vector: false rej. | vector: rejected | hybrid + slack 0.10: false rej. | hybrid + slack 0.10: rejected |
| ---: | ---: | ---: | ---: | ---: |
| 0.54 | 41% | 100% | 41% | 100% |
| 0.56 | 24% | 75% | 12% | 75% |
| 0.64 | 18% | 75% | 6% | 75% |
| 0.66 | 6% | 50% | 6% | 50% |
| 0.68 | 6% | 50% | 0% | 50% |
| **0.70** | 6% | 25% | **0%** | **25%** |
| 0.72 (old) | 6% | 25% | 0% | 25% |
| 0.74 | 6% | 0% | 0% | 0% |
| 0.78 | 0% | 0% | 0% | 0% |

**No threshold is perfect.** Rejecting every unanswerable question needs ≤ 0.54,
which turns away 41% of answerable ones. Never turning one away (vector alone) needs
≥ 0.78, which lets every unanswerable question through. The two groups overlap.
"What is the customer service phone number?" (0.559) is *about* the support
handbook; the handbook just doesn't contain the number. Distance measures topic,
not whether the answer is there.

### The gate rule (`app/services/relevance.py`)

1. A chunk passes if its vector distance ≤ `RAG_MAX_DISTANCE`, in every mode.
2. In hybrid mode it also passes if it is a **strong keyword match** (contains
   every query term, among the top 3 keyword results) and its distance is
   ≤ `RAG_MAX_DISTANCE + RAG_KEYWORD_GATE_SLACK`.
3. If no chunk passes, `/ask` answers "not found" without calling the LLM.

Why this rule: rare exact terms are what embeddings handle worst. "What is part
ZX-9?" has its right chunk at rank 1 in both lists, yet at distance 0.769, so a
vector-only gate rejects it at any threshold that also rejects anything
unanswerable. An all-terms keyword hit is strong independent evidence. The slack
is bounded, so a question that merely shares one word with a document still
needs to be close in meaning. On this set the keyword path let through no
unanswerable question at any threshold: its rejection column equals the vector one.

### Chosen defaults

| Setting | Old | New | Why |
| --- | --- | --- | --- |
| `RAG_SEARCH_MODE` | `vector` | `hybrid` | Same ranking as vector here, plus the keyword gate path and rare-term rescue |
| `RAG_KEYWORD_MATCH` | `strict` | `strict` | Loose dropped hit@5 from 100% to 88% |
| `TEXT_SEARCH_CONFIG` | `english` | `english` | `simple` lost the ZX-9 keyword match |
| `RAG_MAX_DISTANCE` | 0.72 | **0.70** | See below |
| `RAG_KEYWORD_GATE_SLACK` | 0 | **0.10** | Recovers the rare-term question rejected by distance |
| `CHUNK_TOKENS` | 500 | 500 | No measurable difference on this corpus |

Result, compared with the old defaults: false rejections 6% → **0%**, rejection rate
25% → 25%, ranking unchanged (hit@1 76%, hit@3 94%, hit@5 100%).

Why 0.70 and not the "best-looking" 0.68: at 0.68, "no-revenue" (0.6802) is rejected
by a margin of 0.0002, which is luck, not a property of the threshold. 0.70 instead
has margins that would survive small changes:

- 0.06 above the hardest answerable question that relies on distance alone (sl-77, 0.639);
- 0.03 of slack to spare for ZX-9 (0.769 against 0.80);
- 0.034 below the off-topic question it rejects (sourdough, 0.734).

We prefer false acceptances to false rejections. An answerable question turned away
is a dead end for the user. An unanswerable one let through still meets the LLM's
grounding rules (it must cite its sources, and `grounded` is false otherwise),
costs one Gemini call, and usually ends in "the documents don't say".

## 3. Weak spots

- **The gate can't tell "on topic but not covered" from "answerable".** It rejects
  only 1 of 4 unanswerable questions. Phone number, parental leave and company
  revenue all sit as close as real answers. A distance threshold can't fix this. A
  reranker (a cross-encoder scoring question and passage together) or an
  answerability check would, but both cost model calls; the LLM's grounding
  rules are the current backstop.
- **Boilerplate attracts questions.** Every document opens with the same "fictional
  company" paragraph, and "What was Larkspur's revenue?" matches it at 0.680 in
  three documents. Real documents have the same problem (disclaimers, headers,
  copyright notices).
- **Paraphrases often land on a neighbouring chunk** of the right document (rank 2–4).
  The answer still reaches the LLM (top 8), but citations may point at the wrong
  passage.
- **The set is small.** With 17 answerable questions, one question is 6 points. The
  threshold has visible margins, but they rest on a handful of distances. Add
  questions, especially unanswerable ones and real documents, before trusting
  differences of a few points.
- **Chunk size is barely tested** because the Markdown sections are short. Add long,
  heading-free documents (`.txt`, PDFs) to measure it.
- **Distances are model-specific.** After changing `EMBEDDING_MODEL`, the threshold
  must be measured again; the cache keys by model, so the old vectors are simply
  not reused.
- **MRR** isn't reported until `eval/metrics.py: mrr()` is implemented.
