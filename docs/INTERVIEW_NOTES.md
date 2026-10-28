# Interview notes

Questions an interviewer is likely to ask about DocuMind. The answers are
mine to write; the linked files hold the facts to draw on.

## 1. Why pgvector instead of a dedicated vector database?

See [DECISIONS.md](DECISIONS.md#1-pgvector-inside-postgres),
[HARDENING.md](HARDENING.md) section 1.

My answer: TODO(me)

## 2. How do you stop one user from seeing another user's data?

See [ARCHITECTURE.md](ARCHITECTURE.md#data-model),
[SECURITY.md](SECURITY.md).

My answer: TODO(me)

## 3. What happens if the worker dies in the middle of a job?

See [ARCHITECTURE.md](ARCHITECTURE.md#the-ingestion-sequence).

My answer: TODO(me)

## 4. How did you choose the relevance threshold?

See [EVALUATION.md](EVALUATION.md), section 2.

My answer: TODO(me)

## 5. Why hybrid search, and why Reciprocal Rank Fusion instead of adding scores?

See [ARCHITECTURE.md](ARCHITECTURE.md#hybrid-retrieval),
[DECISIONS.md](DECISIONS.md#3-hybrid-search-with-rrf-and-a-measured-threshold).

My answer: TODO(me)

## 6. What is prompt injection, and what did you do about it?

See [SECURITY.md](SECURITY.md#prompt-injection).

My answer: TODO(me)

## 7. How do you know the citations in an answer are real?

See [ARCHITECTURE.md](ARCHITECTURE.md#how-a-question-flows-through-the-system).

My answer: TODO(me)

## 8. Why is the free-tier demo set up differently from the full setup?

See [DECISIONS.md](DECISIONS.md#4-celery-in-full-mode-inline-jobs-in-lite-mode),
[OPERATIONS.md](OPERATIONS.md#lite-mode-free-tier).

My answer: TODO(me)

## 9. How do you protect a free-tier API quota from abuse?

See [SECURITY.md](SECURITY.md#daily-quotas),
[HARDENING.md](HARDENING.md) section 6.

My answer: TODO(me)

## 10. Why does streaming send a final `done` event if the tokens already contain the answer?

See [API.md](API.md#streaming-answers).

My answer: TODO(me)

## 11. How would you scale it to many more users?

See [HARDENING.md](HARDENING.md), [OPERATIONS.md](OPERATIONS.md).

My answer: TODO(me)

## 12. What would you change if you started again?

My answer: TODO(me)

## 13. How do you test code that calls paid external APIs?

See [CI.md](CI.md), [OPERATIONS.md](OPERATIONS.md#tests).

My answer: TODO(me)
