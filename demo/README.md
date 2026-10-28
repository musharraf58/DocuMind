# Demo corpus

The documents in `corpus/` are what the shared demo account
(`python -m scripts.seed_demo`) can search and ask about.

**Rights:** every document here was written for this project and may be
freely redistributed with it. No text was copied from books, websites or
other copyrighted sources. Add only texts you wrote yourself or that are in
the public domain (and say which in this file), never documents you don't
have the rights to.

| File | Source |
| --- | --- |
| `documind-guide.md` | written for this project |
| `coffee-brewing.md` | written for this project |
| `houseplant-care.md` | written for this project |
| `bicycle-maintenance.md` | written for this project |

Questions that work well:

- "What grind size should I use for a French press?"
- "How do I know if I'm overwatering a plant?"
- "How often should I lubricate my bike chain?"
- "How does DocuMind search work?"
- And one the sources can't answer, to see it decline: "What is the best
  espresso machine?"

The seed script stores `.md` and `.txt` files from `corpus/` only, keeps the
demo collection in sync with this folder (new files are added, removed
files are deleted, changed files are replaced), and embeds through the
embedding cache, so seeding again costs no provider calls for text that was
embedded before.
