# Using DocuMind

DocuMind answers questions about your own documents. You upload files into
collections, DocuMind reads and indexes them, and then you can search them
or ask questions in plain language. Every answer cites the passages it was
built from, so you can check it.

## Collections

A collection is a folder for related documents, such as "Course notes" or
"Product manuals". Questions and searches always run inside one collection,
so answers only ever draw on the documents you put there.

## Uploading documents

DocuMind accepts PDF, DOCX, Markdown and plain-text files. After an upload
the document is "pending", then "processing", then "ready". Processing
extracts the text, splits it into overlapping chunks of a few hundred words,
and turns each chunk into an embedding: a list of numbers that captures what
the passage is about. A document that cannot be read, such as a scanned PDF
without a text layer, ends up "failed" with a reason.

Free-tier limits apply: each account has a daily number of uploads, a
storage limit and a daily embedding allowance.

## Searching

Search finds the passages closest to your query. It combines two methods:
meaning (the embeddings, so "automobile" finds "car") and keywords
(Postgres full-text search, so exact names and numbers are found too). The
two result lists are merged with Reciprocal Rank Fusion.

## Asking questions

When you ask a question, DocuMind searches the collection, keeps the most
relevant passages and gives them to a language model with strict
instructions: answer only from these sources and cite them by number. If
the sources don't contain the answer, DocuMind says so instead of guessing.

Follow-up questions keep their context. "What about the second one?" is
first rewritten into a standalone question using the conversation so far.
Answers can also be streamed, appearing word by word.

## The demo account

The demo account is read-only and shared by every visitor. It can ask,
search and read the sample documents, but it cannot upload, delete or
change anything, and other visitors can see the conversations it has. To
upload your own documents, create your own account.
