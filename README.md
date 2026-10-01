# Product Catalogue Q&A Agent (RAG, CLI)

A command-line agent that answers questions about Gutermann's water leak detection product catalogue, using a single markdown file (`product_overview.md`) as its knowledge base and retrieval-augmented generation (RAG) to keep answers grounded in that document. Run it with `python qa_agent.py` and type questions at the `>` prompt.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                 # Windows: copy .env.example .env
# open .env and replace your_key_here with your Gemini API key (from Google AI Studio)

python qa_agent.py
```

- **Embeddings run locally** (sentence-transformers). The model is downloaded once on first run (about 90 MB).
- **Answer generation uses the Gemini API**, so it needs a `GEMINI_API_KEY` in `.env` and an internet connection. The `.env` file is git-ignored and never committed.
- Type `exit` or `quit` to end the session.

Settings sit at the top of `qa_agent.py`: `TOP_K`, `KEYWORD_WEIGHT`, `LLM_MODEL`, `FALLBACK_MODEL`, `SHOW_RETRIEVAL` (prints retrieved chunks and scores, useful for debugging).

## Architecture

```
product_overview.md -> load_and_chunk() -> embed_chunks()      (once, at startup)
                                                |
                                                v
user query -> embed query -> retrieve() -> generate_answer() -> printed answer + [Source: ...]
```

| Function | Responsibility |
|---|---|
| `load_and_chunk(filepath)` | Parses the markdown into one chunk per product, with metadata |
| `embed_chunks(chunks)` | Adds an embedding vector to every chunk (kept in memory) |
| `retrieve(query, chunks, k)` | Scores every chunk against the query and returns the top-k |
| `generate_answer(query, retrieved_chunks)` | Sends the chunks and the question to the LLM and returns a grounded answer |
| `main()` | The CLI loop that ties these together |

## Chunking Strategy

**The document is chunked by structure: one chunk per `###` product heading.** Each chunk holds the product's one-line description and all of its feature bullets together as a single unit. There is no fixed character or token size anywhere in the code.

**Why not fixed-size chunks?** The catalogue is a list of short, self-contained product entries (a description plus 3-7 bullets). A fixed-size window would cut through them: it can split one product's facts across two chunks, or merge the end of one product with the start of the next. Once that happens, a feature can end up next to the wrong product name, which causes hallucinations on presence/absence questions, for example mixing up which ZONESCAN product has hydrophone technology. With one chunk per product, every bullet always stays attached to its own product.

**Metadata attached to each chunk** (extracted with plain markdown parsing, no ML):
- `product_name` (the `###` heading)
- `category` (the parent `##` heading)
- `status` (any availability line, such as "Launching Q2 2026")
- `category_notes` (text that applies to every product in the category)
- `raw_text` (the description and bullets)

**Catalogue quirks the parser handles:**
- **Shared category text.** The "Permanent Leak Detection Monitoring" section has an intro and bullets (no drilling, NB-IoT, 95% connectivity) that sit above its first `###` and apply to both ZONESCAN AI and ZONESCAN HYDRO. These are stored as `category_notes`, embedded with each product in that category, and passed to the LLM labelled as applying to the whole category.
- **A missing heading.** "Acoustic Leak Detection Microphones" came through the PDF conversion as plain text instead of a `##`. A plain line after a product's bullets is treated as a new category.
- **Converter noise.** Image links and HTML comments are stripped, stray tabs are collapsed, and three lost-ligature typos ("Ofline", "ofering", "trafic") are corrected.

## Retrieval Approach

- **Embedding model:** `all-MiniLM-L6-v2` via sentence-transformers. It runs locally, so retrieval needs no API for a knowledge base this small (14 chunks), and it is fast enough to embed everything at startup.
- **What gets embedded:** product name + category + category notes + product text, so queries can match on category context too.
- **Similarity:** cosine similarity between the query embedding and every chunk embedding (vectors are normalized, so this is a dot product).
- **Keyword boost:** the final score is `cosine + KEYWORD_WEIGHT * keyword_overlap`, where the overlap is the fraction of the query's meaningful words that appear in the chunk (stopwords ignored, simple plural handling). I added this after the first test: for "What is the Minimum Level Profiling feature...", pure cosine ranked the right product (AQUASCOPE 3) 7th of 14, because the small embedding model scored all chunks low and unspecifically. An exact feature name is a strong signal that embeddings can miss. Setting `KEYWORD_WEIGHT = 0` gives pure cosine similarity again.
- **`k`:** `TOP_K` is a named constant at the top of the file (default 3). Raise it to 4-5 for questions like "which products use X".

There is deliberately **no rules-based query router**. The catalogue is small enough that one would work, but it would not carry over to large, messy factory documentation, so the agent stays fully retrieval-based.

## Grounding / Hallucination Prevention

The system prompt tells the model to:
- answer **only** from the provided context, with no outside knowledge
- reply exactly *"The document doesn't contain enough information to answer that."* when the answer is not in the retrieved chunks
- never attribute a feature of one product to another, and treat a feature missing from a product's block as absent
- respect each product's `Status` line, for example saying a product is not available to order yet
- name the products it actually used in a final `SOURCES:` line

How the code supports this:
- Each chunk goes to the LLM as a labelled block (Product, Category, Status, Category notes, Details), so availability facts like "Launching Q2 2026" appear on their own line instead of being buried in a bullet.
- Temperature is 0.
- The `SOURCES:` line is parsed and checked against the retrieved product names, so the agent can only cite chunks it was actually given. The line is removed from the printed answer, and no source is shown when the answer is the "not enough information" reply.
- If the model fails to give a sources line, the agent prints `[Retrieved: ...]` instead, which is labelled honestly as what was retrieved, not what was used.

**API failures:** a 503 (model overloaded) is retried with a short backoff, then the agent switches to `FALLBACK_MODEL`. A 429 (quota used up) switches to the fallback immediately, since waiting does not help. Any other error (bad key, bad request) is raised right away, so real problems are not hidden.

## Known Limitations

- **Implicit, need-based queries are hard for naive cosine similarity.** A question like "leaks on plastic pipes over long distances" describes a need, while the catalogue describes features in different words ("designed specifically to find leaks on plastic and large diameter pipes over long distances" on the AQUASCAN TM3, "Frequency Shifting" on the AQUASCOPE 550). The embedding model can rank a loosely related product above the right one. The keyword boost helps when the query shares words with the right chunk, but it is a heuristic, and it does not help when the vocabulary genuinely differs. Query rewriting or a stronger embedding model would be the next step.
- **"Which products have X" questions depend on `k`.** Top-k does not guarantee full recall: if more than `k` products match, some are left out. Raise `TOP_K` for these questions.
- **The keyword boost is hand-tuned.** `KEYWORD_WEIGHT = 0.3` was chosen by testing on this one catalogue, not tuned on a larger set.
- **Sources are self-reported by the model.** They are validated against the retrieved products, but the model decides which of them it "used".
- **Small, clean catalogue only.** One chunk per product works because the document is consistently structured. At larger scale (thousands of SKUs, inconsistent formatting), structure-aware chunking gets harder and pure retrieval quality matters much more. Embeddings would also need to be cached or stored in a vector database instead of rebuilt at every startup.
- **Needs the internet and an API key** for answer generation, and the Gemini free tier has daily request limits per model.

## Example Usage

```
> What is the Minimum Level Profiling feature and which product has it?
Agent: The Minimum Level Profiling feature latches to the lowest noise detected at each
sounding and records leak values independent of passing traffic and other ambient noises.
It is a feature of the AQUASCOPE 3.
[Source: AQUASCOPE 3]

> Can I order the ZONESCAN HYDRO today?
Agent: According to the document, the ZONESCAN HYDRO is listed as "Launching Q2 2026",
so it is not described as available to order yet. The document says to contact sales for
pre-series units.
[Source: ZONESCAN HYDRO]
```

Exact wording varies from run to run.

## Tech Stack

- **Python:** developed and run on 3.14 (macOS)
- **Embeddings:** `sentence-transformers`, model `all-MiniLM-L6-v2` (runs locally)
- **LLM:** Google Gemini API through the `google-genai` library (`gemini-3.8-flash`, with `gemini-3.7-flash` as the fallback)
- **Other dependencies:** `numpy` (similarity maths), `python-dotenv` (loads the API key from `.env`)

## Project Structure

```
qa_agent.py          # the whole agent
product_overview.md  # knowledge base
requirements.txt
.env.example         # template for the API key (copy to .env)
.gitignore
README.md
```
