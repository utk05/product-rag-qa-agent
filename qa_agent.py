"""
qa_agent.py - CLI RAG Q&A agent over a product catalogue (markdown).

Pipeline:  load_and_chunk -> embed_chunks -> (per query) retrieve -> generate_answer

Run:  put GEMINI_API_KEY=... in a .env file, then   python qa_agent.py
"""

import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()  # reads GEMINI_API_KEY from a local .env file (if present)

# Hide the Gemini SDK's harmless "automatic function calling" notice on every request.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

# --------------------------------------------------------------------------
# Settings - the things you'd most likely want to change live
# --------------------------------------------------------------------------
DOC_PATH = "product_overview.md"
TOP_K = 3                            # chunks passed to the LLM. Raise to 4-5 for "which products..." questions
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-3.8-flash"       # any Gemini model name from Google AI Studio works here
FALLBACK_MODEL = "gemini-3.7-flash" # tried if LLM_MODEL stays overloaded (503) or its quota is used up (429); None = off
RETRIES_PER_MODEL = 3                # tries per model on a 503 before moving on (waits 2s, then 4s)
KEYWORD_WEIGHT = 0.3                 # 0 = pure cosine similarity; higher = exact words matter more
SHOW_RETRIEVAL = False               # True = print retrieved chunks + scores (handy for debugging)

REFUSAL = "The document doesn't contain enough information to answer that"

SYSTEM_PROMPT = f"""You answer questions about a product catalogue using ONLY the context provided.

Rules:
1. Use only facts stated in the context. Do not use outside knowledge about these products.
2. If the context does not contain the answer, reply exactly: "{REFUSAL}." Do not guess.
3. Every fact belongs to the product whose block it appears in. Never attribute a feature of one
   product to another. If a product's block does not mention a feature, that product does not
   have it according to the document.
4. Pay attention to the "Status" line. If a product is not yet launched, say it is not
   available to order today and repeat what the document says about how to get it.
5. "Category notes" apply to every product in that category.
6. For recommendations, pick from the products in the context and explain which stated features
   match the user's need. Be concise.
7. End your reply with one final line in exactly this form: SOURCES: <product names you used>
   List only the products whose information you actually used, separated by commas.
   If you used none (for example you gave the "not enough information" reply), write: SOURCES: none"""

# A bullet mentioning any of these is treated as an availability/status line.
STATUS_RE = re.compile(r"launching|coming soon|discontinued|end of life", re.I)

# The catalogue was converted from PDF and lost some "ff" ligatures; fix the ones we found.
OCR_FIXES = {"Ofline": "Offline", "ofering": "offering", "trafic": "traffic"}

# Words ignored when matching query words against chunk text.
STOPWORDS = {
    "the", "and", "for", "are", "what", "which", "who", "how", "does", "can", "has", "have",
    "with", "that", "this", "from", "use", "uses", "any", "all", "our", "you", "your", "need",
    "want", "recommend", "tell", "about", "its", "was", "will", "would", "should", "there",
    "their", "they", "them", "than", "then", "into", "over", "also", "difference", "between",
}

_embedder = None  # loaded once, shared by embed_chunks() and retrieve()


def get_embedder():
    """Load the sentence-transformers model on first use."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


# --------------------------------------------------------------------------
# 1. Ingestion: parse + chunk by structure
# --------------------------------------------------------------------------
def load_and_chunk(filepath):
    """
    Parse the markdown and return one chunk dict per product (### heading).

    Each chunk: product_name, category, status, category_notes, raw_text, embed_text.
    Chunk boundaries come from the markdown structure, never from a character count.
    """
    text = Path(filepath).read_text(encoding="utf-8")

    # Strip converter noise: HTML comments and image links
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    for wrong, right in OCR_FIXES.items():
        text = re.sub(rf"\b{wrong}\b", right, text)

    chunks = []
    category = None
    category_lines = []   # text sitting under a ## heading before its first ### product
    product = None        # {"name": str, "desc": [str], "bullets": [str]}

    def flush():
        """Turn the product being built into a chunk and store it."""
        nonlocal product
        if product is None:
            return
        raw_lines = product["desc"] + ["- " + b for b in product["bullets"]]
        raw_text = "\n".join(raw_lines)
        status = next((b for b in product["bullets"] if STATUS_RE.search(b)), None)
        notes = "\n".join(category_lines)
        chunks.append({
            "product_name": product["name"],
            "category": category,
            "status": status,
            "category_notes": notes,
            "raw_text": raw_text,
            # What we actually embed: name + category + shared category notes + product text
            "embed_text": f"{product['name']}. Category: {category}. {notes}\n{raw_text}",
        })
        product = None

    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()   # also collapses stray tabs
        if not line:
            continue
        is_bullet = line.startswith("•")
        if is_bullet:
            line = line.lstrip("• ").strip()

        if line.startswith("## "):
            flush()
            category, category_lines = line[3:].strip(), []
        elif line.startswith("### "):
            flush()
            product = {"name": line[4:].strip(), "desc": [], "bullets": []}
        elif line.startswith("# "):
            continue  # document title
        elif product is None:
            # Between a ## heading and its first ### product: shared category info.
            if category:
                category_lines.append(("- " if is_bullet else "") + line)
        elif is_bullet:
            product["bullets"].append(line)
        elif product["bullets"]:
            # Plain line after a product's bullets = a category heading the PDF
            # conversion left unformatted (e.g. "Acoustic Leak Detection Microphones").
            flush()
            category, category_lines = line, []
        else:
            product["desc"].append(line)      # the one-line description

    flush()
    return chunks


# --------------------------------------------------------------------------
# 2. Embedding
# --------------------------------------------------------------------------
def embed_chunks(chunks):
    """Attach a normalized embedding vector to every chunk (kept in memory)."""
    model = get_embedder()
    vectors = model.encode([c["embed_text"] for c in chunks], normalize_embeddings=True)
    for chunk, vec in zip(chunks, vectors):
        chunk["embedding"] = vec
    return chunks


# --------------------------------------------------------------------------
# 3. Retrieval
# --------------------------------------------------------------------------
def _words(text):
    """Lowercase words, with a crude plural strip so 'pipes' matches 'pipe'."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words}


def keyword_overlap(query, chunk):
    """Fraction (0-1) of the query's meaningful words that appear in the chunk."""
    q_words = {w for w in _words(query) if w not in STOPWORDS and len(w) > 2}
    if not q_words:
        return 0.0
    return len(q_words & _words(chunk["embed_text"])) / len(q_words)


def retrieve(query, chunks, k=TOP_K):
    """
    Return the top-k chunks for the query (each with a 'score').

    Score = cosine similarity + KEYWORD_WEIGHT * keyword overlap.
    Cosine similarity captures meaning; the small keyword boost rescues queries that
    name an exact feature or product term (e.g. "Minimum Level Profiling") which a
    small embedding model can rank poorly. Set KEYWORD_WEIGHT = 0 for pure cosine.
    """
    model = get_embedder()
    q = model.encode([query], normalize_embeddings=True)[0]
    matrix = np.vstack([c["embedding"] for c in chunks])
    # Vectors are unit length, so a dot product IS cosine similarity.
    cosine = matrix @ q
    scores = cosine + KEYWORD_WEIGHT * np.array([keyword_overlap(query, c) for c in chunks])
    top = np.argsort(scores)[::-1][:k]
    return [{**chunks[i], "score": float(scores[i])} for i in top]


# --------------------------------------------------------------------------
# 4. Generation
# --------------------------------------------------------------------------
def build_context(retrieved_chunks):
    """Format chunks with their metadata so status/category are not buried in prose."""
    blocks = []
    for i, c in enumerate(retrieved_chunks, 1):
        parts = [f"[Chunk {i}]", f"Product: {c['product_name']}", f"Category: {c['category']}"]
        if c["status"]:
            parts.append(f"Status: {c['status']}")
        if c["category_notes"]:
            parts.append(f"Category notes (apply to all products in this category):\n{c['category_notes']}")
        parts.append(f"Details:\n{c['raw_text']}")
        blocks.append("\n".join(parts))
    return "\n\n---\n\n".join(blocks)


def generate_answer(query, retrieved_chunks):
    """
    Ask the LLM to answer the query using only the retrieved chunks.

    Error handling:
      503 (model overloaded)  -> temporary: wait and retry, then fall back to FALLBACK_MODEL
      429 (quota used up)     -> waiting won't help: go straight to FALLBACK_MODEL
      anything else           -> real problem (bad key, bad request): raise immediately
    """
    from google import genai
    from google.genai import types
    client = genai.Client()  # reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment

    user_msg = f"Context:\n\n{build_context(retrieved_chunks)}\n\nQuestion: {query}"
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0)

    models = [LLM_MODEL] + ([FALLBACK_MODEL] if FALLBACK_MODEL else [])
    last_error = None
    for i, model in enumerate(models):
        for attempt in range(RETRIES_PER_MODEL):
            try:
                response = client.models.generate_content(model=model, contents=user_msg, config=config)
                return (response.text or "").strip()
            except Exception as e:
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    last_error = e
                    break                                  # quota: don't wait, switch model
                if "503" in msg or "UNAVAILABLE" in msg:
                    last_error = e
                    if attempt < RETRIES_PER_MODEL - 1:
                        wait = 2 ** (attempt + 1)          # 2s, 4s, ...
                        print(f"  ({model} is busy, retrying in {wait}s...)")
                        time.sleep(wait)
                    continue
                raise                                       # any other error: stop and report
        if i + 1 < len(models):
            print(f"  ({model} unavailable, switching to {models[i + 1]}...)")

    raise RuntimeError(f"All models unavailable. Last error: {last_error}")


def split_sources(answer, retrieved_chunks):
    """
    Split the model's reply into (answer_text, sources).

    The model ends its reply with a line "SOURCES: A, B". We remove that line and keep only
    names that match a retrieved product, so the model can't cite something we never gave it.
    """
    lines = answer.rstrip().splitlines()
    sources = []
    if lines:
        m = re.match(r"^\W*sources\W*:?\W*(.*)$", lines[-1], re.I)
        if m:
            tail = m.group(1).lower()
            sources = [c["product_name"] for c in retrieved_chunks if c["product_name"].lower() in tail]
            lines = lines[:-1]
    return "\n".join(lines).strip(), sources


# --------------------------------------------------------------------------
# 5. CLI
# --------------------------------------------------------------------------
def main():
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        sys.exit("Add GEMINI_API_KEY to a .env file first (see README).")

    print("Loading catalogue and embedding chunks...")
    chunks = embed_chunks(load_and_chunk(DOC_PATH))
    print(f"Ready: {len(chunks)} product chunks. Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break

        retrieved = retrieve(query, chunks, TOP_K)
        if SHOW_RETRIEVAL:
            for c in retrieved:
                print(f"  (retrieved {c['product_name']}  score={c['score']:.3f})")

        try:
            raw_answer = generate_answer(query, retrieved)
        except Exception as e:
            print(f"Agent: [LLM call failed: {e}]\n")
            continue

        answer, sources = split_sources(raw_answer, retrieved)
        print(f"Agent: {answer}")
        # No source line when the model says the document has no answer.
        if not answer.startswith(REFUSAL):
            if sources:
                print("[Source: " + ", ".join(sources) + "]")
            else:
                # Model didn't name its sources: show what was retrieved, labelled honestly.
                print("[Retrieved: " + ", ".join(c["product_name"] for c in retrieved) + "]")
        print()


if __name__ == "__main__":
    main()