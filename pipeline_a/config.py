"""Pipeline A (Vector RAG) — shared configuration.

Runs against the preprocessed samples with real OpenAI embeddings
(text-embedding-3-small) and GPT-4o. Requires OPENAI_API_KEY in .env.
"""
import os
import re
import hashlib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def safe_dir(name: str) -> str:
    """Filesystem-safe, collision-free directory name for a contract title."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:80]
    return f"{slug}_{hashlib.md5(name.encode()).hexdigest()[:8]}"

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA        = ROOT / "data"
INDEX_DIR   = ROOT / "pipeline_a" / "index"      # gitignored
RESULTS_DIR = ROOT / "results"

CUAD_SAMPLE    = DATA / "cuad"    / "processed" / "cuad_sample.json"
MUSIQUE_SAMPLE = DATA / "musique" / "processed" / "musique_sample.json"
CUAD_CHUNKS    = DATA / "cuad"    / "processed" / "cuad_chunks.json"
MUSIQUE_CHUNKS = DATA / "musique" / "processed" / "musique_chunks.json"

# ── Chunking (plan §4: 512 tokens, 64 overlap) ────────────────────────────────
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64
ENCODING_NAME = "cl100k_base"      # tiktoken encoder used by GPT-4o / embeddings

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = 5
# MuSiQue only. "per_question": search that question's 20 paragraphs (recommended,
# matches the dataset design). "global": one pooled index over all paragraphs.
RETRIEVAL_SCOPE = os.getenv("RETRIEVAL_SCOPE", "per_question")

# ── Models ────────────────────────────────────────────────────────────────────
EMBED_MODEL    = "text-embedding-3-small"
GEN_MODEL      = "gpt-4o"
TEMPERATURE    = 0.0

# ── Shared prompt (IDENTICAL across Pipeline A and B, and both datasets) ───────
# Handles both extraction-style questions (CUAD: "highlight the parts related to X")
# and short-answer QA (MuSiQue), while staying grounded in the context.
PROMPT_TEMPLATE = (
    "You answer questions using ONLY the provided context. Do not use outside "
    "knowledge.\n"
    "- If the question asks you to identify or highlight part of the text, answer "
    "by quoting the exact relevant span(s) from the context.\n"
    "- Otherwise give the shortest complete answer (a name, date, number, or "
    "phrase).\n"
    "- Reply exactly \"I don't know\" only if the answer genuinely cannot be found "
    "in the context.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)
