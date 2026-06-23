"""Pipeline B (Graph RAG) — configuration.

Reuses pipeline_a/config.py as the single source of truth for the shared prompt,
generation model, temperature, TOP_K and data paths, so the comparison is fair by
construction. Adds Graph-RAG-specific settings (extractor model, hops, Neo4j).
"""
import os
import importlib.util
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── reuse Pipeline A config (identical prompt / models / paths) ────────────────
_pa_path = Path(__file__).resolve().parent.parent / "pipeline_a" / "config.py"
_spec = importlib.util.spec_from_file_location("pa_config", _pa_path)
pa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pa)

PROMPT_TEMPLATE = pa.PROMPT_TEMPLATE          # IDENTICAL to Pipeline A
GEN_MODEL       = pa.GEN_MODEL                # gpt-4o
TEMPERATURE     = pa.TEMPERATURE              # 0.0
TOP_K           = pa.TOP_K                    # 5
EMBED_MODEL     = pa.EMBED_MODEL              # text-embedding-3-small (entity resolution)
RESULTS_DIR     = pa.RESULTS_DIR
DATA            = pa.DATA
CUAD_SAMPLE     = pa.CUAD_SAMPLE
CUAD_CHUNKS     = pa.CUAD_CHUNKS
MUSIQUE_CHUNKS  = pa.MUSIQUE_CHUNKS

# ── Pipeline B specifics ──────────────────────────────────────────────────────
EXTRACT_MODEL         = "gpt-4o-mini"         # build-time triple extractor (cheap)
MAX_HOPS              = 2                      # subgraph traversal depth
MAX_TRIPLES_CONTEXT   = 40                     # (legacy) cap if serializing triples
EXTRACT_WORKERS       = 8                      # concurrent extraction calls
ENTITY_SIM_THRESHOLD  = 0.85                   # merge entities above this cosine (alias resolution)

CUAD_TRIPLES    = DATA / "cuad"    / "processed" / "cuad_triples.json"
MUSIQUE_TRIPLES = DATA / "musique" / "processed" / "musique_triples.json"

# ── Neo4j ─────────────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
