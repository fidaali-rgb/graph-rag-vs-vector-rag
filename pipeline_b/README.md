# Pipeline B — Graph RAG

LLM triple extraction → Neo4j knowledge graph → entity lookup + 2-hop traversal → GPT-4o.

Uses the **same prompt, generation model (GPT-4o), TOP_K, and metrics** as Pipeline A
(imported from `pipeline_a/config.py`), so the only variable is the retrieval method.

## Run order

```bash
python pipeline_b/extract.py            # 1. extract triples (gpt-4o-mini, no Neo4j)
#    --- start Neo4j (see below) ---
python pipeline_b/build_graph.py        # 2. load triples into Neo4j
python pipeline_b/retrieve_generate.py  # 3. traverse subgraph + generate answers
python evaluation/metrics.py pipeline_b # 4. score (same metrics as Pipeline A)
```

## Neo4j via Docker

```bash
# password is read from NEO4J_PASSWORD in .env
docker run -d --name thesis-neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/<your-password> neo4j:5
```
Then ensure `.env` has `NEO4J_URI=bolt://localhost:7687` and `NEO4J_USER=neo4j`.
Browser UI: http://localhost:7474

## Design notes

- **Scope matches Pipeline A:** entities are keyed by `(name, scope, dataset)`;
  CUAD scope = contract title, MuSiQue scope = question id. Traversal is confined
  to the question's scope — per-contract / per-question, exactly like Pipeline A.
- **Provenance on every triple** (`source_id`, `is_supporting`) lets retrieval be
  scored with the same relevance proxy as Pipeline A (MuSiQue: source paragraph
  `is_supporting`; CUAD: source chunk contains the gold span).
- **Retrieved units = top-5 source chunks** ranked by how many traversed triples
  came from them, so Precision@5 / MRR are directly comparable across pipelines.
- **Subgraph serialization** (the open design point in plan §13): triples are
  rendered as `subject — predicate — object` lines, capped at `MAX_TRIPLES_CONTEXT`.

## Knobs (`config.py`)

- `EXTRACT_MODEL` — `gpt-4o-mini` (build-time extractor; generation stays GPT-4o)
- `MAX_HOPS` — 2 · `MAX_TRIPLES_CONTEXT` — 40 · `EXTRACT_WORKERS` — 8
