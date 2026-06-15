# Pipeline A — Vector RAG

Recursive chunking → FAISS (dense embeddings) → top-5 cosine retrieval → GPT-4o.

Uses OpenAI `text-embedding-3-small` + GPT-4o. Needs `OPENAI_API_KEY` in `.env`.

## Run order

```bash
python pipeline_a/chunk.py              # 1. dedupe + chunk (no API)
python pipeline_a/embed.py              # 2. embed + build FAISS indexes
python pipeline_a/retrieve_generate.py  # 3. retrieve top-5 + generate answers
```

Approximate cost for the 100-pair sample: ~$0.50–1.

## Knobs (`config.py`)

- `RETRIEVAL_SCOPE` — MuSiQue only: `per_question` (default) or `global`.
  Override per run: `RETRIEVAL_SCOPE=global python pipeline_a/embed.py`
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — 512 / 64 tokens.
- `TOP_K` — 5.

## Outputs

- `data/*/processed/*_chunks.json` — chunked corpora
- `pipeline_a/index/` — FAISS indexes (gitignored)
- `results/pipeline_a_{cuad,musique}.json` — per-question answers + ranked
  retrieval (with relevance flags) + latency + token usage, ready for `evaluation/`

**CUAD relevance proxy:** no chunk-level gold exists, so a retrieved chunk counts
as relevant if it contains the gold answer span. Note a long answer span can be
split across a chunk boundary and be missed — a known measurement nuance to refine
in the evaluation phase.
