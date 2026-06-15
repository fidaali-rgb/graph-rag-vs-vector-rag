"""Pipeline A — Step 1: chunk documents into retrieval units.  Runs fully offline.

CUAD    : many QA pairs share the same contract, so contracts are de-duplicated,
          then each is recursively split into 512-token / 64-overlap chunks.
MuSiQue : each paragraph is already a natural retrieval unit; it is split further
          only if it exceeds the chunk size. Chunks stay grouped per question so
          per-question retrieval (the default scope) is possible.

    python pipeline_a/chunk.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

from config import (CUAD_SAMPLE, MUSIQUE_SAMPLE, CUAD_CHUNKS, MUSIQUE_CHUNKS,
                    CHUNK_SIZE, CHUNK_OVERLAP, ENCODING_NAME)
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name=ENCODING_NAME,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def chunk_cuad():
    rows = json.load(open(CUAD_SAMPLE))
    contracts = {}                          # title -> full contract text (deduped)
    for r in rows:
        contracts.setdefault(r["contract_title"], r["context"])

    chunks = []
    for title, context in contracts.items():
        for i, piece in enumerate(splitter.split_text(context)):
            chunks.append({
                "chunk_id": f"{title}::chunk_{i}",
                "dataset":  "cuad",
                "doc_id":   title,
                "text":     piece,
            })

    json.dump(chunks, open(CUAD_CHUNKS, "w"), indent=2)
    print(f"CUAD     : {len(contracts):>3} contracts -> {len(chunks):>4} chunks   -> {CUAD_CHUNKS.name}")
    return chunks


def chunk_musique():
    rows = json.load(open(MUSIQUE_SAMPLE))
    out, total = {}, 0
    for r in rows:
        qchunks = []
        for p in r["paragraphs"]:
            text   = p.get("paragraph_text", "") or ""
            pieces = splitter.split_text(text) or [text]
            for j, piece in enumerate(pieces):
                suffix = f"_{j}" if len(pieces) > 1 else ""
                qchunks.append({
                    "chunk_id":      f"{r['id']}::p{p['idx']}{suffix}",
                    "dataset":       "musique",
                    "para_idx":      p["idx"],
                    "title":         p.get("title", ""),
                    "is_supporting": bool(p.get("is_supporting", False)),
                    "text":          piece,
                })
        out[r["id"]] = {
            "question": r["question"],
            "answer":   r["answer"],
            "n_hops":   r["n_hops"],
            "chunks":   qchunks,
        }
        total += len(qchunks)

    json.dump(out, open(MUSIQUE_CHUNKS, "w"), indent=2)
    print(f"MuSiQue  : {len(out):>3} questions -> {total:>4} para-chunks -> {MUSIQUE_CHUNKS.name}")
    return out


if __name__ == "__main__":
    print("Chunking (offline, no API)…")
    chunk_cuad()
    chunk_musique()
    print("Done.")
