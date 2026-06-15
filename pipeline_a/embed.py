"""Pipeline A — Step 2: embed chunks and build FAISS indexes.

Uses OpenAI text-embedding-3-small.
CUAD    : one index per contract (clause extraction within a known contract).
MuSiQue : per_question -> one small index per question (default);
          global       -> a single pooled index over all paragraphs.

    python pipeline_a/embed.py
"""
import sys, os, json, shutil
sys.path.insert(0, os.path.dirname(__file__))

from config import (CUAD_CHUNKS, MUSIQUE_CHUNKS, INDEX_DIR,
                    EMBED_MODEL, RETRIEVAL_SCOPE, safe_dir)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


def get_embeddings():
    print(f"  embeddings: OpenAI {EMBED_MODEL}")
    return OpenAIEmbeddings(model=EMBED_MODEL)


def _docs(chunks):
    return [Document(page_content=c["text"],
                     metadata={k: v for k, v in c.items() if k != "text"})
            for c in chunks]


def _reset(path):
    if path.exists():
        shutil.rmtree(path)


def build_cuad(emb):
    chunks = json.load(open(CUAD_CHUNKS))
    base = INDEX_DIR / "cuad"
    _reset(base)
    by_contract = {}
    for c in chunks:
        by_contract.setdefault(c["doc_id"], []).append(c)
    for title, cs in by_contract.items():
        FAISS.from_documents(_docs(cs), emb).save_local(str(base / safe_dir(title)))
    print(f"CUAD index    : {len(by_contract)} per-contract indexes "
          f"({len(chunks)} chunks) -> {base}")


def build_musique(emb):
    data = json.load(open(MUSIQUE_CHUNKS))
    base = INDEX_DIR / "musique"
    _reset(base)

    if RETRIEVAL_SCOPE == "global":
        pooled = []
        for qid, q in data.items():
            for c in q["chunks"]:
                c = dict(c); c["question_id"] = qid
                pooled.append(c)
        FAISS.from_documents(_docs(pooled), emb).save_local(str(base / "_global"))
        print(f"MuSiQue index : {len(pooled)} chunks (1 GLOBAL index) -> {base/'_global'}")
    else:
        for qid, q in data.items():
            FAISS.from_documents(_docs(q["chunks"]), emb).save_local(str(base / qid))
        print(f"MuSiQue index : {len(data)} PER-QUESTION indexes -> {base}")


if __name__ == "__main__":
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Building FAISS indexes (scope={RETRIEVAL_SCOPE})…")
    emb = get_embeddings()
    build_cuad(emb)
    build_musique(emb)
    print("Done.")
