"""Pipeline A — Step 3: retrieve top-5 + generate answers with GPT-4o.

Writes results/pipeline_a_cuad.json and results/pipeline_a_musique.json, each
record carrying: predicted answer, gold answer, the ranked retrieved chunks with
a per-chunk relevance flag, latency, and token usage — ready for evaluation/.

    python pipeline_a/retrieve_generate.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))

from config import (CUAD_SAMPLE, MUSIQUE_CHUNKS, INDEX_DIR, RESULTS_DIR,
                    TOP_K, GEN_MODEL, TEMPERATURE, EMBED_MODEL, RETRIEVAL_SCOPE,
                    PROMPT_TEMPLATE, safe_dir)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


def get_embeddings():
    return OpenAIEmbeddings(model=EMBED_MODEL)       # must match the index build


def get_llm():
    return ChatOpenAI(model=GEN_MODEL, temperature=TEMPERATURE)


def load_index(path, emb):
    return FAISS.load_local(str(path), emb, allow_dangerous_deserialization=True)


def generate(llm, question, context):
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    msg = llm.invoke(prompt)
    u = getattr(msg, "usage_metadata", None) or {}
    return msg.content.strip(), {"input_tokens":  u.get("input_tokens", 0),
                                 "output_tokens": u.get("output_tokens", 0)}


def _context(docs):
    return "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))


# ── CUAD ──────────────────────────────────────────────────────────────────────
def run_cuad(emb, llm):
    rows = json.load(open(CUAD_SAMPLE))
    base = INDEX_DIR / "cuad"
    cache = {}                              # contract title -> loaded index
    results = []
    for r in rows:
        q, gold, title = r["question"], r["answer"], r["contract_title"]
        if title not in cache:
            cache[title] = load_index(base / safe_dir(title), emb)
        t0   = time.perf_counter()
        docs = cache[title].similarity_search(q, k=TOP_K)   # scoped to this contract
        pred, usage = generate(llm, q, _context(docs))
        latency = (time.perf_counter() - t0) * 1000
        # CUAD has no chunk-level gold -> proxy: chunk contains the gold answer span
        retrieved = [{
            "rank": i + 1,
            "chunk_id": d.metadata.get("chunk_id"),
            "doc_id":   d.metadata.get("doc_id"),
            "relevant": gold.lower() in d.page_content.lower(),
        } for i, d in enumerate(docs)]
        results.append({
            "id": r["id"], "dataset": "cuad", "question": q,
            "gold_answer": gold, "predicted_answer": pred,
            "retrieved": retrieved, "latency_ms": round(latency, 1),
            "tokens": usage, "scope": "per_contract",
        })
    return results


# ── MuSiQue ───────────────────────────────────────────────────────────────────
def run_musique(emb, llm):
    data = json.load(open(MUSIQUE_CHUNKS))
    glob = load_index(INDEX_DIR / "musique" / "_global", emb) \
        if RETRIEVAL_SCOPE == "global" else None
    results = []
    for qid, q in data.items():
        question, gold = q["question"], q["answer"]
        index = glob if glob is not None else load_index(INDEX_DIR / "musique" / qid, emb)
        t0   = time.perf_counter()
        docs = index.similarity_search(question, k=TOP_K)
        pred, usage = generate(llm, question, _context(docs))
        latency = (time.perf_counter() - t0) * 1000
        retrieved = [{
            "rank": i + 1,
            "chunk_id": d.metadata.get("chunk_id"),
            "title":    d.metadata.get("title"),
            # in global scope a hit may come from another question -> not relevant
            "relevant": bool(d.metadata.get("is_supporting"))
                        and (glob is None or d.metadata.get("question_id") == qid),
        } for i, d in enumerate(docs)]
        results.append({
            "id": qid, "dataset": "musique", "n_hops": q["n_hops"],
            "question": question, "gold_answer": gold, "predicted_answer": pred,
            "retrieved": retrieved, "latency_ms": round(latency, 1),
            "tokens": usage, "scope": RETRIEVAL_SCOPE,
        })
    return results


# ── retrieval-quality summary ─────────────────────────────────────────────────
def summarize(name, results):
    n = len(results)
    hit = sum(1 for r in results if any(c["relevant"] for c in r["retrieved"]))
    prec = sum(sum(c["relevant"] for c in r["retrieved"]) for r in results) / (n * TOP_K)
    lat = sorted(r["latency_ms"] for r in results)[n // 2]
    toks = sum(r["tokens"].get("input_tokens", 0) + r["tokens"].get("output_tokens", 0)
               for r in results)
    print(f"  {name:8} | n={n:3} | hit@{TOP_K}={hit/n:5.1%} | "
          f"precision@{TOP_K}={prec:5.1%} | median {lat:.0f} ms | {toks:,} tokens")


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Retrieve + generate — GPT-4o, scope={RETRIEVAL_SCOPE}\n")
    emb, llm = get_embeddings(), get_llm()

    cuad = run_cuad(emb, llm)
    json.dump(cuad, open(RESULTS_DIR / "pipeline_a_cuad.json", "w"), indent=2)

    musique = run_musique(emb, llm)
    json.dump(musique, open(RESULTS_DIR / "pipeline_a_musique.json", "w"), indent=2)

    print("Retrieval quality:")
    summarize("CUAD", cuad)
    summarize("MuSiQue", musique)
    print(f"\nResults -> {RESULTS_DIR}/pipeline_a_cuad.json, pipeline_a_musique.json")
