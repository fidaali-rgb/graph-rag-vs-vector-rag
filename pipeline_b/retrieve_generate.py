"""Pipeline B — Step 3: graph-guided hybrid retrieval + GPT-4o.

For each question: extract seed entities, match them in the question's scope
(per-contract for CUAD, per-question for MuSiQue), traverse up to 2 hops to find the
connected subgraph, then rank the SOURCE PASSAGES by how many traversed triples came
from them and feed the top-5 passages' full text to GPT-4o (same prompt as Pipeline A).
The graph is the *selector* (which passages are relevant); the LLM reads coherent
text, not fragmented triples. Writes the SAME schema so metrics.py scores both.

    python pipeline_b/retrieve_generate.py
"""
import sys, os, json, time
from collections import OrderedDict
sys.path.insert(0, os.path.dirname(__file__))

from config import (CUAD_SAMPLE, MUSIQUE_CHUNKS, CUAD_CHUNKS, RESULTS_DIR, TOP_K,
                    GEN_MODEL, TEMPERATURE, EXTRACT_MODEL, MAX_HOPS,
                    MAX_TRIPLES_CONTEXT, PROMPT_TEMPLATE,
                    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
from neo4j import GraphDatabase
from openai import OpenAI
from langchain_openai import ChatOpenAI

client = OpenAI()
llm = ChatOpenAI(model=GEN_MODEL, temperature=TEMPERATURE)

TRAVERSE = """
MATCH (e:Entity {scope: $scope})
WHERE any(s IN $seeds WHERE toLower(e.name) CONTAINS toLower(s)
                          OR toLower(s) CONTAINS toLower(e.name))
MATCH path = (e)-[:REL*1..%d]-(:Entity {scope: $scope})
UNWIND relationships(path) AS rel
WITH DISTINCT startNode(rel) AS a, rel, endNode(rel) AS b
RETURN a.name AS s, rel.predicate AS p, b.name AS o,
       rel.source_id AS source_id, rel.is_supporting AS is_supporting
LIMIT 300
""" % MAX_HOPS


def seed_entities(question):
    try:
        r = client.chat.completions.create(
            model=EXTRACT_MODEL, temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content":
                       'Extract the named entities / key noun phrases from the '
                       'question as JSON {"entities": [..]}. Short phrases only.'},
                      {"role": "user", "content": question}])
        return [e for e in json.loads(r.choices[0].message.content).get("entities", [])
                if isinstance(e, str) and e.strip()]
    except Exception:
        return []


def traverse(session, scope, seeds):
    if not seeds:
        return []
    return [dict(r) for r in session.run(TRAVERSE, scope=scope, seeds=seeds)]


def top_sources(triples, k=TOP_K):
    order = OrderedDict()                       # rank source units by triple frequency
    for t in triples:
        sid = t["source_id"]
        if sid not in order:
            order[sid] = {"source_id": sid, "is_supporting": t.get("is_supporting"),
                          "count": 0}
        order[sid]["count"] += 1
    return sorted(order.values(), key=lambda x: -x["count"])[:k]


def generate(question, context):
    msg = llm.invoke(PROMPT_TEMPLATE.format(
        context=context or "(no relevant facts found)", question=question))
    u = getattr(msg, "usage_metadata", None) or {}
    return msg.content.strip(), {"input_tokens":  u.get("input_tokens", 0),
                                 "output_tokens": u.get("output_tokens", 0)}


def _passage_context(srcs, text_map):
    return "\n\n".join(f"[{i+1}] {text_map.get(s['source_id'], '')}"
                       for i, s in enumerate(srcs))


def run_cuad(session):
    rows = json.load(open(CUAD_SAMPLE))
    text_map = {c["chunk_id"]: c["text"] for c in json.load(open(CUAD_CHUNKS))}
    results = []
    for r in rows:
        q, gold, scope = r["question"], r["answer"], r["contract_title"]
        t0 = time.perf_counter()
        seeds = seed_entities(q)
        tris  = traverse(session, scope, seeds)
        srcs  = top_sources(tris)                      # graph selects passages
        pred, usage = generate(q, _passage_context(srcs, text_map))   # LLM reads text
        latency = (time.perf_counter() - t0) * 1000
        retrieved = [{
            "rank": i + 1, "chunk_id": s["source_id"],
            "relevant": gold.lower() in text_map.get(s["source_id"], "").lower(),
        } for i, s in enumerate(srcs)]
        results.append({
            "id": r["id"], "dataset": "cuad", "question": q, "gold_answer": gold,
            "predicted_answer": pred, "retrieved": retrieved,
            "latency_ms": round(latency, 1), "tokens": usage, "scope": "per_contract",
            "n_seeds": len(seeds), "n_triples": len(tris),
        })
    return results


def run_musique(session):
    data = json.load(open(MUSIQUE_CHUNKS))
    text_map = {c["chunk_id"]: c["text"] for q in data.values() for c in q["chunks"]}
    results = []
    for qid, q in data.items():
        question, gold = q["question"], q["answer"]
        t0 = time.perf_counter()
        seeds = seed_entities(question)
        tris  = traverse(session, qid, seeds)
        srcs  = top_sources(tris)                      # graph selects passages
        pred, usage = generate(question, _passage_context(srcs, text_map))
        latency = (time.perf_counter() - t0) * 1000
        retrieved = [{
            "rank": i + 1, "chunk_id": s["source_id"],
            "relevant": bool(s.get("is_supporting")),
        } for i, s in enumerate(srcs)]
        results.append({
            "id": qid, "dataset": "musique", "n_hops": q["n_hops"],
            "question": question, "gold_answer": gold, "predicted_answer": pred,
            "retrieved": retrieved, "latency_ms": round(latency, 1), "tokens": usage,
            "scope": "per_question", "n_seeds": len(seeds), "n_triples": len(tris),
        })
    return results


def _summary(name, res):
    n = len(res)
    hit = sum(1 for r in res if any(c["relevant"] for c in r["retrieved"])) / n
    miss = sum(1 for r in res if not r["retrieved"]) / n        # graph found nothing
    print(f"  {name:8} | n={n:3} | hit@{TOP_K}={hit:5.1%} | empty-subgraph={miss:5.1%}")


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print(f"Pipeline B — Graph RAG ({MAX_HOPS}-hop), generation {GEN_MODEL}\n")
    with driver.session() as s:
        cuad = run_cuad(s)
        json.dump(cuad, open(RESULTS_DIR / "pipeline_b_cuad.json", "w"), indent=2)
        musique = run_musique(s)
        json.dump(musique, open(RESULTS_DIR / "pipeline_b_musique.json", "w"), indent=2)
    driver.close()

    print("Retrieval quality:")
    _summary("CUAD", cuad)
    _summary("MuSiQue", musique)
    print(f"\nResults -> {RESULTS_DIR}/pipeline_b_*.json")
    print("Score with: python evaluation/metrics.py pipeline_b")
