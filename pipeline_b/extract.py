"""Pipeline B — Step 1: extract knowledge-graph triples.  No Neo4j needed.

gpt-4o-mini reads each source unit (CUAD chunk / MuSiQue paragraph) and returns
(subject, relation, object) triples. Each triple keeps provenance (scope + source
chunk id, plus is_supporting for MuSiQue) so the graph stays scoped per-contract /
per-question and retrieval relevance can be scored exactly like Pipeline A.

    python pipeline_b/extract.py
"""
import sys, os, json
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(__file__))

from config import (CUAD_CHUNKS, MUSIQUE_CHUNKS, CUAD_TRIPLES, MUSIQUE_TRIPLES,
                    EXTRACT_MODEL, EXTRACT_WORKERS)
from openai import OpenAI

client = OpenAI()

SYS = (
    "You extract a knowledge graph from text. Return strict JSON of the form "
    '{"triples": [[subject, relation, object], ...]}. '
    "Entities are concise noun phrases; the relation is a short verb phrase. "
    "Extract only facts explicitly stated in the text. If there are none, return "
    'an empty list.'
)


def extract_one(text):
    if not text or not text.strip():
        return []
    try:
        r = client.chat.completions.create(
            model=EXTRACT_MODEL, temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYS},
                      {"role": "user", "content": text[:6000]}])
        data = json.loads(r.choices[0].message.content)
        out = []
        for t in data.get("triples", []):
            if isinstance(t, list) and len(t) == 3 and all(
                    isinstance(x, str) and x.strip() for x in t):
                out.append([t[0].strip(), t[1].strip(), t[2].strip()])
        return out
    except Exception:
        return []


def run(units, label):
    results = [None] * len(units)
    with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as ex:
        futs = {ex.submit(extract_one, u["text"]): i for i, u in enumerate(units)}
        done = 0
        for f in as_completed(futs):
            results[futs[f]] = f.result()
            done += 1
            if done % 100 == 0:
                print(f"  {label}: {done}/{len(units)}")
    triples = []
    for u, trs in zip(units, results):
        for s, p, o in trs:
            row = {"subject": s, "relation": p, "object": o,
                   "scope": u["scope"], "source_id": u["source_id"]}
            if "is_supporting" in u:
                row["is_supporting"] = u["is_supporting"]
            triples.append(row)
    return triples


def cuad_units():
    chunks = json.load(open(CUAD_CHUNKS))
    return [{"text": c["text"], "scope": c["doc_id"], "source_id": c["chunk_id"]}
            for c in chunks]


def musique_units():
    data, units = json.load(open(MUSIQUE_CHUNKS)), []
    for qid, q in data.items():
        for c in q["chunks"]:
            units.append({"text": c["text"], "scope": qid, "source_id": c["chunk_id"],
                          "is_supporting": c["is_supporting"]})
    return units


if __name__ == "__main__":
    print(f"Extracting triples with {EXTRACT_MODEL} ({EXTRACT_WORKERS} workers)…")
    cu = cuad_units();  print(f"CUAD units    : {len(cu)}")
    ct = run(cu, "CUAD")
    json.dump(ct, open(CUAD_TRIPLES, "w"), indent=2)
    print(f"CUAD triples  : {len(ct)} -> {CUAD_TRIPLES.name}")

    mu = musique_units(); print(f"MuSiQue units : {len(mu)}")
    mt = run(mu, "MuSiQue")
    json.dump(mt, open(MUSIQUE_TRIPLES, "w"), indent=2)
    print(f"MuSiQue triples: {len(mt)} -> {MUSIQUE_TRIPLES.name}")
    print("Done. Next: start Neo4j, then python pipeline_b/build_graph.py")
