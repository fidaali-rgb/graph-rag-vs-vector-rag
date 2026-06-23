"""Pipeline B — Step 2: load triples into Neo4j.  Needs a running Neo4j instance.

Entities are MERGEd per (name, scope, dataset) so each contract / question is its
own subgraph. Before loading, entity names are canonicalized via embedding-based
alias resolution *within each scope* (e.g. "Puerto Rico" and "Commonwealth of
Puerto Rico" merge), so multi-hop bridge entities connect instead of fragmenting.
Relationships carry the predicate plus provenance (source_id, is_supporting).

    python pipeline_b/build_graph.py
"""
import sys, os, json
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from openai import OpenAI
from config import (CUAD_TRIPLES, MUSIQUE_TRIPLES, EMBED_MODEL, ENTITY_SIM_THRESHOLD,
                    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
from neo4j import GraphDatabase

_client = OpenAI()


def _embed(names):
    """Return {name: unit-normalized vector} for unique names (batched)."""
    uniq = sorted(set(names))
    vecs = {}
    for i in range(0, len(uniq), 1000):
        batch = uniq[i:i + 1000]
        resp = _client.embeddings.create(model=EMBED_MODEL, input=batch)
        for name, d in zip(batch, resp.data):
            v = np.asarray(d.embedding, dtype=np.float32)
            vecs[name] = v / (np.linalg.norm(v) or 1.0)
    return vecs


def resolve_entities(triples):
    """Merge alias entity names within each scope by embedding cosine similarity.
    Returns new triples with subject/object replaced by the canonical name."""
    names = [t["subject"] for t in triples] + [t["object"] for t in triples]
    vecs = _embed(names)

    # per-scope frequency to pick the canonical (most frequent) surface form
    by_scope = {}
    for t in triples:
        by_scope.setdefault(t["scope"], Counter())
        by_scope[t["scope"]][t["subject"]] += 1
        by_scope[t["scope"]][t["object"]] += 1

    canon = {}                                   # (scope, name) -> canonical name
    merges = 0
    for scope, freq in by_scope.items():
        clusters = []                            # list of (centroid, canonical_name)
        for name, _ in freq.most_common():       # high-frequency names seed clusters
            v = vecs[name]
            best_i, best_sim = -1, ENTITY_SIM_THRESHOLD
            for i, (c, _cn) in enumerate(clusters):
                sim = float(np.dot(v, c))
                if sim >= best_sim:
                    best_i, best_sim = i, sim
            if best_i >= 0:
                canon[(scope, name)] = clusters[best_i][1]
                merges += 1
            else:
                clusters.append((v, name))
                canon[(scope, name)] = name

    out = []
    for t in triples:
        t = dict(t)
        t["subject"] = canon[(t["scope"], t["subject"])]
        t["object"]  = canon[(t["scope"], t["object"])]
        if t["subject"] != t["object"]:          # drop self-loops created by merging
            out.append(t)
    print(f"  entity resolution: {merges} aliases merged, "
          f"{len(triples)}->{len(out)} triples (scope={ENTITY_SIM_THRESHOLD} cos)")
    return out

LOAD = """
UNWIND $rows AS row
MERGE (a:Entity {name: row.subject, scope: row.scope, dataset: $ds})
MERGE (b:Entity {name: row.object,  scope: row.scope, dataset: $ds})
CREATE (a)-[:REL {predicate: row.relation, source_id: row.source_id,
                  scope: row.scope, dataset: $ds,
                  is_supporting: coalesce(row.is_supporting, false)}]->(b)
"""


def load(driver, triples, dataset, batch=1000):
    with driver.session() as s:
        for i in range(0, len(triples), batch):
            s.run(LOAD, rows=triples[i:i + batch], ds=dataset)


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    with driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")                      # clean slate
        s.run("CREATE INDEX entity_lookup IF NOT EXISTS "
              "FOR (e:Entity) ON (e.scope, e.name)")

    print("CUAD:")
    ct = resolve_entities(json.load(open(CUAD_TRIPLES)));    load(driver, ct, "cuad")
    print("MuSiQue:")
    mt = resolve_entities(json.load(open(MUSIQUE_TRIPLES))); load(driver, mt, "musique")

    with driver.session() as s:
        n = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        r = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    print(f"Neo4j loaded: {n} entities, {r} relations "
          f"(CUAD {len(ct)} + MuSiQue {len(mt)} triples)")
    driver.close()


if __name__ == "__main__":
    main()
