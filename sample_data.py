"""
Thesis Data Sampling Script
Graph RAG vs Vector RAG — Fida Ali Baig

Reads your already-downloaded local JSON files and creates
a small starter sample for testing both pipelines.

Put this script in the same folder as your JSON files and run:
    python sample_data.py
"""

import json
import random
from pathlib import Path

random.seed(42)

# ── Folder structure ──────────────────────────────────────────────────────────
for d in ["data/cuad/processed", "data/musique/processed",
          "pipeline_a", "pipeline_b", "evaluation", "results"]:
    Path(d).mkdir(parents=True, exist_ok=True)
print("Folder structure ready.")


# ── CUAD ──────────────────────────────────────────────────────────────────────
print("\nSampling CUAD...")

with open("cuad_train.json", "r") as f:
    cuad_raw = [json.loads(line) for line in f if line.strip()]

print(f"  Full CUAD train size : {len(cuad_raw)} rows")

# Keep only rows that have a real answer
cuad_qa = []
for row in cuad_raw:
    answers = row.get("answers", {})
    texts   = answers.get("text", []) if isinstance(answers, dict) else []
    answer  = texts[0].strip() if texts else ""
    if not answer:
        continue
    cuad_qa.append({
        "id":             row.get("id", ""),
        "contract_title": row.get("title", ""),
        "question":       row.get("question", ""),
        "context":        row.get("context", ""),
        "answer":         answer,
        "type":           "factual_or_relational",
    })

print(f"  Rows with real answers : {len(cuad_qa)}")

# Sample 20 unique contracts, then 50 QA pairs from them
all_titles     = list(set(r["contract_title"] for r in cuad_qa))
sampled_titles = random.sample(all_titles, min(20, len(all_titles)))
cuad_filtered  = [r for r in cuad_qa if r["contract_title"] in sampled_titles]
cuad_sample    = random.sample(cuad_filtered, min(50, len(cuad_filtered)))

with open("data/cuad/processed/cuad_sample.json", "w") as f:
    json.dump(cuad_sample, f, indent=2)

print(f"  Contracts sampled : {len(sampled_titles)}")
print(f"  QA pairs sampled  : {len(cuad_sample)}")
print("  Saved → data/cuad/processed/cuad_sample.json")


# ── MuSiQue ───────────────────────────────────────────────────────────────────
print("\nSampling MuSiQue...")

with open("musique_train.json", "r") as f:
    musique_raw = [json.loads(line) for line in f if line.strip()]

print(f"  Full MuSiQue train size : {len(musique_raw)} rows")

hop2, hop3, hop4 = [], [], []
for row in musique_raw:
    n = int(row.get("id", "2hop").split("hop")[0])

    entry = {
        "id":         row.get("id", ""),
        "question":   row.get("question", ""),
        "answer":     row.get("answer", ""),
        "paragraphs": row.get("paragraphs", []),
        "n_hops":     n,
        "type":       f"multi_hop_{n}",
    }
    if n == 2:   hop2.append(entry)
    elif n == 3: hop3.append(entry)
    else:        hop4.append(entry)

print(f"  2-hop available : {len(hop2)}")
print(f"  3-hop available : {len(hop3)}")
print(f"  4-hop available : {len(hop4)}")

s2 = random.sample(hop2, min(17, len(hop2)))
s3 = random.sample(hop3, min(17, len(hop3)))
s4 = random.sample(hop4, min(16, len(hop4)))
musique_sample = s2 + s3 + s4

with open("data/musique/processed/musique_sample.json", "w") as f:
    json.dump(musique_sample, f, indent=2)

print(f"  Sampled — 2-hop : {len(s2)}  |  3-hop : {len(s3)}  |  4-hop : {len(s4)}")
print(f"  Total sampled   : {len(musique_sample)}")
print("  Saved → data/musique/processed/musique_sample.json")


# ── Quick sanity check ────────────────────────────────────────────────────────
print("\nSanity check — first CUAD sample:")
c = cuad_sample[0]
print(f"  Contract : {c['contract_title']}")
print(f"  Question : {c['question'][:80]}")
print(f"  Answer   : {c['answer'][:80]}")

print("\nSanity check — first MuSiQue sample:")
m = musique_sample[0]
print(f"  Question : {m['question'][:80]}")
print(f"  Answer   : {m['answer']}")
print(f"  Hops     : {m['n_hops']}")


# ── Summary ───────────────────────────────────────────────────────────────────
total = len(cuad_sample) + len(musique_sample)
print("\n" + "="*50)
print("DONE")
print("="*50)
print(f"  CUAD QA pairs    : {len(cuad_sample)}")
print(f"  MuSiQue QA pairs : {len(musique_sample)}")
print(f"  Total QA pairs   : {total}")
print("\nNext step: build Pipeline A (Vector RAG) on this data.")
print("="*50)