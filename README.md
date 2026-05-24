# Graph RAG vs Vector RAG — A Controlled Empirical Evaluation

**Master's Thesis · Fida Ali Baig · ID: 303182**  
Applied Data Science · University of L'Aquila  
---

## Overview

This repository contains the full implementation, datasets, evaluation scripts, and results for my master's thesis:

> *A Controlled Empirical Evaluation of Graph-Based vs. Vector-Based Retrieval-Augmented Generation for Multi-Hop Enterprise Knowledge Retrieval*

The thesis compares two RAG architectures under identical conditions — same documents, same questions, same LLM — with retrieval method as the sole variable.

| | Pipeline A | Pipeline B |
|---|---|---|
| **Method** | Vector RAG | Graph RAG |
| **Storage** | FAISS (dense embeddings) | Neo4j (knowledge graph) |
| **Retrieval** | Cosine similarity (top-5) | Cypher 2-hop subgraph traversal |
| **Embeddings** | text-embedding-3-small (OpenAI) | LLM-assisted NER + relation extraction |
| **Generation** | GPT-4o · temperature 0.0 | GPT-4o · temperature 0.0 |

---

## Research Questions

- **RQ1** — To what extent does Graph RAG improve answer accuracy (Exact Match, F1) over Vector RAG across factual, relational, and multi-hop question types?
- **RQ2** — Does Graph RAG reduce the rate of hallucinated or unsupported statements?
- **RQ3** — For which question categories does structured retrieval yield the greatest advantage in retrieval quality (Precision@5, MRR)?
- **RQ4** — What are the latency, token-cost, and implementation trade-offs of Graph RAG relative to Vector RAG?

---

## Datasets

### CUAD — Contract Understanding Atticus Dataset
- 510 real commercial legal contracts, 13,000+ expert-labelled QA pairs
- 41 legal clause categories (governing law, termination, IP, liability, etc.)
- Used for: **factual and relational** question types
- Source: [Atticus Project](https://www.atticusprojectai.org/cuad) · [Hugging Face](https://huggingface.co/datasets/theatticusproject/cuad-qa)
- Licence: CC BY 4.0

### MuSiQue — Multihop Questions via Single-hop Question Composition
- 25,000 QA pairs requiring 2–4 reasoning hops across multiple documents
- Specifically designed so single-hop retrieval cannot shortcut the answer
- Used for: **multi-hop** question types
- Source: [StonyBrookNLP/musique](https://github.com/StonyBrookNLP/musique) · Trivedi et al., TACL 2022
- Licence: CC BY 4.0

### Working sample (starter set)
| Dataset | Contracts/Docs | QA pairs | Hop distribution |
|---|---|---|---|
| CUAD | 20 contracts | 50 pairs | factual + relational |
| MuSiQue | — | 50 pairs | 17× 2-hop, 17× 3-hop, 16× 4-hop |
| **Total** | | **100 pairs** | |

---

## Repository Structure

```
thesis-graph-rag/
│
├── data/
│   ├── cuad/
│   │   ├── raw/                  # original downloaded files
│   │   └── processed/
│   │       └── cuad_sample.json  # 50 sampled QA pairs
│   └── musique/
│       ├── raw/
│       └── processed/
│           └── musique_sample.json  # 50 sampled QA pairs
│
├── pipeline_a/                   # Vector RAG implementation
│   ├── chunk.py                  # document chunking
│   ├── embed.py                  # embedding generation + FAISS index
│   └── retrieve_generate.py      # retrieval + GPT-4o generation
│
├── pipeline_b/                   # Graph RAG implementation
│   ├── extract.py                # NER + relation extraction → triples
│   ├── build_graph.py            # load triples into Neo4j
│   └── retrieve_generate.py      # Cypher retrieval + GPT-4o generation
│
├── evaluation/
│   ├── metrics.py                # Exact Match, F1, RAGAS, Precision@5, MRR
│   └── significance.py           # McNemar's test
│
├── results/                      # experiment outputs (gitignored if large)
│
├── sample_data.py                # data download + sampling script
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/thesis-graph-rag.git
cd thesis-graph-rag
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set environment variables
```bash
export OPENAI_API_KEY=your_key_here
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=your_password
```

### 4. Download and sample data
```bash
python sample_data.py
```

---

## Evaluation Metrics

| Metric | What it measures |
|---|---|
| Exact Match | Whether the answer exactly matches the ground truth |
| Token-level F1 | Partial match between predicted and ground truth answer |
| RAGAS Faithfulness | Rate of unsupported or hallucinated claims |
| Precision@5 | Relevance of the top-5 retrieved chunks / subgraphs |
| MRR | Mean Reciprocal Rank of the first relevant result |
| Latency | Median query time in milliseconds |
| Token cost | Average prompt + completion tokens per query |

---

## Timeline

| Phase | Weeks | Description |
|---|---|---|
| 1 — Setup | 1–3 | Environment, datasets, GitHub |
| 2 — Vector RAG | 4–7 | Pipeline A implementation |
| 3 — Graph RAG | 8–13 | Pipeline B implementation |
| 4 — Experiments | 14–17 | Run all 4 combinations |
| 5 — Evaluation | 18–21 | Metrics, significance testing, analysis |
| 6 — Writing | 22–26 | Thesis document |

---

## References

- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS.
- Edge et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*. Microsoft Research.
- Trivedi et al. (2022). *MuSiQue: Multihop Questions via Single-hop Question Composition*. TACL.
- Es et al. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation*. arXiv:2309.15217.

---

## Supervisor

**Prof. Francesco Gullo** · University of L'Aquila  
---

## Licence

Code: MIT · Datasets: CC BY 4.0 (see individual dataset pages)
