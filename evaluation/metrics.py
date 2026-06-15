"""Score a pipeline's result files.

Computes answer-quality metrics (SQuAD-style Exact Match + token-F1, plus
abstention rate) and retrieval-quality metrics (hit@k, Precision@k, MRR) from
results/<pipeline>_<dataset>.json. MuSiQue is also broken out by hop count.

    python evaluation/metrics.py                 # scores pipeline_a by default
    python evaluation/metrics.py pipeline_a      # explicit
"""
import sys, json, re, string
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


# ── answer normalization (SQuAD) ──────────────────────────────────────────────
def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    num = sum(common.values())
    if num == 0:
        return 0.0
    prec, rec = num / len(p), num / len(g)
    return 2 * prec * rec / (prec + rec)


def is_abstention(pred: str) -> bool:
    return normalize(pred) == "i dont know"


# ── retrieval metrics from per-chunk relevance flags ──────────────────────────
def retrieval_stats(retrieved):
    rel = [bool(c.get("relevant")) for c in retrieved]
    k = len(rel) or 1
    hit = int(any(rel))
    precision = sum(rel) / k
    mrr = next((1 / (i + 1) for i, r in enumerate(rel) if r), 0.0)
    return hit, precision, mrr


# ── aggregation ───────────────────────────────────────────────────────────────
def score(records):
    n = len(records)
    em = sum(exact_match(r["predicted_answer"], r["gold_answer"]) for r in records) / n
    f1 = sum(token_f1(r["predicted_answer"], r["gold_answer"]) for r in records) / n
    ab = sum(is_abstention(r["predicted_answer"]) for r in records) / n
    hits = [retrieval_stats(r["retrieved"]) for r in records]
    hit = sum(h for h, _, _ in hits) / n
    prec = sum(p for _, p, _ in hits) / n
    mrr = sum(m for _, _, m in hits) / n
    lat = sorted(r["latency_ms"] for r in records)[n // 2]
    toks = sum(r["tokens"].get("input_tokens", 0) + r["tokens"].get("output_tokens", 0)
               for r in records) / n
    return {"n": n, "EM": em, "F1": f1, "abstain": ab, "hit@5": hit,
            "P@5": prec, "MRR": mrr, "median_ms": lat, "avg_tokens": toks}


def fmt(name, s):
    return (f"  {name:14} | n={s['n']:>3} | EM={s['EM']:5.1%} | F1={s['F1']:5.1%} | "
            f"abstain={s['abstain']:5.1%} | hit@5={s['hit@5']:5.1%} | "
            f"P@5={s['P@5']:5.1%} | MRR={s['MRR']:.3f} | "
            f"{s['median_ms']:.0f} ms | {s['avg_tokens']:.0f} tok")


def main(pipeline="pipeline_a"):
    out = {}
    print(f"Scoring {pipeline}\n" + "-" * 110)
    for dataset in ("cuad", "musique"):
        path = RESULTS / f"{pipeline}_{dataset}.json"
        if not path.exists():
            print(f"  {dataset}: (no results file)")
            continue
        recs = json.load(open(path))
        s = score(recs)
        out[dataset] = s
        print(fmt(dataset.upper(), s))
        if dataset == "musique":                     # break out by hop count
            for h in sorted({r["n_hops"] for r in recs}):
                sub = [r for r in recs if r["n_hops"] == h]
                out[f"musique_{h}hop"] = score(sub)
                print(fmt(f"  {h}-hop", out[f"musique_{h}hop"]))
    print("-" * 110)
    json.dump(out, open(RESULTS / f"metrics_{pipeline}.json", "w"), indent=2)
    print(f"Saved -> results/metrics_{pipeline}.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pipeline_a")
