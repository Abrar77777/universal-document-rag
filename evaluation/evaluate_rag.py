from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from rag.rag_pipeline import RAGPipeline


def _precision_at_k(retrieved: list[dict], expected_sources: set[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for item in top if item.get("source") in expected_sources)
    return hits / len(top)


def _recall_at_k(retrieved: list[dict], expected_sources: set[str], k: int) -> float:
    if not expected_sources:
        return 0.0
    found = {item.get("source") for item in retrieved[:k]} & expected_sources
    return len(found) / len(expected_sources)


def _mrr(retrieved: list[dict], expected_sources: set[str]) -> float:
    for rank, item in enumerate(retrieved, start=1):
        if item.get("source") in expected_sources:
            return 1 / rank
    return 0.0


def _answer_grounding(answer: str, retrieved: list[dict]) -> float:
    answer_terms = {word.lower().strip(".,:;!?") for word in answer.split() if len(word) > 4}
    context_terms = set()
    for item in retrieved:
        context_terms.update(
            word.lower().strip(".,:;!?")
            for word in item.get("text", "").split()
            if len(word) > 4
        )
    if not answer_terms:
        return 0.0
    return round(len(answer_terms & context_terms) / len(answer_terms), 4)


def evaluate(dataset_path: str, index_dirs: list[str], top_k: int) -> dict:
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    pipeline = RAGPipeline(index_dirs)
    rows = []

    for i, item in enumerate(dataset, start=1):
        expected_sources = set(item.get("expected_sources", []))
        result = pipeline.answer(
            question=item["question"],
            session_id=f"eval-{i}",
            use_web=False,
            top_k=top_k,
        )
        retrieved = result.get("retrieved_chunks", [])
        rows.append({
            "question": item["question"],
            "precision_at_k": _precision_at_k(retrieved, expected_sources, top_k),
            "recall_at_k": _recall_at_k(retrieved, expected_sources, top_k),
            "mrr": _mrr(retrieved, expected_sources),
            "grounding_proxy": _answer_grounding(result.get("answer", ""), retrieved),
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
        })

    aggregate = {
        "questions": len(rows),
        "precision_at_k": round(mean(row["precision_at_k"] for row in rows), 4) if rows else 0,
        "recall_at_k": round(mean(row["recall_at_k"] for row in rows), 4) if rows else 0,
        "mrr": round(mean(row["mrr"] for row in rows), 4) if rows else 0,
        "grounding_proxy": round(mean(row["grounding_proxy"] for row in rows), 4) if rows else 0,
    }
    return {"aggregate": aggregate, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval and answer grounding.")
    parser.add_argument("--dataset", required=True, help="JSON file with questions and expected_sources.")
    parser.add_argument("--index-dir", action="append", required=True, help="FAISS index directory. Repeat for multiple.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", default="evaluation/results.json")
    args = parser.parse_args()

    result = evaluate(args.dataset, args.index_dir, args.top_k)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
