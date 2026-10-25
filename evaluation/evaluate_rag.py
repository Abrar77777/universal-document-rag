from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
import uuid

from rag.rag_pipeline import RAGPipeline


def _precision_at_k(retrieved: list[dict], expected_sources: set[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for item in top if item.get("source") in expected_sources)
    return hits / k if k > 0 else 0.0


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


def evaluate(dataset_path: str, index_dirs: list[str], top_k: int, retrieval_mode: str = "hybrid", retrieval_only: bool = False) -> dict:
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    pipeline = RAGPipeline(index_dirs)
    rows = []
    run_id = uuid.uuid4().hex

    for i, item in enumerate(dataset, start=1):
        expected_sources = set(item.get("expected_sources", []))
        result = pipeline.answer(
            question=item["question"],
            session_id=f"eval-{run_id}-{i}",
            use_web=False,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            generate=not retrieval_only,
        )
        retrieved = result.get("retrieved_chunks", [])
        rows.append({
            "question": item["question"],
            "precision_at_k": _precision_at_k(retrieved, expected_sources, top_k),
            "recall_at_k": _recall_at_k(retrieved, expected_sources, top_k),
            "mrr": _mrr(retrieved, expected_sources),
            "grounding_proxy": _answer_grounding(result.get("answer", ""), retrieved) if result["generation_status"] == "generated" else None,
            "generation_status": result["generation_status"],
            "retrieval_ms": result["retrieval_ms"],
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
        })

    grounding = [row["grounding_proxy"] for row in rows if row["grounding_proxy"] is not None]
    aggregate = {
        "retrieval_mode": retrieval_mode,
        "top_k": top_k,
        "generated_answers": len(grounding),
        "questions": len(rows),
        "precision_at_k": round(mean(row["precision_at_k"] for row in rows), 4) if rows else 0,
        "recall_at_k": round(mean(row["recall_at_k"] for row in rows), 4) if rows else 0,
        "mrr": round(mean(row["mrr"] for row in rows), 4) if rows else 0,
        "grounding_proxy": round(mean(grounding), 4) if grounding else None,
    }
    return {"aggregate": aggregate, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval and answer grounding.")
    parser.add_argument("--dataset", required=True, help="JSON file with questions and expected_sources.")
    parser.add_argument("--index-dir", action="append", required=True, help="FAISS index directory. Repeat for multiple.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", default="evaluation/results.json")
    parser.add_argument("--retrieval-mode", choices=["hybrid", "vector"], default="hybrid")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip external LLM calls.")
    args = parser.parse_args()

    result = evaluate(args.dataset, args.index_dir, args.top_k, args.retrieval_mode, args.retrieval_only)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
