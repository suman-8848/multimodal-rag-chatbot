"""Retrieval evaluation for the MR Tips & Tricks RAG pipeline.

Measures precision@k and recall@k against a hand-labeled ground-truth set
(eval_set.json) for both Chroma collections queried by RagPipeline.retrieve().

Text ground truth is labeled at section (optionally + subsection) granularity
-- the finest unit that can be hand-labeled reliably without picking apart
individual overlapping chunks. So text recall@k is a hit-rate: 1 if any
top-k chunk matches the expected section, else 0. Image ground truth is
labeled at the exact image-id level, so image recall@k is the standard
"fraction of relevant images found in top-k".
"""

import json
import statistics
from pathlib import Path

from rag.pipeline import RagPipeline

EVAL_SET_PATH = Path(__file__).resolve().parent / "eval_set.json"

# Matches RagPipeline's default top_k_text=4 / top_k_images=2, so this
# evaluates precision/recall of the system exactly as it's deployed.
TEXT_KS = [1, 2, 3, 4]
IMAGE_KS = [1, 2]


def load_eval_set() -> list[dict]:
    return json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))


def _image_id(image_path: str) -> str:
    return Path(image_path).stem


def _text_relevant(source, item: dict) -> bool:
    if source.section not in item["expected_sections"]:
        return False
    expected_subsections = item.get("expected_subsections")
    return not expected_subsections or source.subsection in expected_subsections


def _eval_text(text_sources, item: dict) -> dict:
    hits = [_text_relevant(s, item) for s in text_sources]
    return {
        "precision": {k: sum(hits[:k]) / k for k in TEXT_KS},
        "recall": {k: float(any(hits[:k])) for k in TEXT_KS},
        "hits": hits,
    }


def _eval_image(image_sources, item: dict) -> dict | None:
    expected = item.get("expected_image_ids") or []
    if not expected:
        return None
    ids = [_image_id(s.image_path) for s in image_sources]
    hits = [i in expected for i in ids]
    return {
        "precision": {k: sum(hits[:k]) / k for k in IMAGE_KS},
        "recall": {k: len(set(ids[:k]) & set(expected)) / len(expected) for k in IMAGE_KS},
        "hits": hits,
        "retrieved_ids": ids,
    }


def evaluate(pipeline: RagPipeline, eval_set: list[dict]) -> list[dict]:
    results = []
    for item in eval_set:
        text_sources, image_sources = pipeline.retrieve(item["question"])
        results.append({
            "id": item["id"],
            "question": item["question"],
            "text": _eval_text(text_sources, item),
            "image": _eval_image(image_sources, item),
            "retrieved_sections": [(s.section, s.subsection) for s in text_sources],
            "retrieved_image_ids": [_image_id(s.image_path) for s in image_sources],
            "expected_sections": item["expected_sections"],
            "expected_subsections": item.get("expected_subsections"),
            "expected_image_ids": item.get("expected_image_ids") or [],
        })
    return results


def summarize(results: list[dict]) -> dict:
    image_results = [r["image"] for r in results if r["image"] is not None]
    return {
        "num_questions": len(results),
        "num_image_questions": len(image_results),
        "text_precision": {k: statistics.mean(r["text"]["precision"][k] for r in results) for k in TEXT_KS},
        "text_recall": {k: statistics.mean(r["text"]["recall"][k] for r in results) for k in TEXT_KS},
        "image_precision": {k: statistics.mean(r["precision"][k] for r in image_results) for k in IMAGE_KS},
        "image_recall": {k: statistics.mean(r["recall"][k] for r in image_results) for k in IMAGE_KS},
    }


def print_report(results: list[dict], summary: dict) -> None:
    print("=" * 78)
    print(f"Evaluated {summary['num_questions']} questions "
          f"({summary['num_image_questions']} with image ground truth)\n")

    print("Text retrieval (relevance = section[/subsection] match; recall = hit-rate@k)")
    print("  Precision@k: " + "  ".join(f"k={k}: {summary['text_precision'][k]:.2f}" for k in TEXT_KS))
    print("  Recall@k:    " + "  ".join(f"k={k}: {summary['text_recall'][k]:.2f}" for k in TEXT_KS))

    print("\nImage retrieval (exact image-id match)")
    print("  Precision@k: " + "  ".join(f"k={k}: {summary['image_precision'][k]:.2f}" for k in IMAGE_KS))
    print("  Recall@k:    " + "  ".join(f"k={k}: {summary['image_recall'][k]:.2f}" for k in IMAGE_KS))

    print("\n" + "-" * 78)
    print("Per-question detail:\n")
    for r in results:
        text_ok = "OK" if r["text"]["recall"][max(TEXT_KS)] else "MISS"
        print(f"[{r['id']}] text={text_ok:<4} {r['question']}")
        expected = str(r["expected_sections"])
        if r["expected_subsections"]:
            expected += f" / {r['expected_subsections']}"
        print(f"      expected section(s): {expected}")
        print(f"      retrieved sections:  {r['retrieved_sections']}")
        if r["image"] is not None:
            recall = r["image"]["recall"][max(IMAGE_KS)]
            img_status = "OK" if recall == 1.0 else ("PARTIAL" if recall > 0 else "MISS")
            print(f"      image={img_status:<7} expected: {r['expected_image_ids']}  "
                  f"retrieved: {r['retrieved_image_ids']}")
        print()


def print_markdown_tables(summary: dict) -> None:
    print("-" * 78)
    print("Markdown tables (paste into README):\n")

    print("| Metric | " + " | ".join(f"k={k}" for k in TEXT_KS) + " |")
    print("|---" * (len(TEXT_KS) + 1) + "|")
    print("| Text Precision@k | " + " | ".join(f"{summary['text_precision'][k]:.2f}" for k in TEXT_KS) + " |")
    print("| Text Recall@k (hit-rate) | " + " | ".join(f"{summary['text_recall'][k]:.2f}" for k in TEXT_KS) + " |")
    print()
    print("| Metric | " + " | ".join(f"k={k}" for k in IMAGE_KS) + " |")
    print("|---" * (len(IMAGE_KS) + 1) + "|")
    print("| Image Precision@k | " + " | ".join(f"{summary['image_precision'][k]:.2f}" for k in IMAGE_KS) + " |")
    print("| Image Recall@k | " + " | ".join(f"{summary['image_recall'][k]:.2f}" for k in IMAGE_KS) + " |")


def main() -> None:
    eval_set = load_eval_set()
    print(f"Loaded {len(eval_set)} eval questions from {EVAL_SET_PATH.name}")
    print("Loading RAG pipeline (embedding models + Chroma index)...")
    pipeline = RagPipeline()

    results = evaluate(pipeline, eval_set)
    summary = summarize(results)
    print_report(results, summary)
    print_markdown_tables(summary)


if __name__ == "__main__":
    main()
