"""Terminal harness for RagPipeline: ask a question, see the answer + sources.

Usage (run as a module so the `rag` package resolves correctly):
    python -m rag.cli                  interactive mode
    python -m rag.cli "your question"  one-shot mode
"""

import sys

from rag.pipeline import RagPipeline


def print_result(result) -> None:
    print("\n" + "=" * 70)
    print("ANSWER:")
    print(result.answer.strip())
    print("=" * 70)

    if result.text_sources:
        print("\nText sources used:")
        for i, src in enumerate(result.text_sources, 1):
            header = " > ".join(b for b in (src.section, src.subsection) if b)
            snippet = src.text[:150].replace("\n", " ")
            print(f"  [{i}] ({header}) dist={src.distance:.3f}")
            print(f"      {snippet}...")

    if result.image_sources:
        print("\nImage sources used:")
        for i, src in enumerate(result.image_sources, 1):
            header = " > ".join(b for b in (src.section, src.subsection) if b)
            print(f"  [{i}] {src.image_path} ({header}) dist={src.distance:.3f}")
            print(f"      {src.description[:150]}")
    print()


def main() -> None:
    print("Loading RAG pipeline (embedding models + Chroma index)...")
    pipeline = RagPipeline()
    print(f"Ready. Using Ollama model: {pipeline.ollama_model}\n")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print_result(pipeline.answer(query))
        return

    print("Ask a question about MR Tips & Tricks (empty line or Ctrl+C to quit).\n")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            break
        print_result(pipeline.answer(query))


if __name__ == "__main__":
    main()
