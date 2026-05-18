"""
Terminal interface for querying the GazeRAG pipeline.

Usage:
    python scripts/query_demo.py "what is a saccade?"
    python scripts/query_demo.py --interactive

The interactive mode keeps a prompt open so you can run several queries without
reloading the embedding model each time (which is the slow part).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.pipeline import RAGPipeline  # noqa: E402


def run_single_query(pipeline: RAGPipeline, question: str) -> None:
    """Run one query and print the result."""
    print(f"\n→ Question: {question}\n")
    print("→ Retrieving and generating...\n")
    response = pipeline.ask(question)

    print("=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(response.answer)
    print()
    print("=" * 80)
    print(f"SOURCES ({len(response.sources)})")
    print("=" * 80)
    for i, src in enumerate(response.sources, start=1):
        print(f"\n[{i}] {src['source']} — page {src['page']} (relevance {src['score']:.3f})")
        print(f"    {src['snippet']}")
    print()


def run_interactive(pipeline: RAGPipeline) -> None:
    """REPL mode: ask multiple questions without reloading the pipeline."""
    print("\nGazeRAG interactive mode. Type 'exit' or Ctrl-D to quit.\n")
    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break
        try:
            run_single_query(pipeline, question)
        except Exception as exc:
            print(f"  [!] Error: {exc}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the GazeRAG pipeline.")
    parser.add_argument(
        "question",
        nargs="*",
        help="The question to ask. If omitted, runs in interactive mode.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run an interactive query loop instead of one-shot mode.",
    )
    args = parser.parse_args()

    print("→ Loading RAG pipeline (this takes ~5-10 seconds on first run)...")
    pipeline = RAGPipeline()
    print("  ✓ Ready.\n")

    if args.interactive or not args.question:
        run_interactive(pipeline)
    else:
        question = " ".join(args.question)
        run_single_query(pipeline, question)


if __name__ == "__main__":
    main()
