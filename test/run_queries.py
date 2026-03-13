"""
Main script to run queries against the RAG API and score the results.

Usage:
    python -m test.run_queries                    # Run all queries from config
    python -m test.run_queries --queries test/queries_small.txt  # Use specific file
    python -m test.run_queries --skip-api        # Skip API calls, use existing results
    python -m test.run_queries --skip-scoring    # Skip LLM scoring
"""

import argparse
import json
import time
from pathlib import Path

import requests

from test import config
from test.scorer import get_scorer, ScoreResult


def load_queries(file_path: str) -> list[str]:
    """Load queries from a text file, one per line."""
    queries = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                queries.append(line)
    return queries


def call_query_api(query: str, top_k: int = None) -> dict:
    """Call the /query endpoint and return the response."""
    payload = {"query": query}
    if top_k is not None:
        payload["top_k"] = top_k

    response = requests.post(
        f"{config.API_URL}/query",
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def save_results(results: list[dict], file_path: str):
    """Save results to a JSON file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(results, f, indent=2)


def load_results(file_path: str) -> list[dict]:
    """Load results from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def print_progress(current: int, total: int, question: str):
    """Print progress information."""
    print(f"\n[{current}/{total}] Processing: {question[:50]}...")


def print_score_result(result: ScoreResult):
    """Print a single score result."""
    answered_str = "✓" if result.answered else "✗"
    print(f"  Answered: {answered_str} | Score: {result.score}/10")


def print_summary(results: list[ScoreResult]):
    """Print summary statistics."""
    total = len(results)
    answered_count = sum(1 for r in results if r.answered)
    scores = [r.score for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0
    needs_review = len([r for r in results if r.score < 5])

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total questions:     {total}")
    print(f"Answered:            {answered_count}/{total} ({answered_count/total*100:.1f}%)")
    print(f"Average score:       {avg_score:.2f}/10")
    print(f"Needs review:        {needs_review}")
    print("=" * 60)


def run_queries(queries_file: str = None, skip_api: bool = False, skip_scoring: bool = False, test: bool = False):
    """
    Run the full testing pipeline.

    Args:
        queries_file: Path to queries file. Defaults to config.QUERIES_FILE
        skip_api: If True, skip calling the API and use existing query_results.json
        skip_scoring: If True, skip scoring and just save query results
        test: if True, run with testing (smaller) queries file
    """
    if test:
        queries_file = queries_file or config.QUERIES_TEST_FILE
    else:
        queries_file = queries_file or config.QUERIES_FILE

    print(f"Loading queries from {queries_file}...")
    queries = load_queries(queries_file)
    print(f"Loaded {len(queries)} queries\n")

    query_results = []

    if not skip_api:
        print("Calling RAG API for each query...")
        for i, query in enumerate(queries, 1):
            print_progress(i, len(queries), query)

            try:
                result = call_query_api(query, top_k=config.QUERY_TOP_K)
                query_results.append({
                    "question": query,
                    "response": result,
                })
                print(f"  -> Got {len(result.get('results', []))} chunks")
            except Exception as e:
                print(f"  -> ERROR: {e}")
                query_results.append({
                    "question": query,
                    "error": str(e),
                })

            time.sleep(0.1)

        save_results(query_results, config.QUERY_RESULTS_FILE)
        print(f"\nQuery results saved to {config.QUERY_RESULTS_FILE}")
    else:
        print("Skipping API calls, loading existing results...")
        query_results = load_results(config.QUERY_RESULTS_FILE)
        print(f"Loaded {len(query_results)} query results\n")

    if skip_scoring:
        print("Skipping scoring (--skip-scoring flag set)")
        return

    print("Scoring results with LLM...")
    scorer = get_scorer("ollama")

    scored_results = []
    needs_review = []
    for i, qr in enumerate(query_results, 1):
        question = qr.get("question")
        response = qr.get("response", {})
        chunks = response.get("results", [])

        if not chunks:
            print(f"\n[{i}/{len(query_results)}] No chunks for: {question[:50]}...")
            scored_results.append(ScoreResult(
                question=question,
                answered=False,
                score=1,
                reasoning="No chunks retrieved from API",
            ))
            if 1 < 5:
                needs_review.append(ScoreResult(
                    question=question,
                    answered=False,
                    score=1,
                    reasoning="No chunks retrieved from API",
                ))
            continue

        print(f"\n[{i}/{len(query_results)}] Scoring: {question[:50]}...")

        try:
            score_result = scorer.score(question, chunks)
            scored_results.append(score_result)
            print_score_result(score_result)
            if score_result.score < 5:
                needs_review.append(score_result)
        except Exception as e:
            print(f"  -> Scoring ERROR: {e}")
            scored_results.append(ScoreResult(
                question=question,
                answered=False,
                score=1,
                reasoning=f"Scoring error: {e}",
            ))
            if 1 < 5:
                needs_review.append(ScoreResult(
                    question=question,
                    answered=False,
                    score=1,
                    reasoning=f"Scoring error: {e}",
                ))

        time.sleep(0.1)

    if needs_review:
        review_data = [
            {
                "question": r.question,
                "answered": r.answered,
                "score": r.score,
                "reasoning": r.reasoning,
            }
            for r in needs_review
        ]
        save_results(review_data, config.REVIEW_FILE)
        print(f"\n⚠ {len(needs_review)} results flagged for review (saved to {config.REVIEW_FILE})")

    scored_data = [
        {
            "question": r.question,
            "answered": r.answered,
            "score": r.score,
            "reasoning": r.reasoning,
        }
        for r in scored_results
    ]
    save_results(scored_data, config.SCORES_FILE)
    print(f"\nScored results saved to {config.SCORES_FILE}")

    print_summary(scored_results)


def main():
    parser = argparse.ArgumentParser(description="Run RAG queries and score results")
    parser.add_argument(
        "--queries", "-q",
        type=str,
        default=None,
        help="Path to queries file (default: config.QUERIES_FILE)"
    )
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Skip API calls, use existing query_results.json"
    )
    parser.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Skip LLM scoring"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run only test queries"
    )
    args = parser.parse_args()

    run_queries(
        queries_file=args.queries,
        skip_api=args.skip_api,
        skip_scoring=args.skip_scoring,
        test=args.test
    )


if __name__ == "__main__":
    main()
