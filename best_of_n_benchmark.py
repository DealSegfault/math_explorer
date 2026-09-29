"""Paired AIME comparison of one Violetto sample versus a batched vote."""

import argparse
import json
import time

from benchmark_data import load_aime, paired_improvement_pvalue
from best_of_n import solve_best_of_n
from config import DATA_DIR
from violetto_engine import ViolettoEngine


def compare(engine, problems, n=4, max_tokens=384):
    if not problems:
        raise ValueError("No benchmark problems selected")
    results = []
    for index, problem in enumerate(problems):
        row = {"id": problem["id"], "ground_truth": problem["ground_truth"]}
        for count, label in ((1, "single"), (n, "best_of_n")):
            start = time.perf_counter()
            vote = solve_best_of_n(engine, problem["query"], n=count, max_tokens=max_tokens, seed=index + 1)
            row[label] = {"answer": vote["answer"], "correct": vote["answer"] == problem["ground_truth"],
                          "latency_sec": round(time.perf_counter() - start, 2), "status": vote["status"]}
        results.append(row)
    wins = sum(not row["single"]["correct"] and row["best_of_n"]["correct"] for row in results)
    losses = sum(row["single"]["correct"] and not row["best_of_n"]["correct"] for row in results)
    return {"n": n, "questions": len(results), "single_accuracy": sum(row["single"]["correct"] for row in results) / len(results),
            "best_of_n_accuracy": sum(row["best_of_n"]["correct"] for row in results) / len(results),
            "wins": wins, "losses": losses, "paired_p_value": paired_improvement_pvalue(wins, losses), "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare N=1 with batched Violetto sampling on AIME")
    parser.add_argument("--split", choices=["development", "evaluation"], default="evaluation")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--n", type=int, choices=[4, 8], default=4)
    parser.add_argument("--max-tokens", type=int, default=384)
    args = parser.parse_args()
    results = compare(ViolettoEngine(), load_aime(args.split, args.limit), n=args.n, max_tokens=args.max_tokens)
    path = DATA_DIR / "best_of_n_results.json"
    path.write_text(json.dumps(results, indent=2))
    print(json.dumps({key: value for key, value in results.items() if key != "results"}, indent=2))
