#!/usr/bin/env python3
"""
Automated Benchmarking Suite for Math Explorer Harness.
Evaluates multi-tier reasoning (Local Violetto vs Codex Astra) on competition-grade
mathematics (AIME, AMC, Putnam, Number Theory, Group Theory, Algebra).
Integrates SymPy symbolic verification and ground truth correctness scoring.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, List, Optional

from math_explorer import UnifiedMathHarness

BENCHMARK_PROBLEMS = [
    {
        "id": "AIME_2024_DIVISIBILITY",
        "domain": "number_theory",
        "difficulty_tier": "intermediate",
        "query": "Count all integers n < 1000 such that n is divisible by 6, not divisible by 4, and not divisible by 9. Give the final answer in \\boxed{}.",
        "ground_truth": "55"
    },
    {
        "id": "AMC_2024_FACTORS",
        "domain": "number_theory",
        "difficulty_tier": "intermediate",
        "query": "How many positive integer divisors of 2024 are multiples of 4? Note that 2024 = 2^3 * 11 * 23. Give the final answer in \\boxed{}.",
        "ground_truth": "8"
    },
    {
        "id": "RECIPROCITY_11_13",
        "domain": "number_theory",
        "difficulty_tier": "advanced",
        "query": "Compute the Legendre symbol (11/13) using the Law of Quadratic Reciprocity step by step. State the final value as 1 or -1 in \\boxed{}.",
        "ground_truth": "-1"
    },
    {
        "id": "RECIPROCITY_7_17",
        "domain": "number_theory",
        "difficulty_tier": "advanced",
        "query": "Calculate the Legendre symbol (7/17) using Gauss's Law of Quadratic Reciprocity step by step. State the final value as 1 or -1 in \\boxed{}.",
        "ground_truth": "-1"
    },
    {
        "id": "FINITE_FIELDS_CUBIC_RESIDUES",
        "domain": "algebra",
        "difficulty_tier": "advanced",
        "query": "In the finite field F_13 with 13 elements, determine the number of nonzero elements that are cubic residues (cubes of nonzero elements). State the count in \\boxed{}.",
        "ground_truth": "4"
    },
    {
        "id": "POLYNOMIAL_ROOTS_UNITY",
        "domain": "algebra",
        "difficulty_tier": "advanced",
        "query": "Find the number of positive integers n <= 100 such that the polynomial x^2 + x + 1 divides x^(2n) + 1 in R[x]. State the count in \\boxed{}.",
        "ground_truth": "0"
    }
]

from config import DATA_DIR

class BenchmarkSuite:
    def __init__(self, output_path: str = str(DATA_DIR / "benchmark_results.json")):
        self.output_path = output_path
        self.harness = UnifiedMathHarness()

    def run_benchmark(self, engine_override: Optional[str] = None, problems: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        suite = problems or BENCHMARK_PROBLEMS
        print(f"\n=======================================================", flush=True)
        print(f"STARTING MATH EXPLORER BENCHMARK SUITE ({len(suite)} Problems)", flush=True)
        print(f"Engine Policy: {engine_override or 'Auto (JEV Gated Triage)'}", flush=True)
        print(f"=======================================================\n", flush=True)

        results = []
        start_time = time.time()

        for idx, prob in enumerate(suite, 1):
            pid = prob["id"]
            query = prob["query"]
            gt = prob["ground_truth"]
            print(f"\n--- [{idx}/{len(suite)}] Problem: {pid} (Domain: {prob['domain']}, GT: {gt}) ---", flush=True)

            t0 = time.time()
            exp_res = self.harness.explore(
                query=query,
                engine=engine_override,
                ground_truth=gt
            )
            elapsed = round(time.time() - t0, 2)

            ver = exp_res.get("verification") or exp_res.get("symbolic_verification", {})
            routing = exp_res.get("routing", {})
            gt_match = ver.get("ground_truth_matched")
            ext_ans = ver.get("extracted_answer")

            res_entry = {
                "id": pid,
                "domain": prob["domain"],
                "difficulty_tier": prob["difficulty_tier"],
                "jev_domain": routing.get("domain"),
                "jev_difficulty": routing.get("difficulty_score"),
                "engine_used": exp_res.get("solver_engine"),
                "ground_truth": gt,
                "extracted_answer": ext_ans,
                "correct": gt_match,
                "symbolic_valid_steps": ver.get("cas_checks", {}).get("valid", 0) + ver.get("smt_checks", {}).get("valid", 0),
                "symbolic_total_steps": ver.get("total_checks", 0),
                "symbolic_pass_rate": ver.get("pass_rate", 0.0),
                "verification_status": ver.get("status"),
                "is_verified": ver.get("is_verified", False),
                "latency_sec": elapsed,
                "early_exit": exp_res.get("early_exit", False),
                "jev_gate_plausible": exp_res.get("gate", {}).get("is_plausible", {}).get("noul", 0.0),
                "jev_gate_rigor": exp_res.get("gate", {}).get("rigor_score", {}).get("score", 0.0)
            }
            results.append(res_entry)
            status_str = "CORRECT" if gt_match else ("INCORRECT" if gt_match is False else "UNCONFIRMED")
            print(f">>> Result: {status_str} | Extracted: {ext_ans} | GT: {gt} | Time: {elapsed}s", flush=True)

        total_elapsed = round(time.time() - start_time, 2)
        total_eval = len(results)
        correct_count = sum(1 for r in results if r["correct"] is True)
        accuracy = round(correct_count / total_eval, 3) if total_eval > 0 else 0.0

        # Engine specific accuracy
        violetto_runs = [r for r in results if r["engine_used"] == "local_violetto"]
        astra_runs = [r for r in results if r["engine_used"] == "codex_astra"]

        violetto_correct = sum(1 for r in violetto_runs if r["correct"] is True)
        astra_correct = sum(1 for r in astra_runs if r["correct"] is True)

        summary = {
            "timestamp": time.time(),
            "total_problems": total_eval,
            "solved_correctly": correct_count,
            "overall_accuracy": accuracy,
            "total_elapsed_sec": total_elapsed,
            "avg_latency_sec": round(total_elapsed / total_eval, 2) if total_eval > 0 else 0.0,
            "violetto_metrics": {
                "total_runs": len(violetto_runs),
                "correct": violetto_correct,
                "accuracy": round(violetto_correct / len(violetto_runs), 3) if violetto_runs else 0.0
            },
            "astra_metrics": {
                "total_runs": len(astra_runs),
                "correct": astra_correct,
                "accuracy": round(astra_correct / len(astra_runs), 3) if astra_runs else 0.0
            },
            "symbolic_avg_pass_rate": round(sum(r["symbolic_pass_rate"] for r in results) / total_eval, 3) if total_eval > 0 else 0.0,
            "results": results
        }

        # Save to JSON
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("\n=======================================================", flush=True)
        print(f"BENCHMARK COMPLETED in {total_elapsed}s", flush=True)
        print(f"Overall Accuracy: {accuracy:.1%} ({correct_count}/{total_eval})", flush=True)
        if violetto_runs:
            print(f"Violetto (MPS): {summary['violetto_metrics']['accuracy']:.1%} ({violetto_correct}/{len(violetto_runs)})", flush=True)
        if astra_runs:
            print(f"Codex Astra (xhigh): {summary['astra_metrics']['accuracy']:.1%} ({astra_correct}/{len(astra_runs)})", flush=True)
        print(f"Symbolic Step Validation Rate: {summary['symbolic_avg_pass_rate']:.1%}", flush=True)
        print("=======================================================\n", flush=True)

        return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Math Explorer Benchmarking Suite")
    parser.add_argument("--engine", choices=["local_violetto", "codex_astra"], help="Force solver engine")
    args = parser.parse_args()

    suite = BenchmarkSuite()
    suite.run_benchmark(engine_override=args.engine)
