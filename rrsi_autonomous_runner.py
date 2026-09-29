#!/usr/bin/env python3
"""
Autonomous RRSI Multi-Generation Evolution Runner.
Executes consecutive Regularized Recursive Self-Improvement cycles (arXiv:2609.24972).
Tracks budget annealing, critic screenings, pruner Pareto filtering, and invariant test verifications.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, List

from rrsi_engine import RRSIEngine
from graph_manager import GraphManager
from math_explorer import UnifiedMathHarness

class AutonomousRRSIRunner:
    def __init__(self, target_generations: int = 10):
        self.target_generations = target_generations
        self.gm = GraphManager()
        self.rrsi = RRSIEngine(graph_manager=self.gm)
        self.harness = UnifiedMathHarness()

    def run_evolution_loop(self) -> Dict[str, Any]:
        initial_cfg = self.rrsi.get_current_harness()
        start_gen = initial_cfg.get("generation", 0)
        end_gen = start_gen + self.target_generations

        print(f"\n=======================================================", flush=True)
        print(f"LAUNCHING AUTONOMOUS RRSI EVOLUTION LOOP (Empirical A/B Mode)", flush=True)
        print(f"Generations Target: Gen {start_gen} -> Gen {end_gen} (+{self.target_generations} steps)", flush=True)
        print(f"Initial Budget B({start_gen}): {self.rrsi.compute_budget(start_gen):.3f}", flush=True)
        print(f"=======================================================\n", flush=True)

        history = []
        start_time = time.time()

        for step in range(1, self.target_generations + 1):
            curr_gen = self.rrsi.config.get("generation", 0)
            budget = self.rrsi.compute_budget(curr_gen)
            print(f"[Cycle {step}/{self.target_generations}] Proposing Generation {curr_gen + 1} (Annealed Budget B={budget:.3f})...", flush=True)

            res = self.rrsi.evolve_step(harness_runner=self.harness.run_with_config)
            if not res.get("success"):
                print(f"  [X] Cycle Rejected: Status={res.get('status')} (Reason: {res.get('critic', {}).get('reason')})", flush=True)
                continue

            prop = res["proposal"]
            critic = res["critic"]
            pruner = res["pruner"]
            invars = res["invariants"]
            new_gen = res["generation"]

            print(f"  • Component Mutated: {prop.get('component')} ({prop.get('change_summary')})", flush=True)
            print(f"  • Hypothesis: \"{prop.get('hypothesis')}\"", flush=True)
            print(f"  • Critic Screen: {critic.get('verdict')} (GenScore: {critic.get('generalization_score'):.2f})", flush=True)
            print(f"  • Pruner Pareto: {pruner.get('action')} (Utility: {pruner.get('utility_score'):.2f})", flush=True)
            print(f"  • Invariants: {invars.get('passed')}/{invars.get('total')} Passed", flush=True)
            print(f"  => ACCEPTED: Committed Harness Gen {new_gen} to 3D Graph ({res['harness_node_id']})\n", flush=True)

            history.append({
                "cycle": step,
                "generation": new_gen,
                "proposal": prop,
                "critic": critic,
                "pruner": pruner,
                "invariants": invars
            })
            time.sleep(0.5) # Brief breathing space for disk persistence

        total_time = round(time.time() - start_time, 2)
        final_cfg = self.rrsi.get_current_harness()

        summary = {
            "start_generation": start_gen,
            "final_generation": final_cfg.get("generation"),
            "cycles_executed": len(history),
            "total_time_sec": total_time,
            "initial_config": initial_cfg,
            "final_optimized_config": final_cfg,
            "history": history,
            "graph_stats": self.gm.get_graph_data()["stats"]
        }

        # Save evolution summary
        out_path = "/Users/mac/.gemini/antigravity/scratch/math_explorer/data/rrsi_evolution_summary.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("=======================================================", flush=True)
        print(f"RRSI EVOLUTION LOOP COMPLETED in {total_time}s", flush=True)
        print(f"Harness evolved: Gen {start_gen} -> Gen {final_cfg.get('generation')}", flush=True)
        print(f"Final Annealed Budget: {final_cfg.get('annealed_budget'):.3f}", flush=True)
        print(f"Final Configuration State:")
        for k in ["prompt_system_style", "jev_difficulty_threshold", "jev_arxiv_threshold", "top_k_retrieval", "violetto_temperature", "violetto_top_k", "search_strategy"]:
            print(f"  - {k}: {final_cfg.get(k)}")
        print(f"3D Graph now contains: {summary['graph_stats']['total_nodes']} nodes, {summary['graph_stats']['total_links']} edges.")
        print("=======================================================\n", flush=True)

        return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous RRSI Evolution Runner")
    parser.add_argument("--steps", type=int, default=10, help="Number of evolution generations to execute")
    args = parser.parse_args()

    runner = AutonomousRRSIRunner(target_generations=args.steps)
    runner.run_evolution_loop()
