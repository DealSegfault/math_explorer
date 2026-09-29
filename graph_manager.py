#!/usr/bin/env python3
"""
Graph Manager for Mathematical Exploration & RRSI Self-Improvement System.
Persists dynamic 3D graph state (Nodes & Edges) in JSON format.
Supports real-time node addition, edge wiring, and category filtering.
"""

import os
import json
import time
import threading
from typing import Dict, Any, List, Optional

GRAPH_FILE_DEFAULT = "/Users/mac/.gemini/antigravity/scratch/math_explorer/data/graph_state.json"

# Color Palette for 3D Graph Nodes
NODE_COLORS = {
    "query": "#00e5ff",            # Vivid Cyan
    "jev_decision": "#ffd600",     # Warm Gold/Yellow
    "document": "#2979ff",         # Deep Blue
    "pageindex_node": "#00b0ff",   # Electric Blue
    "violetto_proof": "#d500f9",   # Neon Purple / Violet
    "astra_proof": "#00e676",      # Emerald Green
    "verification": "#76ff03",     # Lime Green
    "harness_state": "#ff1744",    # Crimson Red
    "mutation_proposal": "#ff9100",# Bright Orange
    "critic_eval": "#ffea00",      # Bright Yellow
    "pruner_decision": "#ff6d00",  # Dark Orange
    "invariant_test": "#00e5ff"    # Light Cyan
}

NODE_SIZES = {
    "query": 18,
    "jev_decision": 14,
    "document": 16,
    "pageindex_node": 10,
    "violetto_proof": 20,
    "astra_proof": 22,
    "verification": 12,
    "harness_state": 26,
    "mutation_proposal": 14,
    "critic_eval": 12,
    "pruner_decision": 12,
    "invariant_test": 12
}

class GraphManager:
    def __init__(self, filepath: str = GRAPH_FILE_DEFAULT):
        self.filepath = filepath
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.links: List[Dict[str, Any]] = []
        self._load_or_initialize()

    def _load_or_initialize(self):
        with self._lock:
            if os.path.exists(self.filepath):
                try:
                    with open(self.filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.nodes = {n["id"]: n for n in data.get("nodes", [])}
                        self.links = data.get("links", [])
                        return
                except Exception as e:
                    print(f"Error loading graph file: {e}. Reinitializing baseline.", flush=True)

            self.nodes = {}
            self.links = []
            self._seed_baseline_graph()
            self._save_unlocked()

    def _seed_baseline_graph(self):
        """Seeds initial generation 0 harness state and sample algebraic literature nodes."""
        # Initial Harness Gen 0 Node
        gen0_id = "harness_gen_0"
        self.nodes[gen0_id] = {
            "id": gen0_id,
            "type": "harness_state",
            "label": "Harness Gen 0 (Baseline)",
            "title": "Initial Agent Harness Config",
            "val": NODE_SIZES["harness_state"],
            "color": NODE_COLORS["harness_state"],
            "generation": 0,
            "timestamp": time.time(),
            "data": {
                "generation": 0,
                "jev_difficulty_threshold": 2.5,
                "jev_arxiv_threshold": 0.65,
                "top_k_retrieval": 2,
                "violetto_temperature": 0.6,
                "violetto_top_k": 50,
                "prompt_system_style": "rigorous_math_proof",
                "search_strategy": "hierarchical_pageindex",
                "invariants_passed": 3,
                "annealed_budget": 1.0
            }
        }

        # Corpus Document Node
        doc_id = "doc_ANT_Reciprocity"
        self.nodes[doc_id] = {
            "id": doc_id,
            "type": "document",
            "label": "Corpus: Number Theory & Reciprocity",
            "title": "Algebraic Number Theory & Reciprocity Laws",
            "val": NODE_SIZES["document"],
            "color": NODE_COLORS["document"],
            "generation": 0,
            "timestamp": time.time(),
            "data": {
                "source": "corpus/algebraic_number_theory.md",
                "chapters": ["Divisibility & Ideals", "Legendre & Jacobi Symbols", "Quadratic Reciprocity", "Cubic Reciprocity", "Artin Reciprocity"]
            }
        }

        # Sub-nodes for key theorems in corpus
        theorems = [
            ("sec_quad_recip", "Quadratic Reciprocity", "Gauss Reciprocity Law: (p/q)(q/p) = (-1)^((p-1)/2 * (q-1)/2)"),
            ("sec_euler_crit", "Euler's Criterion", "(a/p) = a^((p-1)/2) mod p"),
            ("sec_cubic_recip", "Cubic Reciprocity", "Eisenstein Integers Z[omega] and primary prime reciprocity")
        ]

        for sec_id, title, desc in theorems:
            self.nodes[sec_id] = {
                "id": sec_id,
                "type": "pageindex_node",
                "label": title,
                "title": title,
                "val": NODE_SIZES["pageindex_node"],
                "color": NODE_COLORS["pageindex_node"],
                "generation": 0,
                "timestamp": time.time(),
                "data": {"content": desc, "doc_id": doc_id}
            }
            self.links.append({
                "source": doc_id,
                "target": sec_id,
                "label": "contains_section",
                "color": "#1e88e5",
                "curvature": 0.1
            })

    def _save_unlocked(self):
        data = {
            "nodes": list(self.nodes.values()),
            "links": self.links,
            "stats": {
                "node_count": len(self.nodes),
                "link_count": len(self.links),
                "last_updated": time.time()
            }
        }
        tmp_path = self.filepath + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, self.filepath)

    def save(self):
        with self._lock:
            self._save_unlocked()

    def add_node(
        self,
        node_id: str,
        node_type: str,
        label: str,
        title: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        generation: int = 0
    ) -> Dict[str, Any]:
        with self._lock:
            node = {
                "id": node_id,
                "type": node_type,
                "label": label,
                "title": title or label,
                "val": NODE_SIZES.get(node_type, 14),
                "color": NODE_COLORS.get(node_type, "#00e5ff"),
                "generation": generation,
                "timestamp": time.time(),
                "data": data or {}
            }
            self.nodes[node_id] = node
            self._save_unlocked()
            return node

    def add_link(
        self,
        source: str,
        target: str,
        label: str,
        color: Optional[str] = None,
        curvature: float = 0.0,
        particles: bool = False
    ):
        with self._lock:
            # Check for duplicates
            for link in self.links:
                if link["source"] == source and link["target"] == target and link["label"] == label:
                    return
            self.links.append({
                "source": source,
                "target": target,
                "label": label,
                "color": color or "#546e7a",
                "curvature": curvature,
                "particles": particles
            })
            self._save_unlocked()

    def record_exploration(
        self,
        query: str,
        routing: Dict[str, Any],
        retrieved_nodes: List[Dict[str, Any]],
        solver_engine: str,
        solution_text: str,
        tokens_or_metrics: Dict[str, Any],
        gate_result: Dict[str, Any],
        generation: int = 0
    ) -> str:
        """
        Records an end-to-end exploration execution as an interconnected sub-graph:
        HarnessGen -> Query -> JevDecision -> [Retrieved Tree Nodes] -> SolverProof -> Verification
        """
        ts = int(time.time() * 1000)
        query_id = f"query_{ts}"
        jev_id = f"jev_{ts}"
        proof_id = f"proof_{ts}"
        gate_id = f"gate_{ts}"

        # 1. Query Node
        self.add_node(
            node_id=query_id,
            node_type="query",
            label=f"Q: {query[:36]}..." if len(query) > 36 else f"Q: {query}",
            title=query,
            data={"query": query, "timestamp": time.time()},
            generation=generation
        )

        # Link from current Harness State
        harness_id = f"harness_gen_{generation}"
        if harness_id in self.nodes:
            self.add_link(harness_id, query_id, label="executed_by", color="#ff1744", particles=True)

        # 2. JEV Decision Node
        jev_summary = f"Diff: {routing.get('difficulty_score', 0):.1f} | Engine: {routing.get('recommended_engine', '')}"
        self.add_node(
            node_id=jev_id,
            node_type="jev_decision",
            label=f"JEV Triage (<100ms)",
            title=f"JEV System One: {jev_summary}",
            data=routing,
            generation=generation
        )
        self.add_link(query_id, jev_id, label="triaged_by", color="#ffd600", particles=True)

        # 3. Retrieved Literature / PageIndex Nodes
        for idx, rn in enumerate(retrieved_nodes):
            rn_id = rn.get("node_id") or f"pnode_{ts}_{idx}"
            rn_title = rn.get("title") or rn.get("full_path") or f"Section {idx+1}"
            if rn_id not in self.nodes:
                self.add_node(
                    node_id=rn_id,
                    node_type="pageindex_node",
                    label=rn_title[:30],
                    title=rn_title,
                    data=rn,
                    generation=generation
                )
            self.add_link(jev_id, rn_id, label="retrieved_context", color="#00b0ff", curvature=0.15)
            self.add_link(rn_id, proof_id, label="fed_to_solver", color="#00b0ff")

        # 4. Solver Proof Node (Violetto or Codex Astra)
        is_astra = (solver_engine == "codex_astra")
        proof_type = "astra_proof" if is_astra else "violetto_proof"
        solver_name = "Codex Astra (gpt-6-astra xhigh)" if is_astra else "Limite 1B Violetto (MPS)"
        self.add_node(
            node_id=proof_id,
            node_type=proof_type,
            label=f"Proof: {solver_name}",
            title=f"Solution by {solver_name}",
            data={
                "engine": solver_engine,
                "solution": solution_text,
                "metrics": tokens_or_metrics
            },
            generation=generation
        )
        self.add_link(jev_id, proof_id, label="dispatched_to", color="#00e676" if is_astra else "#d500f9", particles=True)

        # 5. Verification Gate Node
        plausible = gate_result.get("is_plausible", {}).get("noul", 0.0) if isinstance(gate_result, dict) else 1.0
        rigor = gate_result.get("rigor_score", {}).get("score", 0.0) if isinstance(gate_result, dict) else 2.0
        self.add_node(
            node_id=gate_id,
            node_type="verification",
            label=f"JEV Gate: {plausible:.0%} Plausible | Rigor {rigor:.1f}",
            title="Verification & Confidence Gating",
            data=gate_result,
            generation=generation
        )
        self.add_link(proof_id, gate_id, label="verified_by", color="#76ff03")

        return query_id

    def record_rrsi_step(
        self,
        prev_generation: int,
        new_generation: int,
        proposal: Dict[str, Any],
        critic_result: Dict[str, Any],
        pruner_result: Dict[str, Any],
        invariant_result: Dict[str, Any],
        new_config: Dict[str, Any]
    ) -> str:
        """
        Records an RRSI self-improvement cycle:
        HarnessGen(g) -> Proposal -> Critic -> Pruner -> Invariants -> HarnessGen(g+1)
        """
        ts = int(time.time() * 1000)
        prop_id = f"rrsi_prop_g{new_generation}_{ts}"
        critic_id = f"rrsi_critic_g{new_generation}_{ts}"
        pruner_id = f"rrsi_prune_g{new_generation}_{ts}"
        invar_id = f"rrsi_invar_g{new_generation}_{ts}"
        new_harness_id = f"harness_gen_{new_generation}"
        prev_harness_id = f"harness_gen_{prev_generation}"

        # 1. Proposal Node
        self.add_node(
            node_id=prop_id,
            node_type="mutation_proposal",
            label=f"Mutation: {proposal.get('component', 'config')} (Δ {proposal.get('change_summary', '')})",
            title=f"RRSI Proposed Mutation: {proposal.get('hypothesis', '')}",
            data=proposal,
            generation=new_generation
        )
        if prev_harness_id in self.nodes:
            self.add_link(prev_harness_id, prop_id, label="proposed_from", color="#ff9100", particles=True)

        # 2. Critic Evaluation Node
        self.add_node(
            node_id=critic_id,
            node_type="critic_eval",
            label=f"Critic: {critic_result.get('verdict', 'ACCEPT')} (Score: {critic_result.get('generalization_score', 1.0):.2f})",
            title=f"RRSI Critic Generalization Check",
            data=critic_result,
            generation=new_generation
        )
        self.add_link(prop_id, critic_id, label="evaluated_by", color="#ffea00")

        # 3. Pruner Decision Node
        self.add_node(
            node_id=pruner_id,
            node_type="pruner_decision",
            label=f"Pruner: {pruner_result.get('action', 'KEEP')} (Utility: {pruner_result.get('utility_score', 1.0):.2f})",
            title="RRSI Pruner (Complexity vs Accuracy Pareto)",
            data=pruner_result,
            generation=new_generation
        )
        self.add_link(critic_id, pruner_id, label="pruned_by", color="#ff6d00")

        # 4. Invariant Test Node
        passed = invariant_result.get("passed", 0)
        total = invariant_result.get("total", 0)
        self.add_node(
            node_id=invar_id,
            node_type="invariant_test",
            label=f"Invariants: {passed}/{total} Passed",
            title="Regression & Mathematical Invariants",
            data=invariant_result,
            generation=new_generation
        )
        self.add_link(pruner_id, invar_id, label="tested_against", color="#00e5ff")

        # 5. New Harness Generation Node
        self.add_node(
            node_id=new_harness_id,
            node_type="harness_state",
            label=f"Harness Gen {new_generation}",
            title=f"Active Agent Harness Gen {new_generation}",
            data=new_config,
            generation=new_generation
        )
        self.add_link(invar_id, new_harness_id, label="accepted_as_active", color="#ff1744", particles=True)
        # Direct evolution link between generations
        if prev_harness_id in self.nodes:
            self.add_link(prev_harness_id, new_harness_id, label="evolved_to", color="#ff1744", curvature=0.25)

        return new_harness_id

    def get_graph_data(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "nodes": list(self.nodes.values()),
                "links": list(self.links),
                "stats": {
                    "total_nodes": len(self.nodes),
                    "total_links": len(self.links),
                    "types": {
                        t: sum(1 for n in self.nodes.values() if n["type"] == t)
                        for t in set(n["type"] for n in self.nodes.values())
                    },
                    "timestamp": time.time()
                }
            }

    def reset_graph(self):
        with self._lock:
            self.nodes = {}
            self.links = []
            self._seed_baseline_graph()
            self._save_unlocked()
