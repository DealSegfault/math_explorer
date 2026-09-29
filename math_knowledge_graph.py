#!/usr/bin/env python3
"""
Mathematical Knowledge Graph Engine powered by NetworkX.
Represents mathematical structures (Theorems, Lemmata, Definitions, Conjectures,
Proofs, Counterexamples, and Methods) and their formal relations:
USES, IMPLIES, DEPENDS_ON, GENERALIZES, SPECIAL_CASE_OF, CONTRADICTS, PROVED_BY.
"""

import time
import networkx as nx
from typing import Dict, Any, List, Optional, Tuple

MATH_NODE_COLORS = {
    "theorem": "#ffd600",        # Gold / Yellow
    "lemma": "#00e676",          # Emerald Green
    "definition": "#00b0ff",     # Electric Blue
    "conjecture": "#ff9100",     # Vivid Orange
    "counterexample": "#ff1744", # Crimson Red
    "method": "#d500f9",         # Neon Purple
    "problem": "#00e5ff",        # Vivid Cyan
    "proof": "#76ff03"           # Lime Green
}

MATH_NODE_SIZES = {
    "theorem": 22,
    "lemma": 16,
    "definition": 15,
    "conjecture": 24,
    "counterexample": 18,
    "method": 14,
    "problem": 16,
    "proof": 18
}

class MathKnowledgeGraph:
    def __init__(self):
        self.g = nx.DiGraph()
        self._seed_foundational_graph()

    def _seed_foundational_graph(self):
        """Seeds foundational theorems, lemmata, definitions, and reciprocity relations."""
        # Definitions
        self.add_concept("def_legendre", "definition", "Legendre Symbol (a/p)", "Quadratic residue character modulo odd prime p.")
        self.add_concept("def_jacobi", "definition", "Jacobi Symbol (a/n)", "Generalization of Legendre symbol to odd composite integers.")
        self.add_concept("def_dedekind", "definition", "Dedekind Domain", "Integral domain with unique factorization of nonzero ideals.")
        self.add_concept("def_class_group", "definition", "Ideal Class Group Cl(K)", "Measures failure of unique factorization in ring of integers.")
        self.add_concept("def_eisenstein", "definition", "Eisenstein Integers Z[w]", "Ring of integers Q(sqrt(-3)) with w^3 = 1.")

        # Lemmata
        self.add_concept("lem_gauss", "lemma", "Gauss's Lemma", "Expresses (a/p) in terms of number of negative residues modulo p.")
        self.add_concept("lem_supp_first", "lemma", "First Supplement: (-1/p)", "(-1/p) = (-1)^((p-1)/2), equal to 1 iff p == 1 (mod 4).")
        self.add_concept("lem_supp_second", "lemma", "Second Supplement: (2/p)", "(2/p) = (-1)^((p^2-1)/8), equal to 1 iff p == +-1 (mod 8).")
        self.add_concept("lem_euler_crit", "lemma", "Euler's Criterion", "(a/p) == a^((p-1)/2) (mod p).")

        # Theorems
        self.add_concept("thm_quad_recip", "theorem", "Law of Quadratic Reciprocity", "Gauss 1796: (p/q)(q/p) = (-1)^((p-1)/2 * (q-1)/2) for odd distinct primes.")
        self.add_concept("thm_cubic_recip", "theorem", "Cubic Reciprocity Law", "Primary prime reciprocity in Eisenstein integers Z[w].")
        self.add_concept("thm_artin_recip", "theorem", "Artin Reciprocity Law", "Global class field theory: canonical isomorphism Cl_K = Gal(L/K).")
        self.add_concept("thm_flt", "theorem", "Fermat's Little Theorem", "a^(p-1) == 1 (mod p) for prime p and gcd(a, p) = 1.")

        # Conjectures & Barriers
        self.add_concept("conj_collatz", "conjecture", "Collatz 3x+1 Conjecture", "All positive integers iterate to 1 under T(n).")
        self.add_concept("lem_steiner_bound", "lemma", "Steiner Cycle Bound", "Non-trivial cycles in 3x+1 must satisfy x0*(2^S - 3^k) = sum 3^(k-1-j)*2^(s_j).")
        self.add_concept("meth_baker", "method", "Baker's Linear Forms in Logarithms", "Provides lower bounds for |2^S - 3^k|, ruling out small and medium cycles.")

        # Add Relations
        self.add_relation("lem_euler_crit", "def_legendre", "DEPENDS_ON")
        self.add_relation("lem_gauss", "lem_euler_crit", "USES")
        self.add_relation("thm_quad_recip", "lem_gauss", "DEPENDS_ON")
        self.add_relation("thm_quad_recip", "lem_supp_first", "USES")
        self.add_relation("thm_quad_recip", "lem_supp_second", "USES")
        self.add_relation("thm_cubic_recip", "def_eisenstein", "DEPENDS_ON")
        self.add_relation("thm_artin_recip", "thm_quad_recip", "GENERALIZES")
        self.add_relation("thm_artin_recip", "thm_cubic_recip", "GENERALIZES")
        self.add_relation("thm_artin_recip", "def_class_group", "USES")
        self.add_relation("thm_artin_recip", "def_dedekind", "DEPENDS_ON")
        self.add_relation("def_jacobi", "def_legendre", "GENERALIZES")
        self.add_relation("conj_collatz", "lem_steiner_bound", "USES")
        self.add_relation("lem_steiner_bound", "meth_baker", "DEPENDS_ON")

    def add_concept(self, node_id: str, concept_type: str, title: str, description: str, data: Optional[Dict[str, Any]] = None):
        self.g.add_node(
            node_id,
            id=node_id,
            type=concept_type,
            title=title,
            label=title,
            description=description,
            val=MATH_NODE_SIZES.get(concept_type, 16),
            color=MATH_NODE_COLORS.get(concept_type, "#00e5ff"),
            data=data or {}
        )

    def add_relation(self, source: str, target: str, relation_type: str):
        self.g.add_edge(source, target, label=relation_type, relation=relation_type)

    def calculate_centrality(self) -> Dict[str, float]:
        """Calculates PageRank centrality to highlight key foundational theorems."""
        if len(self.g) == 0:
            return {}
        try:
            return nx.pagerank(self.g)
        except Exception:
            return nx.degree_centrality(self.g)

    def find_proof_path(self, source_id: str, target_id: str) -> List[str]:
        """Finds shortest logical dependency path between two mathematical concepts."""
        try:
            return nx.shortest_path(self.g, source=source_id, target=target_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def export_3d_graph_format(self) -> Dict[str, Any]:
        """Exports graph in 3D Force Graph format with NetworkX centrality metadata."""
        centrality = self.calculate_centrality()
        nodes = []
        for n, attrs in self.g.nodes(data=True):
            node_dict = dict(attrs)
            # Modulate node size by mathematical centrality
            score = centrality.get(n, 0.0)
            base_val = MATH_NODE_SIZES.get(attrs.get("type", "theorem"), 16)
            node_dict["val"] = round(base_val * (1.0 + 3.0 * score), 1)
            node_dict["centrality"] = round(score, 4)
            nodes.append(node_dict)

        links = []
        for u, v, attrs in self.g.edges(data=True):
            links.append({
                "source": u,
                "target": v,
                "label": attrs.get("label", "DEPENDS_ON"),
                "color": "#4fc3f7" if attrs.get("label") == "GENERALIZES" else "#90a4ae",
                "curvature": 0.15 if attrs.get("label") == "GENERALIZES" else 0.0,
                "particles": attrs.get("label") in ["IMPLIES", "GENERALIZES"]
            })

        return {
            "nodes": nodes,
            "links": links,
            "stats": {
                "total_theorems": sum(1 for n, a in self.g.nodes(data=True) if a.get("type") == "theorem"),
                "total_lemmata": sum(1 for n, a in self.g.nodes(data=True) if a.get("type") == "lemma"),
                "total_definitions": sum(1 for n, a in self.g.nodes(data=True) if a.get("type") == "definition"),
                "total_relations": len(self.g.edges),
                "timestamp": time.time()
            }
        }
