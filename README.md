# Math Explorer 3D & RRSI Autonomous Agent Harness

High-speed mathematical exploration harness combining ultra-fast System One routing, vectorless hierarchical literature retrieval, local specialized reasoning on Apple Silicon, frontier deep reasoning, and recursive self-improvement.

![Status](https://img.shields.io/badge/System-Active-00e5ff)
![RRSI](https://img.shields.io/badge/arXiv-2609.24972-ff1744)
![Hardware](https://img.shields.io/badge/Inference-Apple%20Silicon%20MPS-d500f9)
![CAS](https://img.shields.io/badge/CAS-SymPy%20Deterministic-green)
![SMT](https://img.shields.io/badge/SMT-Z3%20Theorem%20Prover-blue)

<p align="center">
  <img src="assets/graph_3d.png" alt="Math Explorer 3D Knowledge Graph" width="850">
  <br>
  <em>Interactive 3D WebGL topological knowledge graph of mathematical proofs, lemmata, SMT counterexamples, and RRSI evolution states.</em>
</p>

---

## 🏛️ High-Speed Architecture Overview

```
                         [ User Question / Conjecture ]
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
        ┌─────────────────────────┐           ┌─────────────────────────┐
        │  Tier 0: CAS/SMT Probe  │           │   Multi-Level SQLite    │
        │ (SymPy Exact Early Exit)│           │   Persistent Cache      │
        └───────────┬─────────────┘           └───────────┬─────────────┘
                    │                                     │
           Solved & Verified? (<5ms)               Cache Hit? (0.1ms)
                    ├────────── YES ──────────────────────┤
                    ▼                                     ▼
             [ INSTANT RETURN ]                   [ INSTANT RETURN ]
                    │ NO
                    ▼
        ┌─────────────────────────────────────────────────────┐
        │       Tier 1: TypeSafe JEV System One Triage        │
        │   (<100ms Domain, Difficulty, arXiv Need, Gating)   │
        └──────────────────────────┬──────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
       ┌─────────────────────────┐   ┌─────────────────────────┐
       │ Stage 1: Tantivy BM25   │   │   Direct Execution      │
       │ Lexical Pruning (<1ms)  │   │  (No context required)  │
       └────────────┬────────────┘   └────────────┬────────────┘
                    ▼                             │
       ┌─────────────────────────┐                │
       │ Stage 2: Parallel JEV   │                │
       │ Bounded Reranking (top3)│                │
       └────────────┬────────────┘                │
                    └──────────────┬──────────────┘
                                   │
       ┌───────────────────────────┴───────────────────────────┐
       │             Adaptive Compute Ladder Dispatch          │
       ├───────────────────────────┬───────────────────────────┤
       │ Tier 2: Violetto 1B (MPS) │  Tier 3: Codex Astra      │
       │ Native GPU Kernels (20t/s)│  (gpt-6-astra xhigh)      │
       └────────────┬──────────────┴─────────────┬─────────────┘
                    │ (If unverified / fail)     │
                    └──────────────►─────────────┘
                                   │
                                   ▼
       ┌───────────────────────────────────────────────────────┐
       │   Tier 4: Deterministic Verification Ensemble         │
       │   (CAS Equality, Z3 SMT Counterexamples, Spot-Checks) │
       └───────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
       ┌───────────────────────────────────────────────────────┐
       │   RRSI Evolution: Pareto Utility U = Q - λ_L*L - λ_C*C│
       │   (Regularized Self-Improvement per arXiv:2609.24972) │
       └───────────────────────────────────────────────────────┘
```

### Core Performance Pillars

1. **Speculative Fast Path & Early Exit (< 5ms)**
   - Probes SymPy CAS and Z3 SMT before routing or retrieval.
   - Exact polynomial factorization, roots of unity, divisibility, and congruences return in `< 5ms` with zero token cost.

2. **Two-Stage Hybrid Literature Retriever (Tantivy + JEV)**
   - **Stage 1 (Lexical Pruning)**: Quickwit Tantivy BM25 searches thousands of PageIndex nodes in `< 1ms`.
   - **Stage 2 (Semantic Reranking)**: Bounded concurrent JEV pool (`max_workers=8`) reranks top-25 down to top-3 in `< 250ms` (down from 30 seconds sequential).

3. **Native Apple Silicon MPS Violetto Engine (20 tok/s)**
   - Replaced manual Python token loop (`probs.cpu()`, `torch.cat`, per-token MPS-CPU sync barriers) with native PyTorch MPS `model.generate()`.
   - Shared persistent in-memory model worker eliminating repeated weights loading.

4. **Multi-Level SQLite Persistent Cache**
   - Thread-safe caching across routing decisions, retrieval hits, JEV relevance scores, solver outputs, and verification verdicts.
   - Cache hits return in `< 0.15ms`.

5. **Adaptive Compute Ladder & Conditional Escalation**
   - Routes by difficulty threshold and escalates adaptively: Tier 0 (CAS) $\to$ Tier 1 (MPS Violetto) $\to$ Tier 2 (Codex Astra).
   - Only invokes frontier models if cheaper tiers fail verification.

6. **Multi-Objective Pareto Utility for RRSI (arXiv:2609.24972)**
   - Objective function: $U = Q - \lambda_L L - \lambda_C C$ where $Q$ is verified quality, $L$ is normalized latency, and $C$ is compute cost.
   - Directly incentivizes the harness to discover low-latency, cost-effective reasoning paths.

7. **Interactive 3D Three.js UI**
   - WebGL force-directed 3D knowledge graph running at `http://localhost:8765`.
   - Real-time nanosecond telemetry (`route_ms`, `prefilter_ms`, `rerank_ms`, `solve_ms`, `verify_ms`, `total_ms`).
   - KaTeX-rendered mathematical inspection panel.

---

## 🚀 Quick Start

### 1. Requirements
- Python 3.10+
- Apple Silicon Mac (for local MPS Violetto inference) or CUDA device
- Codex CLI configured with `gpt-6-astra`
- TypeSafe API Key saved in `~/.typesafe_key`

### 2. Start 3D Web Dashboard
```bash
python3 server.py
# Open http://localhost:8765
```

### 3. CLI Exploration
```bash
# Elementary counting / arithmetic (routed to local Violetto)
python3 math_explorer.py explore "Count all integers n < 1000 such that n is divisible by 6, not 4, not 9."

# Deep algebraic number theory (routed to Codex Astra)
python3 math_explorer.py explore "Compute the Legendre symbol (11/13) using the Law of Quadratic Reciprocity." --engine codex_astra

# Index a new mathematical document
python3 math_explorer.py index corpus/algebraic_number_theory.md --name ANT_Reciprocity
```

---

## 📜 References
- **RRSI**: Regularized Recursive Self-Improvement of Agent Harnesses ([arXiv:2609.24972](https://arxiv.org/abs/2609.24972))
- **Violetto**: `paradigma-inc/limite-1b-violetto` ([HuggingFace](https://huggingface.co/paradigma-inc/limite-1b-violetto))
- **PageIndex**: Vectorless Hierarchical Document Indexing ([VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex))
