#!/usr/bin/env python3
"""Small shadow-trading lab: immutable episodes, local Bayes updates, no live orders."""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Hypothesis:
    id: str
    behavior: str
    regimes: tuple[str, ...]
    complexity: str
    expression: str
    target: str = "direction"


class FinanceLab:
    """Event-sourced playground; observed outcomes are not profitability claims."""

    def __init__(self, ledger_path: str | Path, pooling: float = 0.25):
        if not 0 <= pooling <= 1:
            raise ValueError("pooling must be between 0 and 1")
        self.ledger_path = Path(ledger_path)
        self.pooling = pooling
        self.hypotheses: dict[str, Hypothesis] = {}
        self.records = self._load_records()

    def _load_records(self) -> list[dict[str, Any]]:
        if not self.ledger_path.exists():
            return []
        with self.ledger_path.open(encoding="utf-8") as ledger:
            return [json.loads(line) for line in ledger if line.strip()]

    def register(self, hypothesis: Hypothesis) -> None:
        if hypothesis.id in self.hypotheses:
            raise ValueError(f"duplicate hypothesis: {hypothesis.id}")
        if hypothesis.complexity not in {"low", "medium", "high"}:
            raise ValueError("complexity must be low, medium, or high")
        self.hypotheses[hypothesis.id] = hypothesis

    @staticmethod
    def _outcome(record: dict[str, Any]) -> float | None:
        value = record["updates"].get("outcome", record["updates"].get("direction_success"))
        return None if value is None else float(value)

    def posterior(self, hypothesis_id: str, regime: str | None = None) -> dict[str, float | int]:
        hypothesis = self.hypotheses[hypothesis_id]
        records = self.records if regime is None else [r for r in self.records if r["context"]["regime"] == regime]
        own = [r for r in records if r["hypothesis"]["id"] == hypothesis_id]
        family = [r for r in records if r["hypothesis"]["behavior"] == hypothesis.behavior
                  and r["hypothesis"].get("target", "direction") == hypothesis.target]

        def counts(items: list[dict[str, Any]]) -> tuple[float, float, int]:
            outcomes = [value for r in items if (value := self._outcome(r)) is not None]
            return sum(outcomes), sum(1 - value for value in outcomes), len(outcomes)

        own_successes, own_failures, own_observations = counts(own)
        family_successes, family_failures, _ = counts(family)
        # ponytail: empirical partial pooling; replace with a real hierarchical model after enough episodes.
        alpha = 1 + own_successes + self.pooling * (family_successes - own_successes)
        beta = 1 + own_failures + self.pooling * (family_failures - own_failures)
        return {
            "alpha": alpha,
            "beta": beta,
            "mean": alpha / (alpha + beta),
            "episodes": own_observations,
        }

    def select_shadow(self, regime: str, seed: int = 0, draws: int = 512) -> dict[str, Any]:
        eligible = [h for h in self.hypotheses.values() if regime in h.regimes]
        if not eligible:
            return {"chosen": "no_trade", "action_probability": 1.0, "probabilities": {"no_trade": 1.0}}
        if draws < 1:
            raise ValueError("draws must be positive")

        parameters = {h.id: self.posterior(h.id, regime) for h in eligible}
        propensity_rng = random.Random(f"{seed}:propensity")
        counts = {h.id: 0 for h in eligible}
        for _ in range(draws):
            winner = max(eligible, key=lambda h: propensity_rng.betavariate(
                parameters[h.id]["alpha"], parameters[h.id]["beta"]))
            counts[winner.id] += 1
        probabilities = {key: value / draws for key, value in counts.items()}

        choice_rng = random.Random(f"{seed}:choice")
        chosen = max(eligible, key=lambda h: choice_rng.betavariate(
            parameters[h.id]["alpha"], parameters[h.id]["beta"])).id
        return {"chosen": chosen, "action_probability": probabilities[chosen], "probabilities": probabilities}

    def _append(self, record: dict[str, Any]) -> dict[str, Any]:
        self.records.append(record)
        record["updates"]["posterior_after"] = self.posterior(record["hypothesis"]["id"])
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        # ponytail: single-process append; add a DB transaction if concurrent writers appear.
        try:
            with self.ledger_path.open("a", encoding="utf-8") as ledger:
                ledger.write(json.dumps(record, sort_keys=True) + "\n")
                ledger.flush()
                os.fsync(ledger.fileno())
        except Exception:
            self.records.pop()
            raise
        return record

    def record_observation(
        self, *, observation_id: str, hypothesis_id: str, unit_id: str, regime: str,
        outcome: float, evidence_class: str, source: dict[str, Any], context: dict[str, Any],
    ) -> dict[str, Any]:
        """Record one equal-weight evidence unit; fractional outcomes support clustered rates."""
        if observation_id in {r["episode_id"] for r in self.records}:
            raise ValueError(f"duplicate observation: {observation_id}")
        if hypothesis_id not in self.hypotheses:
            raise KeyError(hypothesis_id)
        if not unit_id or not 0 <= outcome <= 1:
            raise ValueError("unit_id is required and outcome must be in [0, 1]")
        if any(r.get("evidence", {}).get("unit_id") == unit_id and r["hypothesis"]["id"] == hypothesis_id
               for r in self.records):
            raise ValueError(f"duplicate evidence unit: {unit_id}")
        if evidence_class not in {"observed", "association", "scenario", "hypothesis", "rejected"}:
            raise ValueError("invalid evidence_class")
        hypothesis = self.hypotheses[hypothesis_id]
        before = self.posterior(hypothesis_id)
        return self._append({
            "episode_id": observation_id, "record_type": "observation", "closed_at": time.time(),
            "instrument": context.get("instrument", "multi_asset"),
            "context": {"regime": regime, "features": context},
            "hypothesis": {"id": hypothesis.id, "behavior": hypothesis.behavior,
                           "complexity": hypothesis.complexity, "expression": hypothesis.expression,
                           "target": hypothesis.target},
            "evidence": {"unit_id": unit_id, "class": evidence_class, "source": source},
            "updates": {"outcome": outcome, "posterior_before": before, "structural_mutations": []},
        })

    def close_episode(
        self,
        *,
        episode_id: str,
        hypothesis_id: str,
        instrument: str,
        regime: str,
        side: str,
        predicted_return_bps: float,
        entry_price: float,
        exit_price: float,
        path: list[dict[str, float]],
        fees_bps: float,
        slippage_bps: float,
        planned_stop_bps: float,
        execution_mode: str,
        action_probability: float,
        alternatives: list[str],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        if episode_id in {r["episode_id"] for r in self.records}:
            raise ValueError(f"duplicate episode: {episode_id}")
        if hypothesis_id not in self.hypotheses:
            raise KeyError(hypothesis_id)
        if side not in {"long", "short"} or entry_price <= 0 or exit_price <= 0:
            raise ValueError("side must be long/short and prices must be positive")
        if not path or any(point["price"] <= 0 for point in path):
            raise ValueError("path must contain positive prices")
        if any(path[i]["offset_s"] > path[i + 1]["offset_s"] for i in range(len(path) - 1)):
            raise ValueError("path must be chronological")
        if min(fees_bps, slippage_bps, planned_stop_bps) < 0 or not 0 <= action_probability <= 1:
            raise ValueError("costs/stop must be nonnegative and action_probability must be in [0, 1]")
        if execution_mode not in {"observed_maker", "observed_taker", "shadow_maker", "shadow_taker"}:
            raise ValueError("execution_mode must state observed/shadow and maker/taker")

        hypothesis = self.hypotheses[hypothesis_id]
        sign = 1 if side == "long" else -1
        excursions = [sign * (point["price"] - entry_price) / entry_price * 10_000 for point in path]
        gross_bps = sign * (exit_price - entry_price) / entry_price * 10_000
        mfe_bps = max(0.0, max(excursions))
        mae_bps = max(0.0, -min(excursions))
        net_bps = gross_bps - fees_bps - slippage_bps
        direction_success = None if gross_bps == 0 else gross_bps > 0
        before = self.posterior(hypothesis_id)

        losses = {
            "direction": None if direction_success is None else float(not direction_success),
            "magnitude": abs(predicted_return_bps - gross_bps) / max(abs(predicted_return_bps), 1.0),
            "timing": None,  # Needs causal subtrade decisions, not OHLC extrema.
            "risk": max(0.0, mae_bps - planned_stop_bps) / max(planned_stop_bps, 1.0),
            "execution": (fees_bps + slippage_bps) / max(abs(gross_bps), 1.0),
            "exit": max(0.0, mfe_bps - max(gross_bps, 0.0)) / max(mfe_bps, 1.0),
        }
        challenged = [name for name, value in losses.items() if value is not None and value > 0.5]
        record = {
            "episode_id": episode_id,
            "closed_at": time.time(),
            "instrument": instrument,
            "context": {"regime": regime, "features": context},
            "hypothesis": {
                "id": hypothesis.id,
                "behavior": hypothesis.behavior,
                "complexity": hypothesis.complexity,
                "expression": hypothesis.expression,
                "target": hypothesis.target,
            },
            "decision": {
                "side": side,
                "predicted_return_bps": predicted_return_bps,
                "alternatives": alternatives,
                "action_probability": action_probability,
            },
            "execution": {
                "mode": execution_mode,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "fees_bps": fees_bps,
                "slippage_bps": slippage_bps,
                "evidence_class": "observed" if execution_mode.startswith("observed_") else "scenario",
            },
            "outcome": {
                "gross_bps": gross_bps,
                "net_bps": net_bps,
                "mfe_bps": mfe_bps,
                "mae_bps": mae_bps,
                "time_to_mfe_s": path[excursions.index(max(excursions))]["offset_s"],
                "time_to_mae_s": path[excursions.index(min(excursions))]["offset_s"],
                "path": path,
            },
            "attribution": {"losses": losses, "challenged_components": challenged},
            "updates": {
                "direction_success": direction_success,
                "outcome": direction_success,
                "posterior_before": before,
                "structural_mutations": [],
            },
        }
        return self._append(record)

    def quality_diversity_archive(self) -> dict[str, str]:
        archive: dict[str, str] = {}
        for hypothesis in self.hypotheses.values():
            cell = f"{hypothesis.target}:{hypothesis.behavior}:{hypothesis.complexity}"
            incumbent = archive.get(cell)
            if incumbent is None or self.posterior(hypothesis.id)["mean"] > self.posterior(incumbent)["mean"]:
                archive[cell] = hypothesis.id
        return archive

    def evidence_graph(self) -> dict[str, list[dict[str, Any]]]:
        """Derive the graph from the append-only ledger; no second source of truth."""
        nodes: dict[str, dict[str, Any]] = {}
        links: list[dict[str, Any]] = []
        for hypothesis in self.hypotheses.values():
            nodes[hypothesis.id] = {"id": hypothesis.id, "type": "hypothesis", "data": self.posterior(hypothesis.id)}
            for regime in hypothesis.regimes:
                regime_id = f"regime:{regime}"
                nodes[regime_id] = {"id": regime_id, "type": "regime"}
                links.append({"source": hypothesis.id, "target": regime_id, "type": "valid_under"})
        for record in self.records:
            episode_id = f"episode:{record['episode_id']}"
            nodes[episode_id] = {"id": episode_id, "type": record.get("record_type", "episode"),
                                 "data": record.get("outcome", record.get("evidence"))}
            success = self._outcome(record)
            relation = "observes" if record.get("record_type") == "observation" or success is None else ("supports" if success else "contradicts")
            links.append({"source": episode_id, "target": record["hypothesis"]["id"], "type": relation})
        return {"nodes": list(nodes.values()), "links": links}


def demo(ledger_path: str | Path) -> dict[str, Any]:
    lab = FinanceLab(ledger_path)
    lab.register(Hypothesis(
        "H_LIQ_REV", "liquidation_reversal", ("high_vol_mean_reversion",), "low",
        "liq_below > Q90 and toxic_flow < 0 and book_recovers",
    ))
    lab.register(Hypothesis(
        "H_OFI_CONT", "flow_continuation", ("high_vol_mean_reversion", "trend"), "low",
        "signed_ofi > Q80 and spread < Q50",
    ))
    selection = lab.select_shadow("high_vol_mean_reversion", seed=7)
    episode = lab.close_episode(
        episode_id=f"demo-{time.time_ns()}", hypothesis_id=selection["chosen"], instrument="BTCUSDT",
        regime="high_vol_mean_reversion", side="long", predicted_return_bps=18,
        entry_price=100_000, exit_price=100_150,
        path=[{"offset_s": 30, "price": 99_900}, {"offset_s": 120, "price": 100_220},
              {"offset_s": 300, "price": 100_150}],
        fees_bps=4, slippage_bps=2, planned_stop_bps=25,
        execution_mode="shadow_taker", action_probability=selection["action_probability"],
        alternatives=list(selection["probabilities"]), context={"source": "synthetic_demo"},
    )
    graph = lab.evidence_graph()
    return {"selection": selection, "episode": episode, "archive": lab.quality_diversity_archive(),
            "graph": {"nodes": len(graph["nodes"]), "links": len(graph["links"])}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shadow Bayesian finance playground; never places orders")
    parser.add_argument("--ledger", help="Optional persistent JSONL ledger")
    args = parser.parse_args()
    if args.ledger:
        print(json.dumps(demo(args.ledger), indent=2))
    else:
        with tempfile.TemporaryDirectory() as directory:
            print(json.dumps(demo(Path(directory) / "episodes.jsonl"), indent=2))
