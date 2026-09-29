import pytest

from finance_lab import FinanceLab, Hypothesis


def test_episode_updates_direction_without_confusing_execution(tmp_path):
    ledger = tmp_path / "episodes.jsonl"
    hypothesis = Hypothesis("H1", "reversal", ("range",), "low", "imbalance recovers")
    lab = FinanceLab(ledger)
    lab.register(hypothesis)
    selection = lab.select_shadow("range", seed=1)
    record = lab.close_episode(
        episode_id="e1", hypothesis_id="H1", instrument="BTCUSDT", regime="range",
        side="long", predicted_return_bps=10, entry_price=100, exit_price=100.1,
        path=[{"offset_s": 1, "price": 99.9}, {"offset_s": 2, "price": 100.2}],
        fees_bps=8, slippage_bps=5, planned_stop_bps=20,
        execution_mode="shadow_maker", action_probability=selection["action_probability"],
        alternatives=["H1"], context={"snapshot_id": "s1"},
    )

    assert record["outcome"]["gross_bps"] > 0 > record["outcome"]["net_bps"]
    assert record["updates"]["posterior_after"]["mean"] > 0.5
    assert record["attribution"]["losses"]["direction"] == 0
    assert record["attribution"]["losses"]["execution"] > 1
    assert record["execution"]["evidence_class"] == "scenario"
    assert any(link["type"] == "supports" for link in lab.evidence_graph()["links"])

    replayed = FinanceLab(ledger)
    replayed.register(hypothesis)
    assert replayed.posterior("H1")["episodes"] == 1
    assert replayed.records[0]["updates"]["posterior_after"]["episodes"] == 1
    with pytest.raises(ValueError, match="duplicate episode"):
        replayed.close_episode(
            episode_id="e1", hypothesis_id="H1", instrument="BTCUSDT", regime="range",
            side="long", predicted_return_bps=1, entry_price=1, exit_price=1,
            path=[{"offset_s": 1, "price": 1}], fees_bps=0, slippage_bps=0,
            planned_stop_bps=0, execution_mode="shadow_taker", action_probability=1,
            alternatives=[], context={},
        )


def test_clustered_tail_risk_observations_are_contextual_and_equal_weight(tmp_path):
    lab = FinanceLab(tmp_path / "risk.jsonl", pooling=0)
    hypothesis = Hypothesis("H_RISK", "inventory_aging", ("train", "comparison"), "low",
                            "age >= 60m and adverse mark > 1%", target="tail_risk_15m")
    lab.register(hypothesis)
    lab.record_observation(observation_id="o1", hypothesis_id="H_RISK", unit_id="day-a",
                           regime="train", outcome=1 / 3, evidence_class="association",
                           source={"data_hash": "abc"}, context={"cluster_size": 3})
    lab.record_observation(observation_id="o2", hypothesis_id="H_RISK", unit_id="day-b",
                           regime="comparison", outcome=1, evidence_class="association",
                           source={"data_hash": "abc"}, context={"cluster_size": 20})

    assert lab.posterior("H_RISK")["episodes"] == 2
    assert lab.posterior("H_RISK", "train")["mean"] == pytest.approx((1 + 1 / 3) / 3)
    assert lab.posterior("H_RISK", "comparison")["mean"] == pytest.approx(2 / 3)
    assert lab.quality_diversity_archive() == {"tail_risk_15m:inventory_aging:low": "H_RISK"}
    assert lab.evidence_graph()["links"][-1]["type"] == "observes"
    with pytest.raises(ValueError, match="duplicate evidence unit"):
        lab.record_observation(observation_id="o3", hypothesis_id="H_RISK", unit_id="day-a",
                               regime="comparison", outcome=0, evidence_class="association",
                               source={"data_hash": "abc"}, context={})
